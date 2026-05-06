import asyncio
import base64
import io
import logging
import os
import re
import time
from dataclasses import dataclass
from multiprocessing import Manager, Process, set_start_method
from typing import List, Optional, Tuple, cast

import pymupdf
import torch
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from PIL import Image, UnidentifiedImageError

from ...type import FileDescriptor, MultimodalSample
from ..utils import clean_image, clean_text
from .base import Processor, ProcessorConfig

IMG_REGEX = r"!\[\]\(_page_\d+_[A-Za-z0-9_]+\.(jpeg|jpg|png|gif)\)"

PDF_BACKEND_MARKER = "marker"
PDF_BACKEND_PYMUPDF4LLM = "pymupdf4llm"
PDF_BACKEND_MISTRAL = "mistral"


@dataclass
class PDFProcessorConfig(ProcessorConfig):
    """Configuration for :class:`PDFProcessor`.

    Inherits all fields from :class:`ProcessorConfig`.

    Attributes:
        backend: Which PDF extraction engine to use. One of:

            * ``"marker"`` (default) - marker-pdf with GPU-accelerated OCR;
              slower but supports scanned documents.
            * ``"pymupdf4llm"`` - LLM-friendly markdown extraction via
              `pymupdf4llm <https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/>`_;
              much faster, but does **not** OCR scanned documents.
            * ``"mistral"`` - cloud OCR via Mistral's ``mistral-ocr-latest``
              endpoint. Sends each PDF to the API; reads the API key from the
              ``MISTRAL_API_KEY`` environment variable (loaded from a ``.env``
              file at the project root by :mod:`mmore.run_process`).

        mistral_model: Mistral OCR model name (default
            ``"mistral-ocr-latest"``).
        mistral_max_calls_per_second: Outbound rate limit when batch-processing
            with the Mistral backend.
        mistral_max_retries: Number of retries (with exponential backoff) on
            transient Mistral API failures.

    Example YAML (inside the ``processors:`` block of your config file)::

        processors:
          PDFProcessor:
            backend: mistral
            mistral_model: mistral-ocr-latest
    """

    backend: str = PDF_BACKEND_MARKER
    mistral_model: str = "mistral-ocr-latest"
    mistral_max_calls_per_second: int = 5
    mistral_max_retries: int = 3


class PDFProcessor(Processor):
    CONFIG_CLASS = PDFProcessorConfig
    artifact_dict = None

    def __init__(self, config=None):
        super().__init__(config=config or PDFProcessorConfig())
        self.converter = None
        self._mistral_client = None

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() == ".pdf"

    @staticmethod
    def load_models(disable_image_extraction: bool = False):
        if PDFProcessor.artifact_dict is None:
            PDFProcessor.artifact_dict = create_model_dict()

        marker_config = {
            "disable_image_extraction": disable_image_extraction,
            "languages": None,
            "use_llm": False,
            "disable_multiprocessing": False,
            "paginate_output": True,
        }
        config_parser = ConfigParser(marker_config)
        converter = PdfConverter(
            artifact_dict=PDFProcessor.artifact_dict,
            config=config_parser.generate_config_dict(),
        )

        converter.initialize_processors(list(converter.default_processors))

        return converter

    # overwriting the process_batch
    def process_batch(
        self, files_paths: List[str], fast_mode: bool = False, num_workers: int = 1
    ) -> List[MultimodalSample]:
        backend = getattr(self.config, "backend", PDF_BACKEND_MARKER)

        if backend == PDF_BACKEND_MISTRAL and not fast_mode:
            return self._process_batch_mistral(files_paths)

        if backend == PDF_BACKEND_PYMUPDF4LLM and not fast_mode:
            results: List[MultimodalSample] = []
            for file_path in files_paths:
                try:
                    results.append(self.process(file_path))
                except Exception as e:
                    logging.error(f"Failed to process {file_path}: {str(e)}")
            return results

        if fast_mode:  # No GPU available - fallback to default
            return super().process_batch(files_paths, fast_mode, num_workers)
        else:
            if not torch.cuda.is_available():
                num_gpus = 1
            else:
                num_gpus = torch.cuda.device_count()

            # 1 GPU available or length of files_paths is less than 10 we just do single-GPU
            if num_gpus == 1 or len(files_paths) < 10:
                if self.converter is None:
                    self.converter = PDFProcessor.load_models(
                        disable_image_extraction=not self.config.extract_images
                    )

                results = []
                for file_path in files_paths:
                    try:
                        res = self.process(file_path)
                        results.append(res)
                    except Exception as e:
                        logging.error(f"Failed to process {file_path}: {str(e)}")

                return results
            else:  # Multiple GPUs available
                batches = self._split_files(files_paths, num_gpus)

                try:
                    set_start_method("spawn", force=True)
                except RuntimeError:
                    pass

                manager = Manager()
                output_queue = manager.Queue()
                error_queue = manager.Queue()
                processes = []

                for i, batch in enumerate(batches):
                    if not batch:
                        continue
                    gpu_id = i % num_gpus
                    p = Process(
                        target=self._process_parallel,
                        args=(
                            batch,
                            gpu_id,
                            self.config.extract_images,
                            output_queue,
                            error_queue,
                        ),
                    )
                    processes.append(p)
                    p.start()

                results = []

                while any(p.is_alive() for p in processes):
                    if not error_queue.empty():
                        error = error_queue.get()
                        raise RuntimeError(f"Child process failed: {error}")
                    while not output_queue.empty():
                        results.extend(output_queue.get())

                while not output_queue.empty():
                    results.extend(output_queue.get())

                if not error_queue.empty():
                    error = error_queue.get()
                    raise RuntimeError(f"Child process failed: {error}")

                return results

    # Regex matching marker page separators: \n\n{page_id}----...\n\n
    _PAGE_SEP_RE = re.compile(r"\n\n\{(\d+)\}-{3,}\n\n")

    def process(self, file_path: str) -> MultimodalSample:
        backend = getattr(self.config, "backend", PDF_BACKEND_MARKER)
        if backend == PDF_BACKEND_PYMUPDF4LLM:
            return self._process_pymupdf4llm(file_path)
        if backend == PDF_BACKEND_MISTRAL:
            return self._process_mistral(file_path)
        if backend != PDF_BACKEND_MARKER:
            logging.warning(
                f"Unknown PDF backend '{backend}', falling back to '{PDF_BACKEND_MARKER}'."
            )

        if self.converter is None:
            self.converter = PDFProcessor.load_models(
                disable_image_extraction=not self.config.extract_images
            )

        rendered = self.converter(file_path)
        text, _, images = text_from_rendered(rendered)
        text = re.sub(str(IMG_REGEX), self.config.attachment_tag, cast(str, text))
        images = list(images.values())

        paragraph_starts, text = self._parse_pagination(cast(str, text))

        metadata = {"file_path": file_path}
        if paragraph_starts:
            metadata["paragraph_starts"] = paragraph_starts

        return self.create_sample([text], images, metadata)

    def _process_pymupdf4llm(self, file_path: str) -> MultimodalSample:
        try:
            import pymupdf4llm
        except ImportError as e:
            raise ImportError(
                "pymupdf4llm is required when PDFProcessor.backend='pymupdf4llm'. "
                "Install it with `pip install pymupdf4llm` or reinstall the "
                "`process` extra: `pip install -e .[process]`."
            ) from e

        page_chunks = pymupdf4llm.to_markdown(
            file_path,
            page_chunks=True,
            write_images=False,
            embed_images=False,
            show_progress=False,
        )

        all_text_parts: List[str] = []
        embedded_images: List[Image.Image] = []
        paragraph_starts: List[Tuple[int, int, int]] = []
        current_position = 0

        pdf_doc = pymupdf.Document(file_path) if self.config.extract_images else None

        try:
            for chunk in page_chunks:
                chunk_metadata = chunk.get("metadata", {}) or {}
                page_num = int(chunk_metadata.get("page", 0))
                text = chunk.get("text", "") or ""

                if text.strip():
                    para_idx = 0
                    offset_in_page = 0
                    for segment in text.split("\n\n"):
                        if segment.strip():
                            paragraph_starts.append(
                                (current_position + offset_in_page, page_num, para_idx)
                            )
                            para_idx += 1
                        offset_in_page += len(segment) + 2

                    all_text_parts.append(text)
                    current_position += len(text)

                if self.config.extract_images and pdf_doc is not None:
                    page_index = page_num - 1 if page_num > 0 else page_num
                    if 0 <= page_index < pdf_doc.page_count:
                        page = pdf_doc[page_index]
                        for img_info in page.get_images(full=False):
                            image = self._extract_pymupdf_image(pdf_doc, img_info[0])
                            if image and clean_image(image):
                                embedded_images.append(image)
                                attachment_text = self.config.attachment_tag
                                all_text_parts.append(attachment_text)
                                current_position += len(attachment_text)
        finally:
            if pdf_doc is not None:
                pdf_doc.close()

        paragraph_starts.append((current_position, -1, -1))

        metadata = {
            "file_path": file_path,
            "paragraph_starts": paragraph_starts,
            "document_type": "pdf",
        }

        full_text = "".join(all_text_parts)
        return self.create_sample([full_text], embedded_images, metadata)

    @staticmethod
    def _extract_pymupdf_image(pdf_doc, xref) -> Optional[Image.Image]:
        try:
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image.get("image")
            if image_bytes is None:
                logging.error(f"No image data found for xref {xref}")
                return None
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except KeyError as e:
            logging.error(f"KeyError while extracting image for xref {xref}: {e}")
            return None
        except UnidentifiedImageError as e:
            logging.error(
                f"UnidentifiedImageError: could not identify image for xref {xref}: {e}"
            )
            return None
        except Exception as e:
            logging.error(
                f"Unexpected error while extracting image for xref {xref}: {e}"
            )
            return None

    def _get_mistral_client(self):
        if self._mistral_client is not None:
            return self._mistral_client
        try:
            from mistralai import Mistral
        except ImportError as e:
            raise ImportError(
                "mistralai is required when PDFProcessor.backend='mistral'. "
                "Install it with `pip install mistralai` or reinstall the "
                "`process` extra: `pip install -e .[process]`."
            ) from e

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "Mistral backend requires the MISTRAL_API_KEY environment "
                "variable. Add it to your .env file at the project root."
            )
        self._mistral_client = Mistral(api_key=api_key)
        return self._mistral_client

    @staticmethod
    def _encode_pdf_b64(pdf_path: str) -> str:
        with open(pdf_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _decode_b64_image(image_base64: str) -> Optional[Image.Image]:
        try:
            if image_base64.startswith("data:"):
                image_base64 = image_base64.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
        except Exception as e:
            logging.error(f"Error decoding image from base64: {e}")
            return None

    def _mistral_call_with_retry(self, func, *args, **kwargs):
        max_retries = self.config.mistral_max_retries
        for i in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if i == max_retries:
                    logging.error(
                        f"Mistral API call failed after {max_retries + 1} attempts: {e}"
                    )
                    raise
                wait_time = min(2**i, 60)
                time.sleep(wait_time)

    def _process_batch_mistral(
        self, files_paths: List[str]
    ) -> List[MultimodalSample]:
        rate = max(1, self.config.mistral_max_calls_per_second)
        min_interval = 1.0 / rate

        async def run() -> List[MultimodalSample]:
            loop = asyncio.get_event_loop()
            results: List[MultimodalSample] = []
            tasks: List[Tuple[asyncio.Task, str]] = []

            for file_path in files_paths:
                task = loop.run_in_executor(None, self._process_mistral, file_path)
                tasks.append((cast(asyncio.Task, task), file_path))
                await asyncio.sleep(min_interval)

            for task, file_path in tasks:
                try:
                    results.append(await task)
                except Exception as e:
                    logging.error(f"Failed to process {file_path}: {e}")
            return results

        return asyncio.run(run())

    def _process_mistral(self, file_path: str) -> MultimodalSample:
        client = self._get_mistral_client()
        b64 = self._encode_pdf_b64(file_path)

        ocr_response = self._mistral_call_with_retry(
            client.ocr.process,
            model=self.config.mistral_model,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{b64}",
            },
            include_image_base64=True,
        )

        all_text_parts: List[str] = []
        embedded_images: List[Image.Image] = []
        attachment_tag = self.config.attachment_tag

        pages = getattr(ocr_response, "pages", None)
        if pages is None and isinstance(ocr_response, dict):
            pages = ocr_response.get("pages", [])

        for page in pages or []:
            text = getattr(page, "markdown", None)
            page_images = getattr(page, "images", None)
            if text is None and isinstance(page, dict):
                text = page.get("markdown", "")
                page_images = page.get("images", [])

            for img in page_images or []:
                img_id = getattr(img, "id", None) or img.get("id")
                img_b64 = getattr(img, "image_base64", None) or img.get("image_base64")
                if img_id and text:
                    text = text.replace(img_id, attachment_tag)
                if self.config.extract_images and img_b64:
                    decoded = self._decode_b64_image(img_b64)
                    if decoded is not None:
                        embedded_images.append(decoded)

            if text:
                all_text_parts.append(text)

        return self.create_sample(
            all_text_parts,
            embedded_images,
            {"file_path": file_path, "document_type": "pdf"},
        )

    @classmethod
    def _parse_pagination(
        cls, text: str
    ) -> Tuple[
        List[Tuple[int, int, int]],
        str,
    ]:
        """Parse marker pagination separators to build paragraph_starts,
        then strip the separators from the text."""
        separators = list(cls._PAGE_SEP_RE.finditer(text))
        if not separators:
            return [], text

        page_texts: List[Tuple[int, str]] = []  # (page_id, page_content)
        prev_end = 0
        for match in separators:
            page_id = int(match.group(1))
            page_content = text[prev_end : match.start()]
            page_texts.append((page_id, page_content))
            prev_end = match.end()
        trailing = text[prev_end:]
        if trailing.strip():
            last_page_id = int(separators[-1].group(1)) + 1
            page_texts.append((last_page_id, trailing))

        paragraph_starts: List[Tuple[int, int, int]] = []
        current_position = 0

        for page_id, page_content in page_texts:
            para_idx = 0
            offset_in_page = 0
            for segment in page_content.split("\n\n"):
                if segment.strip():
                    paragraph_starts.append(
                        (current_position + offset_in_page, page_id, para_idx)
                    )
                    para_idx += 1
                offset_in_page += len(segment) + 2

            current_position += len(page_content)

        paragraph_starts.append((current_position, -1, -1))

        clean_text = "".join(content for _, content in page_texts)

        return paragraph_starts, clean_text

    def process_fast(self, file_path: str) -> MultimodalSample:
        pdf_doc = pymupdf.Document(file_path)
        all_text_parts = []
        embedded_images = []
        paragraph_starts: List[
            Tuple[int, int, int]
        ] = []  # (char_offset, page_num, para_index)
        current_position = 0

        def _extract_images(pdf_doc, xref) -> Optional[Image.Image]:
            try:
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image.get("image")

                if image_bytes is None:
                    logging.error(f"No image data found for xref {xref}")

                return Image.open(io.BytesIO(image_bytes)).convert("RGB")

            except KeyError as e:
                logging.error(f"KeyError while extracting image: {e}")
                return None

            except UnidentifiedImageError as e:
                logging.error(
                    f"UnidentifiedImageError: Could not identify image file for xref {xref}: {e}"
                )
                return None

            except Exception as e:
                logging.error(
                    f"Unexpected error while extracting image for xref {xref}: {e}"
                )
                return None

        for page_num, page in enumerate(pdf_doc):
            text = clean_text(page.get_text())  # type: ignore[attr-defined]

            if text.strip():
                para_idx = 0
                offset_in_page = 0
                for segment in text.split("\n\n"):
                    if segment.strip():
                        paragraph_starts.append(
                            (current_position + offset_in_page, page_num, para_idx)
                        )
                        para_idx += 1
                    offset_in_page += len(segment) + 2  # +2 for the "\n\n" separator

                all_text_parts.append(text)
                current_position += len(text)

            if self.config.extract_images:
                for img_info in page.get_images(full=False):
                    image = _extract_images(pdf_doc, img_info[0])
                    if image and clean_image(image):
                        # clean image filters images below size 512x512 and variance below 100, these are defaults and can be changed
                        embedded_images.append(image)
                        attachment_text = self.config.attachment_tag
                        all_text_parts.append(attachment_text)
                        current_position += len(attachment_text)

        paragraph_starts.append((current_position, -1, -1))
        metadata = {
            "file_path": file_path,
            "paragraph_starts": paragraph_starts,
            "document_type": "pdf",
        }

        full_text = "".join(all_text_parts)
        return self.create_sample([full_text], embedded_images, metadata)

    # Functions for parallelizing across GPUs
    def _split_files(self, files_paths, num_batches):
        file_sizes = [(file, self.get_file_size(file)) for file in files_paths]
        sorted_files = sorted(file_sizes, key=lambda x: x[1], reverse=True)

        batches = [[] for _ in range(num_batches)]
        batch_sizes = [0] * num_batches

        for file, size in sorted_files:
            min_index = batch_sizes.index(min(batch_sizes))
            batches[min_index].append(file)
            batch_sizes[min_index] += size

        batches = [batch for batch in batches if batch]
        return batches

    def _process_parallel(
        self, files_paths, gpu_id, extract_images: bool, output_queue, error_queue
    ):
        try:
            torch.cuda.set_device(gpu_id)

            if PDFProcessor.artifact_dict is None:
                PDFProcessor.artifact_dict = create_model_dict()

            marker_config = {
                "disable_image_extraction": not extract_images,
                "languages": None,
                "use_llm": False,
                "disable_multiprocessing": False,
                "device": f"cuda:{gpu_id}",
            }

            config_parser = ConfigParser(marker_config)
            self.converter = PdfConverter(
                artifact_dict=PDFProcessor.artifact_dict,
                config=config_parser.generate_config_dict(),
            )

            batch_results = []
            for file in files_paths:
                try:
                    result = self.process(file)
                    batch_results.append(result)
                except Exception as e:
                    logging.error(f"Failed to process {file}: {str(e)}")
                    batch_results.append(None)  # handle partial failures

            output_queue.put(batch_results)

        except Exception as e:
            error_queue.put(f"GPU {gpu_id} failed: {str(e)}")
            raise e
        finally:
            torch.cuda.empty_cache()
            if hasattr(self, "converter"):
                del self.converter
