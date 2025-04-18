import pypdfium2
import logging
import fitz  # PyMuPDF
import io
import logging
from PIL import Image, UnidentifiedImageError
from typing import List
from src.mmore.type import FileDescriptor, MultimodalSample
from .base import Processor, ProcessorConfig
from src.mmore.process.utils import clean_text, clean_image

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
import re
from multiprocessing import Process, Queue, set_start_method, Manager
import torch
import time


IMG_REGEX = "!\[\]\(_page_\d+_[A-Za-z0-9_]+\.(jpeg|jpg|png|gif)\)"

class PDFProcessor(Processor):
    def __init__(self, config=None):
        super().__init__(config=config or ProcessorConfig())

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() == ".pdf"

    # overwriting the process_batch
    def process_batch(self, files, fast_mode, num_workers):
        if fast_mode: # No GPU available - fallback to default
            return super().process_batch(files, fast_mode, num_workers)
        else:
            if not torch.cuda.is_available():
                num_gpus = 1
            else:
                num_gpus = torch.cuda.device_count()

            if num_gpus == 1 or len(files) < 10: # 1 GPU available or length of files is less than 10 we just do single-GPU
                marker_config = {
                    "disable_image_extraction": not self.config.custom_config.get("extract_images", True),
                    "languages": None,
                    "use_llm": False,
                    "disable_multiprocessing": False,
                }
                config_parser = ConfigParser(marker_config)
                self.converter = PdfConverter(
                    artifact_dict=create_model_dict(),
                    config=config_parser.generate_config_dict()
                )
                results = []
                for file_path in files:
                    try:
                        res = self.process(file_path)
                        results.append(res)
                    except Exception as e:
                        logging.error(f"Failed to process {file_path}: {str(e)}")

                return results
            else: # Multiple GPUs available
                batches = self._split_files(files, num_gpus)

                try:
                    set_start_method('spawn', force=True)
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
                        args=(batch, gpu_id, self.config.custom_config, output_queue, error_queue)
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

    def process(self, file_path: str) -> List[MultimodalSample]:
        rendered = self.converter(file_path)
        text, _, images = text_from_rendered(rendered)
        text = re.sub(IMG_REGEX, "<attachment>", text)
        images = images.values()
        return self.create_sample([text], images, file_path)

    def process_fast(self, file_path) -> MultimodalSample:
        pdf_doc = fitz.open(file_path)
        all_text = []
        embedded_images = []

        def _extract_images(pdf_doc, xref) -> Image.Image:
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
                logging.error(f"UnidentifiedImageError: Could not identify image file for xref {xref}: {e}")
                return None

            except Exception as e:
                logging.error(f"Unexpected error while extracting image for xref {xref}: {e}")
                return None

        for page in pdf_doc:
            text = clean_text(page.get_text())
            if text.strip():
                all_text.append(text)

            if self.config.custom_config.get("extract_images", True):
                for img_info in page.get_images(full=False):
                    image = _extract_images(pdf_doc, img_info[0])
                    if clean_image(image):  # clean image filters images below size 512x512 and variance below 100, these are defaults and can be changed
                        embedded_images.append(image)
                        all_text.append(self.config.attachment_tag)
            else:
                embedded_images = []

        return self.create_sample(all_text, embedded_images, file_path)

    ### Functions for parallelizing across GPUs
    def _split_files(self, files, num_batches):
        file_sizes = [(file, self.get_file_size(file)) for file in files]
        sorted_files = sorted(file_sizes, key=lambda x: x[1], reverse=True)

        batches = [[] for _ in range(num_batches)]
        batch_sizes = [0] * num_batches

        for file, size in sorted_files:
            min_index = batch_sizes.index(min(batch_sizes))
            batches[min_index].append(file)
            batch_sizes[min_index] += size

        batches = [batch for batch in batches if batch]
        return batches

    def _process_parallel(self, files, gpu_id, config_custom, output_queue, error_queue):
        try:
            torch.cuda.set_device(gpu_id)

            marker_config = {
                "disable_image_extraction": not config_custom.get("extract_images", True),
                "languages": None,
                "use_llm": False,
                "disable_multiprocessing": False,
                "device": f"cuda:{gpu_id}"
            }

            config_parser = ConfigParser(marker_config)
            self.converter = PdfConverter(
                artifact_dict=create_model_dict(),
                config=config_parser.generate_config_dict()
            )

            batch_results = []
            for file in files:
                try:
                    result = self.process(file)
                    batch_results.append(result)
                except Exception as e:
                    logging.error(f"Failed to process {file}: {str(e)}")
                    batch_results.append(None)  # handle partial failures

            output_queue.put(batch_results)

        except Exception as e:
            error_queue.put(f"GPU {gpu_id} failed: {str(e)}")
            raise
        finally:
            torch.cuda.empty_cache()
            if hasattr(self, 'converter'):
                del self.converter