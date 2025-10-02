import base64
import io
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Process, set_start_method
from typing import List, Optional

import fitz  # PyMuPDF
import requests
import torch
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForVision2Seq, AutoProcessor

from ...type import FileDescriptor, MultimodalSample
from ..utils import clean_image, clean_text
from .base import Processor, ProcessorConfig

IMG_REGEX = r"!\[\]\(_page_\d+_[A-Za-z0-9_]+\.(jpeg|jpg|png|gif)\)"


class PDFProcessor(Processor):
    artifact_dict = None

    def __init__(self, config=None):
        super().__init__(config=config or ProcessorConfig())
        self.converter = None
        self.image_analyzer = None
        self.analyze_images = self.config.custom_config.get("analyze_images", False)
        self.image_analyzer_type = self.config.custom_config.get(
            "image_analyzer_type", "smoldocling"
        )

        # Initialize image analyzer if needed
        if self.analyze_images:
            self._init_image_analyzer()

    def _init_image_analyzer(self):
        """Initialize the appropriate image analyzer based on configuration"""
        if self.image_analyzer_type == "smoldocling" and self.analyze_images:
            self.image_analyzer = SmolDoclingImageAnalyzer()
        elif self.image_analyzer_type == "mistral" and self.analyze_images:
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                logging.warning(
                    "MISTRAL_API_KEY environment variable not set. MistralOCR will not work."
                )
            self.image_analyzer = MistralOCRImageAnalyzer(api_key=api_key)
        else:
            self.image_analyzer = None

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
                        disable_image_extraction=not self.config.custom_config.get(
                            "extract_images", True
                        )
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
                            self.config.custom_config,
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

    def process(self, file_path: str) -> MultimodalSample:
        if self.converter is None:
            self.converter = PDFProcessor.load_models(
                disable_image_extraction=not self.config.custom_config.get(
                    "extract_images", True
                )
            )

        rendered = self.converter(file_path)
        text, _, images = text_from_rendered(rendered)
        text = re.sub(IMG_REGEX, "<attachment>", text)

        # If image analysis is enabled, analyze the images
        if self.analyze_images and self.image_analyzer and images:
            image_texts = self._analyze_images(images.values())
            # Combine original text with image analysis results
            for img_text in image_texts:
                if img_text and img_text.strip():
                    text += f"\n\nImage content: {img_text}"

        return self.create_sample([text], list(images.values()), file_path)

    def _analyze_images(self, images):
        """Analyze images using the configured image analyzer"""
        if not self.image_analyzer:
            return []

        return self.image_analyzer.analyze_batch(images)

    def process_fast(self, file_path: str) -> MultimodalSample:
        pdf_doc = fitz.open(file_path)
        all_text = []
        embedded_images = []

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

        for page in pdf_doc:
            text = clean_text(page.get_text())  # type: ignore[attr-defined]
            if text.strip():
                all_text.append(text)

            if self.config.custom_config.get("extract_images", True):
                page_images = []
                for img_info in page.get_images(full=False):
                    image = _extract_images(pdf_doc, img_info[0])
                    if image and clean_image(image):
                        # clean image filters images below size 512x512 and variance below 100, these are defaults and can be changed
                        embedded_images.append(image)
                        page_images.append(image)
                        all_text.append(self.config.attachment_tag)

                # If image analysis is enabled, analyze the images
                if self.analyze_images and self.image_analyzer and page_images:
                    image_texts = self._analyze_images(page_images)
                    # Add image analysis results to the text
                    for img_text in image_texts:
                        if img_text and img_text.strip():
                            all_text.append(f"Image content: {img_text}")
            else:
                embedded_images = []

        return self.create_sample(all_text, embedded_images, file_path)

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
        self, files_paths, gpu_id, config_custom, output_queue, error_queue
    ):
        try:
            torch.cuda.set_device(gpu_id)

            # Pass along the image analysis configuration
            analyze_images = config_custom.get("analyze_images", False)
            image_analyzer_type = config_custom.get(
                "image_analyzer_type", "smoldocling"
            )

            if PDFProcessor.artifact_dict is None:
                PDFProcessor.artifact_dict = create_model_dict()

            marker_config = {
                "disable_image_extraction": not config_custom.get(
                    "extract_images", True
                ),
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

            # Initialize image analyzer if needed
            if analyze_images:
                if image_analyzer_type == "smoldocling":
                    self.image_analyzer = SmolDoclingImageAnalyzer(
                        device=f"cuda:{gpu_id}"
                    )
                elif image_analyzer_type == "mistral":
                    api_key = os.environ.get("MISTRAL_API_KEY")
                    self.image_analyzer = MistralOCRImageAnalyzer(api_key=api_key)
                else:
                    self.image_analyzer = None
            else:
                self.image_analyzer = None

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
            if hasattr(self, "image_analyzer"):
                del self.image_analyzer


class SmolDoclingImageAnalyzer:
    """Image analyzer using SmolDocling for OCR and image understanding"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load the SmolDocling model"""
        try:
            self.processor = AutoProcessor.from_pretrained("google/smoldocling")
            self.model = AutoModelForVision2Seq.from_pretrained(
                "google/smoldocling"
            ).to(self.device)
            logging.info("SmolDocling model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load SmolDocling model: {str(e)}")
            self.model = None
            self.processor = None

    def analyze(self, image) -> str:
        """Analyze a single image and return the extracted text"""
        if self.model is None or self.processor is None:
            logging.error("SmolDocling model not loaded")
            return ""

        try:
            # Prepare the image for the model
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Generate text from the image
            generated_ids = self.model.generate(
                **inputs, max_length=512, num_beams=4, early_stopping=True
            )

            # Decode the generated text
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return generated_text
        except Exception as e:
            logging.error(f"Error analyzing image with SmolDocling: {str(e)}")
            return ""

    def analyze_batch(self, images) -> List[str]:
        """Analyze a batch of images and return the extracted text for each"""
        results = []

        # Process images in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(images), 4)) as executor:
            results = list(executor.map(self.analyze, images))

        return results


class MistralOCRImageAnalyzer:
    """Image analyzer using MistralOCR API"""

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.api_url = "https://api.mistral.ai/v1/ocr"

        if not api_key:
            logging.warning("MistralOCR API key not provided. API calls will fail.")

    def analyze(self, image) -> str:
        """Analyze a single image using MistralOCR API"""
        if not self.api_key:
            logging.error("MistralOCR API key not provided")
            return ""

        try:
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()

            # Encode image to base64
            encoded_image = base64.b64encode(img_byte_arr).decode("utf-8")

            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {"image": encoded_image, "model": "mistral-ocr"}

            # Make the API request
            response = requests.post(self.api_url, headers=headers, json=payload)

            if response.status_code == 200:
                result = response.json()
                return result.get("text", "")
            else:
                logging.error(
                    f"MistralOCR API error: {response.status_code} - {response.text}"
                )
                return ""

        except Exception as e:
            logging.error(f"Error analyzing image with MistralOCR: {str(e)}")
            return ""

    def analyze_batch(self, images) -> List[str]:
        """Analyze a batch of images using MistralOCR API"""
        results = []

        # Process images sequentially to avoid API rate limits
        for image in images:
            results.append(self.analyze(image))

        return results
