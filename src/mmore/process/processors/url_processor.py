import logging
import os
import io
import requests
import torch
from urllib.parse import urljoin
from typing import List
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import trafilatura
from multiprocessing import Pool, cpu_count
import re
from urllib.parse import urljoin
from src.mmore.process.utils import clean_text, create_sample, evenly_split_across_gpus
from src.mmore.type import URLDescriptor
from .processor import Processor, ProcessorResult, ProcessorConfig
from surya.ocr import run_ocr
from surya.model.detection.model import (
    load_model as load_det_model,
    load_processor as load_det_processor,
)
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

logger = logging.getLogger(__name__)


class URLProcessor(Processor):
    def __init__(self, urls: List[URLDescriptor], config=None):
        super().__init__(files=[], config=config or ProcessorConfig())
        self.urls = urls
        self.ocr_models = None  # Models will be loaded per process
        self.driver = None  # WebDriver will be initialized per process

    @classmethod
    def accepts(cls, input_obj) -> bool:
        return isinstance(input_obj, URLDescriptor)

    def require_gpu(self) -> bool:
        return True, False

    def load_models(self, device=None):
        """Load the OCR models, optionally on a specific device."""
        if self.ocr_models is None:
            device = torch.device(device) if device is not None else torch.device("cpu")
            try:
                det_model = load_det_model()
                det_model.to(device)
                det_processor = load_det_processor()
                rec_model = load_rec_model()
                rec_model.to(device)
                rec_processor = load_rec_processor()
                self.ocr_models = (det_model, det_processor, rec_model, rec_processor)
                logger.info(f"OCR models loaded successfully on device {device}.")
            except Exception as e:
                logger.error(f"Error loading OCR models on device {device}: {e}")
        self._init_selenium_driver()

    def process_with_cpu(self, process_method) -> ProcessorResult:
        return NotImplementedError("URLProcessor is not implemented")

    def process_implementation(self, file_path):
        return NotImplementedError("URLProcessor is not implemented")

    def process_fast_implementation(self, file_path):
        return NotImplementedError("URLProcessor is not implemented")

    def _render(self, url: str) -> List[Image.Image]:
        return NotImplementedError("URLProcessor is not implemented")

    def _init_selenium_driver(self) -> None:
        """Initialize the Selenium WebDriver."""
        if self.driver is None:
            try:
                options = Options()
                options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                self.driver = webdriver.Chrome(options=options)
                logger.info("Selenium webdriver initialized")
            except Exception as e:
                logger.error(f"Error initializing Selenium WebDriver: {e}")

    def _extract_images(self, url: str) -> List[Image.Image]:
        def _resolve_image_url(base_url: str, src: str) -> str:
            if src.startswith("//"):
                return "http:" + src
            elif src.startswith("/") or not src.startswith("http"):
                return urljoin(base_url, src)
            return src

        self.driver.get(url)
        self.driver.implicitly_wait(5)

        image_elements = self.driver.find_elements("tag name", "img")
        images = []
        for img_elem in image_elements:
            src = img_elem.get_attribute("src")
            if src:
                try:
                    src = self._resolve_image_url(url, src)
                    response = requests.get(src, timeout=5)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    images.append(image)
                except Exception as img_e:
                    logger.error(f"Failed to download or process image {src}: {img_e}")
        return images

    def _create_sample(self, texts: List[str], images: List[Image.Image]) -> dict:
        return {
            "text": "\n".join(texts),
            "modalities": [{"type": "image", "value": self._save_temp_image(img)} for img in images]
        }

    def _cleanup(self):
        """Clean up resources, the Selenium WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                logger.error("Failed to quit the Selenium WebDriver.")

    def __del__(self):
        self.cleanup()
