import logging
import os
import io
from bs4 import BeautifulSoup
from PIL import Image
import requests
from typing import List
from src.mmore.process.utils import clean_text
from src.mmore.type import FileDescriptor, MultimodalSample
from .base import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class HTMLProcessor(Processor):
    def __init__(self, config=None):
        """
        A processor for HTML files. Extracts text content and optionally embedded images.

        Attributes:
            files (List[FileDescriptor]): List of HTML files to be processed.
            config (ProcessorConfig): Config for the processor, which includes options such as
                                    the placeholder tag for embedded images (e.g., "<attachment>").
        """
        super().__init__(config=config or ProcessorConfig())

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() in [".html", ".htm"]

    def process(self, file_path: str) -> MultimodalSample:
        """
        Process a single HTML file. Extracts text content and embedded images.

        Args:
            file_path (str): Path to the HTML file.

        Returns:
            MultimodalSample: A dictionary containing processed text and embedded images.
        """

        def _extract_images(soup: BeautifulSoup) -> List[Image.Image]:
            """
            Extract images embedded in HTML (by URL or local).

            Args:
                soup (BeautifulSoup): Parsed HTML soup.

            Returns:
                List[Image.Image]: A list of PIL images.
            """
            images = []
            for img_tag in soup.find_all("img"):
                src = img_tag.get("src")
                if not src:
                    continue
                try:
                    if src.startswith("http"):
                        response = requests.get(src)
                        image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    else:
                        local_path = os.path.join(os.path.dirname(file_path), src)
                        image = Image.open(local_path).convert("RGB")
                    images.append(image)
                except Exception as e:
                    logger.error(f"Failed to load image {src}: {e}")
            return images

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
        except Exception as e:
            logger.error(f"Failed to open HTML file {file_path}: {e}")
            return self.create_sample([], [], file_path)

        soup = BeautifulSoup(html, "html.parser")

        if self.config.custom_config.get("extract_images", True):
            embedded_images = _extract_images(soup)
        else:
            embedded_images = []

        all_text = []
        body = soup.body if soup.body else soup  # fallback if no <body> tag

        for tag in body.find_all(string=True):  # better: use string=True instead of text=True
            if tag.parent.name not in ["script", "style"]:
                cleaned = clean_text(tag)
                if cleaned.strip():
                    all_text.append(cleaned)
        if self.config.custom_config.get("extract_images", True):
            for img_tag in soup.find_all("img"):
                all_text.append(self.config.attachment_tag)

        return self.create_sample(all_text, embedded_images, file_path)
