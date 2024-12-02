import logging
from typing import List
import trafilatura
from urllib.parse import urljoin
from src.mmore.process.utils import clean_text, create_sample, evenly_split_across_gpus
from src.mmore.type import URLDescriptor
from .processor import Processor, ProcessorConfig
import re 
import requests
from PIL import Image
import io

logger = logging.getLogger(__name__)

class URLProcessor(Processor):
    def __init__(self, urls: List[URLDescriptor], config=None):
        """
        Initialize the URLProcessor.

        :param urls: List of URLDescriptor objects to process.
        :param config: ProcessorConfig object with configuration settings.
        """
        super().__init__(files=urls, config=config or ProcessorConfig()) 
        self.urls = urls
        self.ocr_models = None  # Models will be loaded per process
        self.driver = None  # WebDriver will be initialized per process

    @classmethod
    def accepts(cls, input_obj) -> bool:
        return isinstance(input_obj, URLDescriptor)

    def require_gpu(self) -> bool:
        return True, False

    def process_implementation(self, file_path): # TODO: OCR implementation
        return NotImplementedError("URLProcessor is not implemented")

    def process_fast_implementation(self, file_path: str) -> dict:
        try: # wrap in try because urls can be buggy
            downloaded = trafilatura.fetch_url(file_path)
            if not downloaded:
                raise ValueError(f"Failed to fetch content from URL: {file_path}")
            result = trafilatura.extract(downloaded, include_images=True)
            if not result:
                raise ValueError(f"Failed to extract content from URL: {file_path}")

            image_list = []
            # replace all ![] with <attachment>
            text = re.sub(r'!\[.*\]\(.*\)', "<attachment>", result)
            images = re.findall(r'!\[.*\]\(.*\)', result)

            for image in images:
                try:
                    image_url = re.search(r'\(.*\)', image).group(0)[1:-1]
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(image_url, headers=headers, timeout=5)
                    response.raise_for_status()
                    img = Image.open(io.BytesIO(response.content)).convert("RGB")
                    image_list.append(img)
                except Exception as e:
                    print(f"Failed to process image {image}: {e}")
            
            texts = [clean_text(text)]
            return create_sample(texts, image_list)
        except Exception as e:
            logger.error(f"Failed to process URL {file_path}: {e}")
            return create_sample("", []) 

