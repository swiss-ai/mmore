import pypdfium2
import logging
import fitz  # PyMuPDF
import io
import logging
from PIL import Image, UnidentifiedImageError
from typing import List
from src.mmore.type import FileDescriptor, MultimodalSample, MultimodalRawInput
from .base import Processor, ProcessorConfig
from src.mmore.process.utils import clean_text, clean_image

from tqdm import tqdm

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

class PDFProcessor(Processor):
    def __init__(self, config=None):
        super().__init__(config=config or ProcessorConfig())    

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool: 
        return file.file_extension.lower() == ".pdf"

    # overwriting the process_batch 
    def process_batch(self, files, fast_mode, num_workers):
        if fast_mode:
            return super().process_batch(files, fast_mode, num_workers)
        else:
            # marker config . pass in through config   
            marker_config = {
                "disable_image_extraction": not self.config.custom_config.get("extract_images", True),
                "languages": None, 
                "use_llm": False, # if you want to use an llm to clean the output
                "disable_multiprocessing": False,
            }  
            config_parser = ConfigParser(marker_config)
            self.converter = PdfConverter(artifact_dict=create_model_dict(), config=config_parser.generate_config_dict())

            results = []
            for file_path in tqdm(files, desc="Processing PDFs...", total=len(files)):
                results.append(self.process(file_path))

            return results

    def process(self, file_path: str) -> List[MultimodalSample]:
        rendered = self.converter(file_path)
        text, _, images = text_from_rendered(rendered)
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
