import pypdfium2
from marker.convert import convert_single_pdf
import logging
import fitz  # PyMuPDF
import io
import tempfile
import torch
import torch.multiprocessing as mp
import logging
import os
from multiprocessing import Pool, cpu_count
from PIL import Image, UnidentifiedImageError
from typing import List, Tuple, Any, Dict
from . import md_processor
import re
from marker.models import load_all_models
import tempfile
from marker.settings import settings
from src.mmore.type import FileDescriptor, MultimodalSample, MultimodalRawInput
from .processor import Processor, ProcessorConfig
import subprocess
from src.mmore.process.utils import clean_text, clean_image

BATCH_SIZE = 1

logger = logging.getLogger(__name__)

class PDFProcessor(Processor):
    def __init__(self, config=None):
        super().__init__(config=config or ProcessorConfig())
        self.ocr_models = {device: None for device in range(torch.cuda.device_count())}

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool: 
        return file.file_extension.lower() == ".pdf"
    

    def process_batch(self, files, fast_mode, num_workers):
        if fast_mode:
            return super().process_batch(files, fast_mode, num_workers)
        else:
            # NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert ../pdf_in ../md_out
            # TODO
            env_vars = {
                'NUM_DEVICES': f"{torch.cuda.device_count()}",
                'NUM_WORKERS': f"{num_workers}",
            }
            print(self.config.custom_config.get("data_path", None))
            command = ['marker_chunk_convert', "/home/stef/Desktop/School/EPFL/MA3/semester_project/mmore/examples/sample_data/pdf", "/home/stef/Desktop/School/EPFL/MA3/semester_project/mmore/examples/outputs/"]
            subprocess.run(command, env=env_vars, capture_output=True, text=True)

    def process(self, file_path: str) -> List[MultimodalSample]:
        self.process_batch([file_path])

    def process_fast(self, file_path) -> MultimodalSample:
        pdf_doc = fitz.open(file_path)
        extracted_text = []
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
                extracted_text.append(text)

            for img_info in page.get_images(full=False):
                image = _extract_images(pdf_doc, img_info[0])
                if clean_image(image):  # clean image filters images below size 512x512 and variance below 100, these are defaults and can be changed
                    embedded_images.append(image)
                    extracted_text.append(self.config.attachment_tag)

        return self.create_sample(extracted_text, embedded_images, file_path)    
