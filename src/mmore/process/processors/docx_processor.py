import logging
import io
from docx import Document
from typing import List
from PIL import Image
from src.mmore.process.utils import clean_text, create_sample
from src.mmore.type import FileDescriptor
from .processor import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class DOCXProcessor(Processor):
    def __init__(self, files, config=None):
        super().__init__(files, config=config or ProcessorConfig())
        self.ocr_models = None

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() in [".docx"]

    def require_gpu(self) -> bool:
        return False, False

    def process_implementation(self, file_path: str) -> dict:
        # First, we define a helper functions
        def _extract_images(doc: Document) -> List[Image.Image]:
            images = []
            for rel in doc.part.rels.values():
                if "image" in rel.reltype:
                    image = Image.open(io.BytesIO(rel.target_part.blob)).convert("RGB")
                    images.append(image)
            return images

        try:
            doc = Document(file_path)
        except Exception as e:
            logger.error(f"Failed to open Word file {file_path}: {e}")
            return create_sample([], [])

        embedded_images = _extract_images(doc)
        all_text = []

        for para in doc.paragraphs:
            cleaned = clean_text(para.text)
            if cleaned.strip():
                all_text.append(cleaned)

            xml = para._p.xml
            # check if there are any images in the paragraph, replace with <attachment> placeholder
            if "w:drawing" in xml:
                all_text.append("<attachment>")

        return create_sample(all_text, embedded_images, file_path)
