import logging
import io
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from PIL import Image
from ...process.utils import clean_text, create_sample
from ...type import FileDescriptor
from .processor import Processor, ProcessorConfig
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class PPTXProcessor(Processor):
    def __init__(self, files, config=None):
        super().__init__(files, config=config or ProcessorConfig())
        self.ocr_models = None

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() in [".pptx"]

    def require_gpu(self) -> bool:
        return False, False

    def _extract_slide_content(
            self, slide, all_text: List[str], embedded_images: List[Image.Image]
    ):
        """
        Extract text and images from a slide, and append to the provided lists.
        """
        # Sort shapes by their vertical position
        shape_list = sorted(
            (shape for shape in slide.shapes if hasattr(shape, "top")),
            key=lambda s: s.top,
        )

        for shape in shape_list:
            # Extract text from shape
            if shape.has_text_frame:
                cleaned_text = clean_text(shape.text)
                if cleaned_text.strip():
                    all_text.append(cleaned_text)

            # Extract images from shape
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                try:
                    pil_image = Image.open(io.BytesIO(shape.image.blob)).convert("RGB")
                    embedded_images.append(pil_image)
                    all_text.append("<attachment>")
                except Exception as e:
                    logger.error(f"Failed to extract image from slide: {e}")

    def process_implementation(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single PPTX file. Extracts text, images, and notes from each slide.
        The elements are sorted by their vertical position.
        """
        logger.info(f"Processing PowerPoint file: {file_path}")
        try:
            prs = Presentation(file_path)
        except Exception as e:
            logger.error(f"Failed to open PowerPoint file {file_path}: {e}")
            return {"text": "", "modalities": []}

        all_text = []
        embedded_images = []

        try:
            for slide in prs.slides:
                # Extract text and images from each shape on the slide
                self._extract_slide_content(slide, all_text, embedded_images)

                # Extract text from slide notes if present
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame
                    for paragraph in notes.paragraphs:
                        if paragraph.text:
                            cleaned = clean_text(paragraph.text)
                            if cleaned.strip():
                                all_text.append(cleaned)

        except Exception as e:
            logger.error(f"[PPTX] Error processing slides in {file_path}: {e}")

        return create_sample(all_text, embedded_images, file_path)
