import io
import logging
from typing import List, Optional, Tuple

import pymupdf
from PIL import Image, UnidentifiedImageError

from ...type import FileDescriptor, MultimodalSample
from ..utils import clean_image
from .base import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class PyMuPDF4LLMProcessor(Processor):
    """Alternative PDF processor using ``pymupdf4llm`` for LLM-friendly markdown.

    This processor relies on `pymupdf4llm <https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/>`_
    to convert PDFs directly to markdown without GPU-accelerated OCR. It is
    much faster than the default :class:`PDFProcessor` (marker-based) but does
    not perform OCR on scanned documents.

    To opt in, set ``file_type_processors`` in your config::

        file_type_processors:
          .pdf: PyMuPDF4LLMProcessor
    """

    def __init__(self, config=None):
        super().__init__(config=config or ProcessorConfig())

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() == ".pdf"

    def process(self, file_path: str) -> MultimodalSample:
        try:
            import pymupdf4llm
        except ImportError as e:
            raise ImportError(
                "pymupdf4llm is required for PyMuPDF4LLMProcessor. "
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
                metadata = chunk.get("metadata", {}) or {}
                page_num = int(metadata.get("page", 0))
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
                            image = self._extract_image(pdf_doc, img_info[0])
                            if image and clean_image(image):
                                embedded_images.append(image)
                                attachment_text = self.config.attachment_tag
                                all_text_parts.append(attachment_text)
                                current_position += len(attachment_text)
        finally:
            if pdf_doc is not None:
                pdf_doc.close()

        paragraph_starts.append((current_position, -1, -1))

        sample_metadata = {
            "file_path": file_path,
            "paragraph_starts": paragraph_starts,
            "document_type": "pdf",
        }

        full_text = "".join(all_text_parts)
        return self.create_sample([full_text], embedded_images, sample_metadata)

    @staticmethod
    def _extract_image(pdf_doc, xref) -> Optional[Image.Image]:
        try:
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image.get("image")
            if image_bytes is None:
                logger.error(f"No image data found for xref {xref}")
                return None
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except KeyError as e:
            logger.error(f"KeyError while extracting image for xref {xref}: {e}")
            return None
        except UnidentifiedImageError as e:
            logger.error(
                f"UnidentifiedImageError: could not identify image for xref {xref}: {e}"
            )
            return None
        except Exception as e:
            logger.error(f"Unexpected error while extracting image for xref {xref}: {e}")
            return None
