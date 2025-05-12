import io
import logging
from typing import List

from docx import Document
from docx.document import Document as DocumentType
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from PIL import Image

from ...type import FileDescriptor, MultimodalSample
from ..utils import clean_text
from .base import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class DOCXProcessor(Processor):
    def __init__(self, config=None):
        """
        A processor for handling Microsoft Word documents (.docx). Extracts text content and embedded images.

        Attributes:
            files (List[FileDescriptor]): List of DOCX files to be processed.
            config (ProcessorConfig): Config for the processor, which includes options such as
                                    the placeholder tag for embedded images (e.g., "<attachment>").
        """
        super().__init__(config=config or ProcessorConfig())

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        """
        Args:
            file (FileDescriptor): The file descriptor to check.

        Returns:
            bool: True if the file is a DOCX file, False otherwise.
        """
        return file.file_extension.lower() in [".docx"]

    def process(self, file_path: str) -> MultimodalSample:
        """
        Process a single DOCX file. Extracts text content and embedded images.

        Args:
            file_path (str): Path to the DOCX file.

        Returns:
            dict: A dictionary containing processed text and embedded images.

        The method parses the DOCX file, cleans and extracts textual content from paragraphs,
        and extracts embedded images. Images in the paragraphs are replaced with a placeholder tag
        defined in the processor configuration (e.g., "<attachment>").
        """

        # First, we define a helper functions
        def _extract_images(doc: DocumentType) -> List[Image.Image]:
            """
            Extract embedded images from the DOCX document.

            Args:
                doc (Document): The DOCX document object.

            Returns:
                List[Image.Image]: A list of extracted PIL images.
            """
            images = []
            for rel in doc.part.rels.values():
                if rel.reltype == RT.IMAGE and not rel.is_external:
                    try:
                        blob = rel.target_part.blob
                        image = Image.open(io.BytesIO(blob)).convert("RGB")
                        images.append(image)
                    except Exception as e:
                        logger.error(f"Failed to extract image: {e}")
            return images

        try:
            doc = Document(file_path)
        except Exception as e:
            logger.error(f"Failed to open Word file {file_path}: {e}")
            return self.create_sample([], [], file_path)

        if self.config.custom_config.get("extract_images", True):
            embedded_images = _extract_images(doc)
        else:
            embedded_images = []

        all_text = []
        for para in doc.paragraphs:
            cleaned = clean_text(para.text)

            if cleaned.strip():
                all_text.append(cleaned)

            if self.config.custom_config.get("extract_images", True):
                xml = para._p.xml
                # check if there are any images in the paragraph, replace with <attachment> token
                if "w:drawing" in xml:
                    all_text.append(self.config.attachment_tag)

        return self.create_sample(all_text, embedded_images, file_path)
