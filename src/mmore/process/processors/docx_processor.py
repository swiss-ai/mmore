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
        """
        A processor for handling Microsoft Word documents (.docx). Extracts text content and embedded images.

        Attributes:
            files (List[FileDescriptor]): List of DOCX files to be processed.
            config (ProcessorConfig): Config for the processor, which includes options such as
                                    the placeholder tag for embedded images (e.g., "<attachment>").
        """
        super().__init__(files, config=config or ProcessorConfig())

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        """
        Args:
            file (FileDescriptor): The file descriptor to check.

        Returns:
            bool: True if the file is a DOCX file, False otherwise.
        """
        return file.file_extension.lower() in [".docx"]

    def require_gpu(self) -> bool:
        """
        Returns:
            tuple: A tuple (False, False) indicating no GPU requirement for both standard and fast modes.
        """        
        return False, False

    def process_implementation(self, file_path: str) -> dict:
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
        def _extract_images(doc: Document) -> List[Image.Image]:
            """
            Extract embedded images from the DOCX document.

            Args:
                doc (Document): The DOCX document object.

            Returns:
                List[Image.Image]: A list of extracted PIL images.
            """
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
            return create_sample([], [], None)

        embedded_images = _extract_images(doc)
        all_text = []
        for para in doc.paragraphs:
            cleaned = clean_text(para.text)
            
            if cleaned.strip():
                all_text.append(cleaned)

            xml = para._p.xml
            # check if there are any images in the paragraph, replace with <attachment> token
            if "w:drawing" in xml: 
                all_text.append(self.config.attachment_tag)

        return create_sample(all_text, embedded_images, file_path)
