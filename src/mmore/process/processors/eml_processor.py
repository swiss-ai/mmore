import logging
import io
import email
from email import policy
from PIL import Image
from typing import Dict, Any
from src.mmore.process.utils import clean_text, create_sample
from src.mmore.type import FileDescriptor
from .processor import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class EMLProcessor(Processor):
    """
    A processor for handling email files (.eml). Extracts email headers, text content, and embedded images.

    Attributes:
        files (List[FileDescriptor]): List of EML files to be processed.
        config (ProcessorConfig): Configuration for the processor, including options such as the 
                                   placeholder tag for embedded images (e.g., "<attachment>").
    """
    def __init__(self, files, config=None):
        """
        Args:
            files (List[FileDescriptor]): List of files to process.
            config (ProcessorConfig, optional): Configuration for the processor. Defaults to None.
        """
        super().__init__(files, config=config or ProcessorConfig())
    
    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        """
        Args:
            file (FileDescriptor): The file descriptor to check.

        Returns:
            bool: True if the file is an EML file, False otherwise.
        """
        return file.file_extension.lower() in [".eml"]

    def require_gpu(self) -> bool:
        """
        Returns:
            tuple: A tuple (False, False) indicating no GPU requirement for both standard and fast modes.
        """
        return False, False

    def process_implementation(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single EML file. Extracts text content, email headers, and embedded images. 

        Args:
            file_path (str): Path to the EML file.

        Returns:
            dict: A dictionary containing processed text, embedded images, and metadata.

        The method parses the EML file, extracts email headers, text content, and embedded images.
        Embedded images are replaced with a placeholder tag from the processor configuration.
        """
        try:
            with open(file_path, 'rb') as f:
                msg = email.message_from_bytes(f.read(), policy=policy.default)
        except Exception as e:
            logger.error(f"Failed to open EML file {file_path}: {e}")
            return create_sample([], [], file_path)

        all_text = []
        embedded_images = []

        # extract email headers
        headers = [
            f"From: {msg.get('From', '')}",
            f"To: {msg.get('To', '')}",
            f"Subject: {msg.get('Subject', '')}",
            f"Date: {msg.get('Date', '')}"
        ]
        all_text.extend([clean_text(header) for header in headers if header])
        
        for part in msg.walk():
            # extract text
            if part.get_content_type() == 'text/plain':
                try:
                    text = part.get_content()
                    cleaned = clean_text(text)
                    if cleaned.strip():
                        all_text.append(cleaned)
                except Exception as e:
                    logger.error(f"Error extracting text from EML: {e}")

            # extract images
            elif part.get_content_type().startswith('image/'):
                try:
                    image_data = part.get_payload(decode=True)
                    image = Image.open(io.BytesIO(image_data)).convert("RGB")
                    embedded_images.append(image)
                    all_text.append(self.config.attachment_tag) # default token is "<attachment>"
                except Exception as e:
                    logger.error(f"Error extracting image from EML: {e}")

        return create_sample(all_text, embedded_images, file_path)