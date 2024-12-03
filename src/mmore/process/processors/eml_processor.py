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
    def __init__(self, files, config=None):
        super().__init__(files, config=config or ProcessorConfig())
    
    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() in [".eml"]

    def require_gpu(self) -> bool:
        return False, False

    def process_implementation(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single EML file. Extracts text content and embedded images.
        """
        logger.info(f"Processing EML file: {file_path}")
        print(f"Processing EML file: {file_path}")
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
                    all_text.append("<attachment>")
                except Exception as e:
                    logger.error(f"Error extracting image from EML: {e}")

        return create_sample(all_text, embedded_images, file_path)