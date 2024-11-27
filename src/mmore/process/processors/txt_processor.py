import logging
from typing import List
from src.mmore.process.utils import clean_text
from src.mmore.type import FileDescriptor
from .processor import Processor

logger = logging.getLogger(__name__)


class TextProcessor(Processor):
    def __init__(self, files: List[FileDescriptor], config=None):
        super().__init__(files, config=config)

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() in [".txt"]

    def require_gpu(self) -> bool:
        return False, False

    def process_implementation(self, file_path: str) -> dict:
        """
        Process a text file, clean its content, and return a dictionary with the cleaned text.

        :param file_path: Path to the text file.
        :return: A dictionary containing cleaned text and an empty list of modalities.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return {"text": "", "modalities": []}
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in file {file_path}: {e}")
            return {"text": "", "modalities": []}

        cleaned_text = clean_text(text)
        return {"text": cleaned_text, "modalities": [], "metadata": {"file_path": file_path}}
