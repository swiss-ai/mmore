import logging
from typing import List
from src.mmore.process.utils import clean_text
from src.mmore.type import FileDescriptor, MultimodalSample
from .processor import Processor

logger = logging.getLogger(__name__)


class TextProcessor(Processor):
    """
    A processor for handling plain text files (.txt). Reads and cleans the text content.

    Attributes:
        files (List[FileDescriptor]): List of files to be processed.
        config (ProcessorConfig): Configuration for the processor.
    """
    def __init__(self, files: List[FileDescriptor], config=None):
        """
        Args:
            files (List[FileDescriptor]): List of files to process.
            config (ProcessorConfig, optional): Configuration for the processor. Defaults to None.
        """
        super().__init__(files, config=config)

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        """
        Args:
            file (FileDescriptor): The file descriptor to check.

        Returns:
            bool: True if the file is a plain text file (.txt), False otherwise.
        """
        return file.file_extension.lower() in [".txt"]

    def require_gpu(self) -> bool:
        """
        Returns:
            tuple: A tuple (False, False) indicating no GPU requirement for both standard and fast modes.
        """
        return False

    def process(self, file_path: str) -> MultimodalSample:
        """
        Process a text file, clean its content, and return a dictionary with the cleaned text.

        Args:
            file_path (str): Path to the text file.

        Returns:
            dict: A dictionary containing cleaned text, an empty list of modalities, and metadata.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                all_text = f.read()
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return self.create_sample([], [], file_path)
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in file {file_path}: {e}")
            return self.create_sample([], [], file_path)

        all_text = clean_text(all_text)
        return self.create_sample([all_text], [], file_path)
