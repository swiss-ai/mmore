import os
import io
import logging
import pandas as pd
from typing import List
from PIL import Image as PILImage
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as OpenPyXLImage
from src.mmore.process.utils import clean_text, create_sample
from mmore.types.type import FileDescriptor
from .processor import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class SpreadsheetProcessor(Processor):
    """
    A processor for handling spreadsheet files, including Excel and CSV/TSV files.
    Extracts text and images (if applicable) from supported spreadsheet files.

    Attributes:
        files (List[FileDescriptor]): List of files to be processed.
        config (ProcessorConfig): Configuration for the processor.
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
            bool: True if the file is a supported spreadsheet format, False otherwise.
        """
        return file.file_extension.lower() in [".xlsx", ".xls", ".csv", ".tsv"]

    def require_gpu(self) -> bool:
        """
        Returns:
            tuple: A tuple (False, False) indicating no GPU requirement for both standard and fast modes.
        """        
        return False, False

    def process_implementation(self, file_path):
        """
        Process a spreadsheet file to extract text and images (if applicable).

        Args:
            file_path (str): Path to the spreadsheet file.

        Returns:
            dict: A dictionary containing extracted text and images.

        The method extracts text from supported spreadsheet formats and images only from `.xlsx` files.
        """
        # First, we define helper functions
        def _extract_text(file_path: str) -> str:
            """
            Extract text content from an Excel or CSV/TSV file.

            Args:
                file_path (str): Path to the spreadsheet file.

            Returns:
                str: Extracted text content.
            """
            # Helper function to extract text from an Excel or CSV file
            def _extract_text_excel(file_path: str, ext: str) -> str:
                """
                Textception 

                Args:
                    file_path (str): Path to the Excel file.
                    ext (str): File extension (.xlsx or .xls).

                Returns:
                    str: Extracted text content from all sheets.
                """
                engine = "openpyxl" if ext == ".xlsx" else "xlrd"
                try:
                    df_dict = pd.read_excel(file_path, sheet_name=None, engine=engine)
                except Exception as e:
                    logger.error(f"Error reading Excel file {file_path}: {e}")
                    return ""
                text = ""
                for sheet_name, df in df_dict.items():
                    text += f"Sheet: {sheet_name}\n"
                    text += df.to_string(index=False) + "\n\n"
                return text.strip()

            def _extract_text_csv(file_path: str, ext: str) -> str:
                """
                Textception part 2

                Args:
                    file_path (str): Path to the CSV/TSV file.
                    ext (str): File extension (.csv or .tsv).

                Returns:
                    str: Extracted text content.
                """
                sep = "\t" if ext == ".tsv" else ","
                try:
                    df = pd.read_csv(file_path, sep=sep)
                except Exception as e:
                    logger.error(f"Error reading CSV file {file_path}: {e}")
                    return ""
                return df.to_string(index=False)

            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".xlsx", ".xls"]:
                return _extract_text_excel(file_path, ext)
            elif ext in [".csv", ".tsv"]:
                return _extract_text_csv(file_path, ext)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

        def _extract_images(file_path: str) -> List[PILImage.Image]:
            """
            Extract images from an Excel file.

            Args:
                file_path (str): Path to the Excel file.

            Returns:
                List[PILImage.Image]: List of extracted images.
            """
            ext = os.path.splitext(file_path)[1].lower()

            if ext != ".xlsx":
                logger.warning(
                    f"Image extraction is only supported for .xlsx files. Skipping {file_path}."
                )
                return []

            try:
                images = []
                wb = load_workbook(filename=file_path, data_only=True)
                for sheet in wb.worksheets:
                    if hasattr(sheet, "_images"):
                        for image in getattr(sheet, "_images", []):
                            if isinstance(image, OpenPyXLImage):
                                img_bytes = image._data()
                                img = PILImage.open(io.BytesIO(img_bytes)).convert(
                                    "RGB"
                                )
                                images.append(img)
                logger.info(f"Extracted {len(images)} images from {file_path}.")
                return images
            except Exception as e:
                logger.error(f"Failed to extract images from {file_path}: {e}")
                return []

        text = _extract_text(file_path)
        cleaned_text = clean_text(text)
        images = _extract_images(file_path)

        return create_sample([cleaned_text], images, file_path)
