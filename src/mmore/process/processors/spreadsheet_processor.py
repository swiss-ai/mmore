import os
import io
import logging
import pandas as pd
from typing import List
from PIL import Image as PILImage
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as OpenPyXLImage
from ...process.utils import clean_text, clean_image, create_sample
from ...type import FileDescriptor
from .processor import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class SpreadsheetProcessor(Processor):
    def __init__(self, files, config=None):
        super().__init__(files, config=config or ProcessorConfig())

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() in [".xlsx", ".xls", ".csv", ".tsv"]

    def require_gpu(self) -> bool:
        return False, False

    def process_implementation(self, file_path):
        # First, we define helper functions
        def _extract_text(file_path: str) -> str:
            # Helper function to extract text from an Excel or CSV file
            def _extract_text_excel(file_path: str, ext: str) -> str:
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
        cleaned_images = [img for img in images if clean_image(img)]

        return create_sample([cleaned_text], cleaned_images, file_path)
