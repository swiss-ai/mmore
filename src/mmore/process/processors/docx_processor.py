import io
import uuid
import os
from pathlib import Path

import mammoth
import tempfile
import logging
from markdownify import markdownify
from typing import Dict, Any
from ...type import FileDescriptor, MultimodalRawInput, MultimodalSample

from .base import Processor, ProcessorConfig
from PIL import Image
import mimetypes

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

        all_images = []

        def _convert_image(image: mammoth.documents.Image) -> Dict[str, Any]:
            if not self.config.custom_config.get("extract_images", False):
                return {"src" : ""}

            with image.open() as image_bytes:
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes.read()))

                    # Generate unique image path and save the image there
                    image_path = Path(os.path.join(image_output_dir, str(uuid.uuid4())))
                    image_path = image_path.with_suffix(
                        mimetypes.guess_extension(image.content_type)
                    )

                    pil_image.save(image_path)
                    logger.info(f"Saving image {image_path}")
                    all_images.append(MultimodalRawInput(type="image", value=str(image_path)))

                    print("Saving image to {image_path}")

                    return {"src" : "", "alt": self.config.attachment_tag}

                except Exception as e:
                    logger.warning(
                        f"Failed to load image with MIME type {image.content_type}: {e}"
                    )
                    return {"src": "", "alt": ""}

        image_output_dir = self.config.custom_config.get("image_output_dir", None)

        # If no image_output_dir is specified then create a temporary output dir
        if image_output_dir is None:
            image_output_dir = tempfile.mkdtemp(prefix="mmore_docx_")
            logger.info(f"Saving files in {image_output_dir}")

        try:
            with open(file_path, "rb") as docx_fileobj:
                result = mammoth.convert_to_html(
                    docx_fileobj,
                    convert_image=mammoth.images.img_element(_convert_image),
                )

        except Exception as e:
            logger.warning(f"Failed to convert {file_path}: {e}")
            return self.create_sample([], [], file_path)

        markdown = markdownify(result.value)

        sample = MultimodalSample(
            text=markdown,
            modalities=all_images,
            metadata={"file_path" : file_path}
        )

        return sample
