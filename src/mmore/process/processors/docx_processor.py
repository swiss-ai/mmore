import io
import logging
import mimetypes
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Optional, cast

import mammoth
from mammoth.documents import Image as m_Image
from markdownify import markdownify
from PIL import Image

from ...type import FileDescriptor, MultimodalRawInput, MultimodalSample
from .base import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class DOCXProcessor(Processor):
    """Processor for Microsoft Word documents (``.docx``).

    Extracts text content and embedded images using
    `mammoth <https://github.com/mwilliamson/python-mammoth>`_.
    """

    def __init__(self, config=None):
        super().__init__(config=config or ProcessorConfig())

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() in [".docx"]

    def process(self, file_path: str) -> MultimodalSample:
        """Extract text and images from a ``.docx`` file.

        Args:
            file_path: Path to the ``.docx`` file.

        Returns:
            A :class:`~mmore.type.MultimodalSample` containing the extracted
            text (as Markdown) and image paths.
        """

        all_images = []

        # Images are saved to output_path/images/ when available,
        # falling back to a temporary directory.
        if self.config.output_path:
            image_output_dir = os.path.join(self.config.output_path, self.IMAGES_DIR)
            os.makedirs(image_output_dir, exist_ok=True)
        else:
            image_output_dir = tempfile.mkdtemp(prefix="mmore_docx_")
            logger.info(f"Saving DOCX images in temp dir {image_output_dir}")

        def _convert_image(image: m_Image) -> Dict[str, Any]:
            if not self.config.extract_images:
                return {"src": ""}

            content_type = cast(Optional[str], image.content_type)

            with cast(Callable[[], BinaryIO], image.open)() as image_bytes:
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes.read()))

                    if content_type is None:
                        raise ValueError("Invalid content type")

                    image_path = Path(os.path.join(image_output_dir, str(uuid.uuid4())))
                    extension = mimetypes.guess_extension(content_type)
                    if extension is None:
                        raise ValueError(
                            "Unable to determine the extension of the image"
                        )

                    image_path = image_path.with_suffix(extension)
                    pil_image.save(image_path)
                    logger.info(f"Saving image to {image_path}")
                    all_images.append(
                        MultimodalRawInput(type="image", value=str(image_path))
                    )

                    return {"src": "", "alt": self.config.attachment_tag}

                except Exception as e:
                    logger.warning(
                        f"Failed to load image with MIME type {content_type}: {e}"
                    )
                    return {"src": "", "alt": ""}

        try:
            with open(file_path, "rb") as docx_fileobj:
                result = mammoth.convert_to_html(
                    docx_fileobj,
                    convert_image=mammoth.images.img_element(_convert_image),
                )

        except Exception as e:
            logger.warning(f"Failed to convert {file_path}: {e}")
            return self.create_sample([], [], {"file_path": file_path})

        markdown = markdownify(result.value)

        sample = MultimodalSample(
            text=re.sub(r"!\[<([^>]+)>\]\(\)", r"<\1>", markdown),
            modalities=all_images,
            metadata={"file_path": file_path},
        )

        return sample
