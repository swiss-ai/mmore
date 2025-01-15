import logging
import markdown
import markdownify
from mmore.types.type import FileDescriptor
from .processor import Processor, ProcessorConfig
import tempfile
from PIL import Image
import os
import io

logger = logging.getLogger(__name__)


class MarkdownProcessor(Processor):
    """
    A processor for handling Markdown files (.md). Extracts text content and embedded images.

    Attributes:
        files (List[FileDescriptor]): List of Markdown files to be processed.
        config (ProcessorConfig): Configuration for the processor, including options such as the 
                                   placeholder tag for embedded images (e.g., "<attachment>").
        md (markdown.Markdown): Instance of the Markdown parser used to convert content to HTML.
    """
    def __init__(self, files, config=None):
        """
        Args:
            files (List[FileDescriptor]): List of files to process.
            config (ProcessorConfig, optional): Configuration for the processor. Defaults to None.
        """
        super().__init__(files, config=config or ProcessorConfig())
        self.md = markdown.Markdown()

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        """
        Args:
            file (FileDescriptor): The file descriptor to check.
        Returns:
            bool: True if the file is a Markdown (.md) file, False otherwise.
        """
        return file.file_extension.lower() in [".md"]

    def require_gpu(self) -> bool:
        """
        Returns:
            tuple: A tuple (False, False) indicating no GPU requirement for both standard and fast modes.
        """
        return False, False

    def process_implementation(self, file_path: str) -> dict:
        """
        Process a Markdown file to extract text and embedded images.

        Args:
            file_path (str): Path to the Markdown file.

        Returns:
            dict: A dictionary containing processed text, embedded images, and metadata.

        The method parses the Markdown file content, converts it to HTML, extracts image links,
        downloads or loads local images, and replaces image tags with a placeholder defined
        in the processor configuration.
        """

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"text": [], "modalities": [], "error": "File not found"}  # TODO: ???

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading file {file_path}: {e}")
            return {"text": [], "modalities": [], "error": "Encoding error"}
        except IOError as e:
            logger.error(f"IO error reading file {file_path}: {e}")
            return {"text": [], "modalities": [], "error": "IO error"}

        try:
            content, embedded_images = self.process_md(content, file_path, self.config.attachment_tag)
            return {
                "text": content,
                "modalities": [{"type": "image", "value": img} for img in embedded_images],
                "metadata": {"file_path": file_path}
            }

        except Exception as e:
            logger.error(f"[MD Processor] Error processing markdown content: {e}")
            return {"text": [], "modalities": [], "error": "Processing error"}  # TODO: ???

    @staticmethod
    def save_temp_image(image: Image.Image, base_path: str) -> str:
        """
        Save an image to a temporary file.

        Args:
            image (Image.Image): The PIL Image to save.
            base_path (str): Directory path where the temporary file should be saved.

        Returns:
            str: Path to the saved temporary file.
        """
        os.makedirs(base_path, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=base_path, suffix='.png') as tmp:
            image.save(tmp.name)

        return tmp.name

    @staticmethod
    def process_md(
            content: str, file_path: str, attachment_tag: str = None
    ) -> str:
        """
        The actual proccessing logic for Markdown files. 

        Args:
            content (str): The content of the Markdown file.
            file_path (str): Path to the Markdown file.
            attachment_tag (str, optional): Tag to replace image placeholders with. Defaults to <attachment>.

        Returns:
            tuple: Processed text and a list of paths to extracted images.
        """

        md = markdown.Markdown()
        html = md.convert(content)

        # Extract image links
        image_tags = [line for line in html.split("\n") if "<img" in line]
        embedded_images = []
        for tag in image_tags:
            src = tag.split('src="')[1].split('"')[0]
            image = None
            try:
                # Check if the URL is valid
                src_path = os.path.join(os.path.dirname(file_path), src)
                if src.startswith(('http://', 'https://')):
                    # Download remote image
                    import requests
                    response = requests.get(src, timeout=10)
                    if response.status_code == 200:
                        # Save to temp directory or process as needed
                        image_data = response.content
                        image = Image.open(io.BytesIO(image_data))

                        path = MarkdownProcessor.save_temp_image(image, base_path=os.path.join(os.getcwd(), 'tmp'))
                        embedded_images.append(path)
                    else:
                        logger.error(f"Failed to download image from {src}. Status code: {response.status_code}")
                        html = html.replace(tag, "")
                        continue

                elif os.path.exists(src_path):
                    image = Image.open(src_path)
                    embedded_images.append(src_path)
                else:
                    html = html.replace(tag, "")
                    logger.error(f"Image {src} not found in {file_path}")
                    continue
            except Exception as e:
                html = html.replace(tag, "")
                logger.error(f"Error processing image {src}: {str(e)}")
                continue

            html = html.replace(tag, "#attachment")

        content = markdownify.markdownify(html)
        content = content.replace(
            "#attachment", attachment_tag if attachment_tag else "<attachment>"
        )
        return content, embedded_images
