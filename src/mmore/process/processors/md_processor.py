import logging
import markdown
import markdownify
from ...type import FileDescriptor
from .processor import Processor, ProcessorConfig
import tempfile
from PIL import Image
import os
import io

logger = logging.getLogger(__name__)


class MarkdownProcessor(Processor):
    def __init__(self, files, config=None):
        super().__init__(files, config=config or ProcessorConfig())
        self.md = markdown.Markdown()

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() in [".md"]

    def require_gpu(self) -> bool:
        return False, False

    def process_implementation(self, file_path: str) -> dict:
        """
        Process a Markdown file to extract text and embedded images.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            dict: Contains processed text and embedded images
        """
        logger.info(f"Processing Markdown file: {file_path}")

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
            content, embedded_images = self.process_md(content, file_path)
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
        os.makedirs(base_path, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=base_path, suffix='.png') as tmp:
            image.save(tmp.name)

        return tmp.name

    @staticmethod
    def process_md(
            content: str, file_path: str, attachment_tag: str = None
    ) -> str:
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
