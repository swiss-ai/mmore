import asyncio
import base64
import io
import logging
from typing import List

from mistralai import Mistral
from PIL import Image

from ...type import FileDescriptor, MultimodalSample
from .base import Processor, ProcessorConfig


class CloudPDFProcessor(Processor):

    def __init__(self, config=None):
        super().__init__(config=config or ProcessorConfig())
        api_key = config.custom_config.get("MISTRAL_API_KEY", "test_key")
        self.mistral_client = Mistral(api_key)


    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() == ".pdf"

    # Using Mistral's documentation base64 encoder
    @staticmethod
    def _encode_pdf(pdf_path):
        """Encode the pdf to base64."""
        try:
            with open(pdf_path, "rb") as pdf_file:
                return base64.b64encode(pdf_file.read()).decode('utf-8')
        except FileNotFoundError:
            logging.error(f"File not found: {pdf_path}")
            return None
        except Exception as e:
            logging.error(f"Error encoding PDF {pdf_path}: {e}")
            return None

    @staticmethod
    def _decode_image(image_base64):
        try:
            # Handle data URL format (e.g., "data:image/jpeg;base64,...")
            if image_base64.startswith("data:"):
                # Extract the base64 part after the comma
                image_base64 = image_base64.split(",", 1)[1]
            
            bytes_io = io.BytesIO(base64.b64decode(image_base64))
            pil_image = Image.open(bytes_io).convert("RGB")
            return pil_image
        except Exception as e:
            logging.error(f"Error decoding image from base64: {e}")
            return None

    @staticmethod
    def extract_results(ocr_response, attachment_tag="<attachment>"):
        all_text = []
        embedded_images = []
        
        # Handle both dictionary and object responses
        if hasattr(ocr_response, 'pages'):
            # Object response (real API)
            pages = ocr_response.pages
            for page in pages:
                text = page.markdown
                for image in page.images:
                    text = text.replace(image.id, attachment_tag)
                    decoded_image = CloudPDFProcessor._decode_image(image.image_base64)
                    if decoded_image:
                        embedded_images.append(decoded_image)
                all_text.append(text)
        else:
            # Dictionary response (mocked in tests)
            for page in ocr_response["pages"]:
                text = page['markdown']
                for image in page["images"]:
                    text = text.replace(image['id'], attachment_tag)
                    decoded_image = CloudPDFProcessor._decode_image(image['image_base64'])
                    if decoded_image:
                        embedded_images.append(decoded_image)
                all_text.append(text)
        return all_text, embedded_images

    def api_call_with_retry(self, func, *args, **kwargs):
        max_retries = self.config.custom_config.get('asyncio_max_retries', 3)
        for i in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if i == max_retries:
                    logging.error(f"API call failed after {max_retries + 1} attempts: {e}")
                    raise e
                wait_time = min(2 ** i, 60) # Exponential backoff
                import time
                time.sleep(wait_time)

    def _process_sync(self, file_path: str) -> MultimodalSample:
        """Synchronous version of process method for batch processing"""
        base64_file = CloudPDFProcessor._encode_pdf(file_path)
        if base64_file is None:
            raise Exception(f"Failed to encode PDF: {file_path}")

        ocr_response = self.api_call_with_retry(
            self.mistral_client.ocr.process,
            model=self.config.custom_config.get("mistral_model", "mistral-ocr-latest"),
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_file}"
            },
            include_image_base64=True
        )
        # The response is already a dictionary, not a response object
        all_text, embedded_images = CloudPDFProcessor.extract_results(ocr_response, self.config.attachment_tag)
        return self.create_sample(all_text, embedded_images, file_path)

    async def _process_async(self, file_path: str) -> MultimodalSample:
        """Async version of process method for batch processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_sync, file_path)

    def process(self, file_path: str) -> MultimodalSample:
        """Synchronous wrapper for single file processing"""
        return self._process_sync(file_path)

    def process_batch(
        self, files_paths: List[str], fast_mode: bool = False, num_workers: int = 1
    ) -> List[MultimodalSample]:
        """Process batch of files with rate limiting"""
        if fast_mode:
            raise NotImplementedError

        # Get max API calls per second from config, default to 5
        max_api_calls_per_second = self.config.custom_config.get('max_api_call_per_second', 5)

        async def process_batch_async():
            # Create queue and add all files
            queue = asyncio.Queue()
            for file_path in files_paths:
                queue.put_nowait(file_path)

            # Rate limiting variables
            min_interval = 1.0 / max_api_calls_per_second
            results = []
            item_id = 0
            active_tasks: List[tuple[asyncio.Task, int]] = []

            # Process files with rate limiting
            while not queue.empty() or active_tasks:
                # Start new requests at the specified rate
                while not queue.empty():
                    try:
                        file_path = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    task = asyncio.create_task(
                        self._process_async(file_path)
                    )
                    active_tasks.append((task, item_id))
                    item_id += 1
                    await asyncio.sleep(min_interval)

                # Process completed tasks
                if active_tasks:
                    done, _ = await asyncio.wait(
                        [task for task, _ in active_tasks],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    for task in done:
                        task_id = next(tid for t, tid in active_tasks if t == task)
                        try:
                            result = await task
                            results.append(result)
                        except Exception as e:
                            logging.error(f"Failed to process {files_paths[task_id]}: {str(e)}")
                            results.append(None)

                        active_tasks = [(t, tid) for t, tid in active_tasks if t != task]

                # Sleep for a short interval to avoid busy waiting
                await asyncio.sleep(0.01)

            return results

        # Run the async batch processing
        return asyncio.run(process_batch_async())