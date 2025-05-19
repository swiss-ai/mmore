import datetime
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch.multiprocessing as mp
from PIL import Image

from ...dashboard.backend.client import DashboardClient
from ...process.crawler import FileDescriptor, URLDescriptor
from ...process.execution_state import ExecutionState
from ...type import MultimodalRawInput, MultimodalSample

logger = logging.getLogger(__name__)


class ProcessorConfig:
    """
    A dataclass that represents the configuration of a processor.

    Attributes:
        attachment_tag (str): Tag used for attachments (default: "<attachment>") - This is what we use for Multimodal Meditron.
        custom_config (Dict[str, Any]): Dictionary of custom configurations.
    """

    def __init__(
        self,
        attachement_tag: str = "<attachment>",
        dashboard_backend_url: Optional[str] = None,
        custom_config: Dict[str, Any] = {},
    ):
        self.attachment_tag = attachement_tag
        self.dashboard_backend_url = dashboard_backend_url
        self.custom_config = custom_config
        self.custom_config["attachment_tag"] = attachement_tag


class ProcessorRegistry:
    """
    Registry for managing and accessing available processors.

    Attributes:
        _registry (List[type]): List of registered processor classes.
    """

    _registry = []

    @classmethod
    def register(cls, processor_class):
        """
        Register a processor class.
        """
        cls._registry.append(processor_class)

    @classmethod
    def get_processors(cls):
        """
        Returns a list of all registered processors.
        """
        return cls._registry


class AutoProcessor:
    @classmethod
    def from_file(cls, file: FileDescriptor):
        """
        Determine and return the appropriate processor for the given file.

        Args:
            file (FileDescriptor): The file descriptor to process.

        Returns:
            Processor: The appropriate processor for the file, or None if no processor is found.
        """

        for processor in ProcessorRegistry.get_processors():
            if processor.accepts(file):
                return processor

        logger.warning(f"No registered processor found for file {file}")
        return None


class Processor(ABC):
    """
    Base class for processors, which process a list of files.

    Attributes:
        files (List[Union[FileDescriptor, URLDescriptor]]): The files to process.
        config (ProcessorConfig): Configuration for the processor.
    """

    IMAGES_DIR: str = "images"

    def __init__(
        self,
        config: ProcessorConfig,
    ):
        """
        Args:
            files (List[Union[FileDescriptor, URLDescriptor]]): The files to process.
            config (ProcessorConfig): Configuration for the processor.
        """

        self.config = config

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        """
        Returns True if the processor can accept the file, False otherwise.
            :param file: The file to check.
        """
        raise NotImplementedError

    @abstractmethod
    def process(self, file_path) -> MultimodalSample:
        """
        ABSTRACT METHOD:
        Process a single file and return the result.

        Args:
            file_path (str): The path to the file to process.

        Returns:
            MultimodalSample: The result of the processing operation.
        """
        pass

    def process_fast(self, file_path: str) -> MultimodalSample:
        """
        Process a single file in fast mode and return the result.
        This method should be overwritten if a processor supports fast mode.

        Args:
            file_path (str): The path to the file to process.

        Returns:
            MultimodalSample: The result of the processing operation.
        """
        return self.process(file_path)

    def __call__(
        self, files: List[Union[FileDescriptor, URLDescriptor]], fast: bool = False
    ) -> List[MultimodalSample]:
        """
        Process the files, either in fast mode or normal mode.

        Args:
            files (List[Union[FileDescriptor, URLDescriptor]]): The files to process.
            fast (bool): Whether to use fast processing (default: False).

        Returns:
            List[MultimodalSample]: The result of the processing operation.
        """
        if ExecutionState.get_should_stop_execution():
            logger.warning("ExecutionState says to stop, Processor execution aborted")
            return []
        files_paths = [file.file_path for file in files]
        res = self.process_batch(files_paths, fast, num_workers=os.cpu_count() or 1)
        new_state = self.ping_dashboard(files_paths)
        ExecutionState.set_should_stop_execution(new_state)
        return res

    def process_batch(
        self, files_paths: List[str], fast_mode: bool = False, num_workers: int = 1
    ) -> List[MultimodalSample]:
        """
        Processes a single file using a custom processing method.
        This method should be overwritten if a processor supports custom batch processing.

        Args:
            file_path (str): The path to the file to process.
            fast_mode (bool): Whether to use fast processing (default: False).
            num_workers (int): Number of workers to use for multiprocessing (default: 1).

        Returns:
            MultimodalSample: The result of the processing operation.
        """
        # use fast mode if user requests it
        process_func = self.process_fast if fast_mode else self.process

        # process files in parallel using multiprocessing pool
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(process_func, files_paths)

        return results

    @classmethod
    def get_file_len(cls, file: FileDescriptor) -> int:
        """
        Used for dispatching.
        For files with unequal size distribution, this helps dispatch tasks
        more appropriately based on the computation size it represents.

        Specifically used in PDFProcessor.

        Args:
            file (FileDescriptor): The file to be processed.
        """
        return 1

    def create_sample(
        self, texts: List[str], images: List[Image.Image], file_path
    ) -> MultimodalSample:
        """
        Create a sample dictionary containing text, images, and optional metadata.
        This function is called within all processors.

        Args:
            texts (List[str]): List of text strings.
            images (List[Image.Image]): List of images.
            path (str, optional): Path for metadata. Defaults to None.

        Returns:
            dict: Sample dictionary with text, image modalities, and metadata.
        """

        def _save_temp_image(image: Image.Image, base_path) -> Optional[str]:
            """
            Save an image as a temporary file.

            Args:
                image (Image.Image): Image to save.
                base_path (str, optional): Base directory for saving the file.

            Returns:
                str: Path to the saved image.
            """
            try:
                # use systems temp dir if no path is provided
                temp_dir = os.path.abspath(base_path)
                date_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, prefix=date_prefix, suffix=".png", dir=temp_dir
                )
                temp_file_path: str = temp_file.name
                image.save(temp_file_path, format="PNG")
                temp_file.close()
                # if base_path:
                #    return Path(temp_file_path).relative_to(base_path)
                # return temp_file_path
                return temp_file_path
            except Exception as e:
                logger.error(f"Failed to save temporary image: {e}")

        image_base_path = os.path.join(
            self.config.custom_config.get("output_path", ""), self.IMAGES_DIR
        )

        # create dir if it does not exist
        os.makedirs(image_base_path, exist_ok=True)

        sample = MultimodalSample(
            "\n".join(texts),
            [
                MultimodalRawInput("image", tmp_path)
                for img in images
                if (tmp_path := _save_temp_image(img, base_path=image_base_path))
            ],
            {"file_path": file_path} if file_path is not None else dict(),
        )
        return sample

    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        Get size of the file (in bytes).
        """
        return os.path.getsize(file_path)

    def ping_dashboard(self, finished_file_paths) -> bool:
        """
        Sends a ping to the dashboard to indicate that the processing is complete.
        Opportunity to check if the processing should be stopped (the idea is to not do this at the beginning of the process for tradeoff/performance reasons).
        """
        if os.environ.get("RANK") is not None:
            worker_id = os.environ.get("RANK")
        else:
            worker_id = os.getpid()
        return DashboardClient(self.config.dashboard_backend_url).report(
            str(worker_id), finished_file_paths
        )
