import datetime
import logging
import json
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union
from uuid import uuid4

from src.mmore.process.crawler import FileDescriptor, URLDescriptor
from src.mmore.type import MultimodalSample, MultimodalRawInput
from PIL import Image
import torch
import torch.multiprocessing as mp
import os

logger = logging.getLogger(__name__)

class ProcessorConfig:
    """
    A dataclass that represents the configuration of a processor.
    
    Attributes:
        attachment_tag (str): Tag used for attachments (default: "<attachment>") - This is what we use for Multimodal Meditron.
        custom_config (Dict[str, Any]): Dictionary of custom configurations.
    """

    def __init__(self, attachement_tag: str = "<attachment>", custom_config: Dict[str, Any] = {}, extract_images: bool = True):
        self.attachment_tag = attachement_tag
        self.extract_images = extract_images
        self.custom_config = custom_config


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


class Processor:
    """
    Base class for processors, which process a list of files.

    Attributes:
        files (List[Union[FileDescriptor, URLDescriptor]]): The files to process.
        config (ProcessorConfig): Configuration for the processor.
    """
    IMAGES_DIR = "images"

    def __init__(
            self,
            config: ProcessorConfig = None,
    ):
        """
        Args:
            files (List[Union[FileDescriptor, URLDescriptor]]): The files to process.
            config (ProcessorConfig): Configuration for the processor.
        """

        self.config = config

    def __call__(self, files, fast: bool = False) -> List[MultimodalSample]:
        """
        Process the files, either in fast mode or normal mode.

        Args:
            fast (bool): Whether to use fast processing (default: False).

        Returns:
            List[MultimodalSample]: The result of the processing operation.
        """
        files_paths = [file.file_path for file in files]
        res = self.process_batch(files_paths, fast, num_workers=1) # self.config.num_workers ...
        return res

    def process(self, file_path) -> MultimodalSample:
        """
        Process the files using the standard processing method.

        Returns:
            List[MultimodalSample]: The result of the processing operation.
        """
        raise NotImplementedError
    
    def process_fast(self, file_path) -> List[MultimodalSample]:
        return self.process(file_path)

    def process_batch(self, files_paths, fast_mode, num_workers) -> List[MultimodalSample]:
        """
        Processes a single file using a custom processing method.
            :param file: The file to process.
            :param process_method: The method to use for processing.
        """

        # for all 
        with mp.Pool(processes=num_workers) as pool:
            process = self.process if not fast_mode else self.process_fast
            results = pool.map(process, files_paths)
        
        return results
    
        # def process_files_on_gpu(self, files, gpu_id, fast_mode, results):
        #     """
        #     Process a chunk of files on a specific GPU.
        #     Each process in the multiprocessing pool handles one GPU.
        #     """
        #     # Set the GPU device for this process
        #     device = torch.device(f"cuda:{gpu_id}")
        #     torch.cuda.set_device(device)

        #     # Load models onto the specified GPU
        #     self.load_models(device)

        #     # Process each file using the provided method
        #     for file in files:
        #         try:
        #             result = self.process_one_file(file.file_path, fast=fast_mode)
        #             if type(result) == dict:
        #                 result = [result]
        #             results.append(result)
        #         except Exception as e:
        #             logger.error(f"Failed processing {os.path.basename(file.file_path)}: {str(e)}")
        #             results.append(None)

        #     # Clean up after processing
        #     self.cleanup()

        # num_gpus = torch.cuda.device_count()
        # if num_gpus <= 0:
        #     raise ValueError("No GPUs available.")

        # def split_list_evenly(x_list, nbr_splits):
        #     """
        #     Evenly split a list of things across multiple GPUs.

        #     :param x_list: List of things to split.
        #     :param nbr_splits: Number of GPUs to split across.
        #     :return: List of things split across GPUs.
        #     """
        #     x_per_gpu = len(x_list) // nbr_splits
        #     x_split = [x_list[i * x_per_gpu: (i + 1) * x_per_gpu] for i in range(nbr_splits)]
        #     # NOTE : If the number of things is not divisible by the number of GPUs, the last GPU will get the remainder, and can get 0 to x_per_gpu - 1 things.
        #     # Nevertheless, if we consider that processing each thing takes the same amount of time, this will not be a problem.
        #     return x_split

        # chunks = split_list_evenly(self.files, num_gpus)

        # mp.set_start_method("spawn", force=True)
        # results = mp.Manager().list()

        # processes = []

        # for gpu_id, files in enumerate(chunks):
        #     process = mp.Process(
        #         target=process_files_on_gpu,
        #         args=(files, gpu_id, fast_mode, results),
        #     )
        #     processes.append(process)
        #     process.start()

        # for process in processes:
        #     process.join()

        # results = list(results)
        # results = self.reconstruct_results(results)
        # results = [sample for sample in results if sample is not None]

        # # results can be a list of list of samples, we need to flatten it
        # if isinstance(results[0], list):
        #     results = [
        #         sample
        #         for sublist in results
        #         for sample in sublist
        #         if sample is not None
        #     ]

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        """
        Returns True if the processor can accept the file, False otherwise.
            :param file: The file to check.
        """
        raise NotImplementedError

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

    def create_sample(self, texts: List[str], images: List[Image.Image], file_path) -> MultimodalSample:
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

        def _save_temp_image(image: Image.Image, base_path) -> str:
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
                temp_file = tempfile.NamedTemporaryFile(delete=False, prefix=date_prefix, suffix=".png", dir=temp_dir)
                temp_file_path: str = temp_file.name
                image.save(temp_file_path, format="PNG")
                temp_file.close()
                #if base_path:
                #    return Path(temp_file_path).relative_to(base_path)
                #return temp_file_path
                return temp_file_path
            except Exception as e:
                logger.error(f"Failed to save temporary image: {e}")

        image_base_path = os.path.join(
            self.config.custom_config.get("output_path", None),
            self.IMAGES_DIR
        )

        # create dir if it does not exist 
        os.makedirs(image_base_path, exist_ok=True)

        sample = MultimodalSample(
            "\n".join(texts), 
            [MultimodalRawInput("image", _save_temp_image(img, base_path=image_base_path)) for img in images], \
            {"file_path": file_path} if file_path is not None else None
        )
        return sample