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


class ProcessorResult:
    """
    Represents the result of a processing operation, containing processed samples.

    Attributes:
        samples (List[MultimodalSample]): List of processed multimodal samples.
    """

    def __init__(self, data: Union[List[Dict[str, Any]], List[MultimodalSample]]):
        """
        Args:
            data (List[Dict[str, Any]]): A list of dictionaries representing processed samples.
        """
        self.samples = []
        for sample in data:
            if isinstance(sample, MultimodalSample):
                self.samples.append(sample)
            else:
                self.samples.append(MultimodalSample.from_dict(sample))

    def to_jsonl(self, path: str, append: bool = False):
        """
        Write the processed samples to a JSONL file.

        Args:
            path (str): The file path to write to.
            append (bool): Whether to append to the file (default: False).
        """
        with open(path, "a" if append else "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

    def __repr__(self) -> str:
        """
        Returns:
            str: A string representation of the processed samples.
        """
        str_text = ""
        for sample in self.samples:
            str_text += json.dumps(sample.to_dict()) + "\n"
        return str_text

    @classmethod
    def merge(cls, others):
        """
        Merge multiple ProcessorResult instances into one.

        Args:
            others (Union[ProcessorResult, List[ProcessorResult]]): A single or list of ProcessorResult instances.

        Returns:
            ProcessorResult: A merged ProcessorResult instance.
        """
        if not isinstance(others, list):
            others = [others]

        samples = []
        for other in others:
            samples.extend(other.samples)

        return ProcessorResult(samples)

    @classmethod
    def from_jsonl(cls, path: str):
        """
        Load a ProcessorResult from a JSONL file.

        Args:
            path (str): The file path to read from.

        Yields:
            MultimodalSample: A sample containing text and modalities.
        """

        with open(path, "r") as f:
            for line in f.readlines():
                yield MultimodalSample.from_dict(json.loads(line))


class ProcessorConfig:
    """
    A dataclass that represents the configuration of a processor.
    
    Attributes:
        attachment_tag (str): Tag used for attachments (default: "<attachment>") - This is what we use for Multimodal Meditron.
        custom_config (Dict[str, Any]): Dictionary of custom configurations.
    """

    def __init__(self, attachement_tag: str = "<attachment>", custom_config: Dict[str, Any] = {}):
        self.attachment_tag = attachement_tag
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

    def __init__(
            self,
            files: List[Union[FileDescriptor, URLDescriptor]],
            config: ProcessorConfig = None,
    ):
        """
        Args:
            files (List[Union[FileDescriptor, URLDescriptor]]): The files to process.
            config (ProcessorConfig): Configuration for the processor.
        """

        self.files = files
        self.config = config

    def __call__(self, fast: bool = False) -> ProcessorResult:
        """
        Process the files, either in fast mode or normal mode.

        Args:
            fast (bool): Whether to use fast processing (default: False).

        Returns:
            ProcessorResult: The result of the processing operation.
        """
        return self.process(fast)

    def __contains__(self, file: FileDescriptor) -> bool:
        """
        Check if the processor accepts a given file.

        Args:
            file (FileDescriptor): The file to check.

        Returns:
            bool: True if the processor accepts the file, False otherwise.
        """
        return self.accepts(file)

    def process(self, fast: bool = False) -> ProcessorResult:
        """
        Process the files using the standard processing method.

        Returns:
            ProcessorResult: The result of the processing operation.
        """
        gpu_required, _ = self.require_gpu()
        try:
            if gpu_required:
                results = self.process_with_gpu(fast_mode=fast)
            else:
                results = self.process_with_cpu(fast_mode=fast)
            return results
        except Exception as e:
            logger.error(f"Failed processing: {str(e)}")
            return ProcessorResult([])

    def process_files_on_gpu(self, files, gpu_id, fast_mode, results):
        """
        Process a chunk of files on a specific GPU.
        Each process in the multiprocessing pool handles one GPU.
        """
        # Set the GPU device for this process
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        # Load models onto the specified GPU
        self.load_models(device)

        # Process each file using the provided method
        for file in files:
            try:
                result = self.process_one_file(file.file_path, fast_mode)
                if type(result) == dict:
                    result = [result]
                results.append(result)
            except Exception as e:
                logger.error(f"Failed processing {os.path.basename(file.file_path)}: {str(e)}")
                results.append(None)

        # Clean up after processing
        self.cleanup()

    def process_with_gpu(self, fast_mode) -> ProcessorResult:
        """
        Processes a single file using a custom processing method.
            :param file: The file to process.
            :param process_method: The method to use for processing.
        """

        def process_files_on_gpu(self, files, gpu_id, fast_mode, results):
            """
            Process a chunk of files on a specific GPU.
            Each process in the multiprocessing pool handles one GPU.
            """
            # Set the GPU device for this process
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)

            # Load models onto the specified GPU
            self.load_models(device)

            # Process each file using the provided method
            for file in files:
                try:
                    result = self.process_one_file(file.file_path, fast=fast_mode)
                    if type(result) == dict:
                        result = [result]
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed processing {os.path.basename(file.file_path)}: {str(e)}")
                    results.append(None)

            # Clean up after processing
            self.cleanup()

        num_gpus = torch.cuda.device_count()
        if num_gpus <= 0:
            raise ValueError("No GPUs available.")

        def split_list_evenly(x_list, nbr_splits):
            """
            Evenly split a list of things across multiple GPUs.

            :param x_list: List of things to split.
            :param nbr_splits: Number of GPUs to split across.
            :return: List of things split across GPUs.
            """
            x_per_gpu = len(x_list) // nbr_splits
            x_split = [x_list[i * x_per_gpu: (i + 1) * x_per_gpu] for i in range(nbr_splits)]
            # NOTE : If the number of things is not divisible by the number of GPUs, the last GPU will get the remainder, and can get 0 to x_per_gpu - 1 things.
            # Nevertheless, if we consider that processing each thing takes the same amount of time, this will not be a problem.
            return x_split

        chunks = split_list_evenly(self.files, num_gpus)

        mp.set_start_method("spawn", force=True)
        results = mp.Manager().list()

        processes = []

        for gpu_id, files in enumerate(chunks):
            process = mp.Process(
                target=self.process_files_on_gpu,
                args=(files, gpu_id, fast_mode, results),
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        results = list(results)
        results = self.reconstruct_results(results)
        results = [sample for sample in results if sample is not None]

        # results can be a list of list of samples, we need to flatten it
        if isinstance(results[0], list):
            results = [
                sample
                for sublist in results
                for sample in sublist
                if sample is not None
            ]
        return ProcessorResult(results)

    def process_with_cpu(self, fast_mode) -> ProcessorResult:
        num_cores = mp.cpu_count()
        with mp.Pool(processes=num_cores) as pool:
            process = partial(self.process_one_file, fast=fast_mode)
            paths = [file.file_path for file in self.files]
            results = pool.map(process, paths)
        return ProcessorResult(results)

    def load_models(self, device: str):
        """
        Loads the models onto the specified device.
            :param device: The device to load the models onto.
        """
        raise NotImplementedError

    def process_one_file(self, file_path: str, fast: bool = False) -> ProcessorResult:
        """
        Processes a single file in a standard way.
            :param file: The file to process.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return self.create_sample([], [], file_path)

    def require_gpu(self) -> bool:
        """
        Returns if the processor requires a GPU
        """
        raise NotImplementedError

    def accepts(self, file: FileDescriptor) -> bool:
        """
        Returns True if the processor can accept the file, False otherwise.
            :param file: The file to check.
        """
        raise NotImplementedError

    def get_file_len(self, file: FileDescriptor) -> int:
        """
          Used for dispatching.
          For files with unequal size distribution, this helps dispatch tasks
          more appropriately based on the computation size it represents.

          Specifically used in PDFProcessor.

          Args:
              file (FileDescriptor): The file to be processed.
          """
        return 1

    def create_sample(self, texts: List[str], images: List[Image.Image], path) -> ProcessorResult:
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

        def _save_temp_image(image: Image.Image, base_path=None) -> str:
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
                temp_dir = base_path or tempfile.gettempdir()
                date_prefix = datetime.now().strftime("%Y-%m-%d_")
                temp_file = tempfile.NamedTemporaryFile(delete=False, prefix=date_prefix, suffix=".png", dir=temp_dir)
                temp_file_path: str = temp_file.name
                image.save(temp_file_path, format="PNG")
                temp_file.close()
                if base_path:
                    return Path(temp_file_path).relative_to(base_path)
                return temp_file_path
            except Exception as e:
                logger.error(f"Failed to save temporary image: {e}")

        base_path = os.environ.get("MMORE_RESULTS_PATH", None)
        if base_path is not None:
            base_path = Path(base_path) / str(uuid4())
            base_path.mkdir(exist_ok=True)

        sample = {
            "text": "\n".join(texts),
            "modalities": [
                {"type": "image", "value": _save_temp_image(img, base_path=base_path)}
                for img in images
            ],
            "metadata": {"file_path": path},
        }

        if path is None:
            sample.pop("metadata")

        if base_path:
            try:
                with open(base_path / "descriptor.json", "w") as f:
                    json.dump(sample, f)

                return ProcessorResult([sample])
            except:
                pass

        return ProcessorResult([sample])
