import logging
import json
from typing import Any, Dict, List, Union
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

    def __init__(self, data: List[Dict[str, Any]]):
        """
        Args:
            data (List[Dict[str, Any]]): A list of dictionaries representing processed samples.
        """
        self.samples = []
        for sample in data:
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
        return self.process_fast() if fast else self.process()

    def __contains__(self, file: FileDescriptor) -> bool:
        """
        Check if the processor accepts a given file.

        Args:
            file (FileDescriptor): The file to check.

        Returns:
            bool: True if the processor accepts the file, False otherwise.
        """
        return self.accepts(file)

    def process(self) -> ProcessorResult:
        """
        Process the files using the standard processing method.

        Returns:
            ProcessorResult: The result of the processing operation.
        """
        gpu_required, _ = self.require_gpu()
        try:
            results = self.process_with_gpu(fast_mode=False) if gpu_required else self.process_with_cpu(fast_mode=False)
            return self._consolidate_modalities(
                results)  # Consolidate modalities: ensure that the files will be available after processing
        except Exception as e:
            return ProcessorResult([])

    def process_fast(self) -> ProcessorResult:
        """
        Process the files using the fast processing method.

        Returns:
            ProcessorResult: The result of the fast processing operation.
        """
        try:
            _, gpu_required = self.require_gpu()
            results = self.process_with_gpu(fast_mode=True) if gpu_required else self.process_with_cpu(fast_mode=True)
            return self._consolidate_modalities(
                results)  # Consolidate modalities: ensure that the files will be available after processing
        except NotImplementedError:
            logger.warning(
                f"Fast processing not implemented for {self.__class__.__name__}, falling back to generic processing."
            )
            return self.process()

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
                if fast_mode:
                    process_method = self.process_fast_implementation
                else:
                    process_method = self.process_implementation
                result = process_method(file.file_path)
                if type(result) == dict:
                    result = [result]
                results.append(result)
            except Exception as e:
                logger.error(f"Failed processing {os.path.basename(file.file_path)}: {str(e)}")
                results.append(None)

        # Clean up after processing
        self.cleanup()

    def process_with_gpu(self, fast_mode) -> Dict[str, Any]:
        """
        Processes a single file using a custom processing method.
            :param file: The file to process.
            :param process_method: The method to use for processing.
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus <= 0:
            raise ValueError("No GPUs available.")
        
        chunks = self.split_files_across_gpus()
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
            if fast_mode:
                process_method = self.process_fast_implementation
            else:
                process_method = self.process_implementation
            results = pool.map(process_method, [file.file_path for file in self.files])
        self.cleanup()  # Clean up the processor, if necessary (useful for quitting selenium drivers etc.)
        return ProcessorResult(results)

    def reconstruct_results(self, results):
        return results

    def load_models(self, device: str):
        """
        Loads the models onto the specified device.
            :param device: The device to load the models onto.
        """
        raise NotImplementedError

    def process_implementation(self, file_path):
        """
        Processes a single file in a standard way.
            :param file: The file to process.
        """
        raise NotImplementedError

    def process_fast_implementation(self, file_path):
        """
        Processes a single file in a fast way.
            :param file: The file to process.
        """
        logger.debug(f"No fast implementation, using standard for {os.path.basename(file_path)}")
        return self.process_implementation(file_path)

    def require_gpu(self) -> bool:
        """
        Returns if the procesor and the fast processor require a GPU.
        """
        raise NotImplementedError

    def split_files_across_gpus(self) -> List[List[FileDescriptor]]:
        """
        Splits the files across the available GPUs.
        """
        raise NotImplementedError

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        """
        Returns True if the processor can accept the file, False otherwise.
            :param file: The file to check.
        """
        raise NotImplementedError

    def cleanup(self):
        """
        Cleans up the processor.
        """
        pass

    @classmethod
    def compute_rank(cls, list_of_files: List[FileDescriptor]) -> int:
        """
        Computes the rank of the files (a measure of how much processing is required).
            :param list_of_files: The list of files to compute the size of.
        """
        return len(list_of_files)  # Default rank is the number of files

    @classmethod
    def get_file_len(cls, file: FileDescriptor) -> int:
        """
        Returns the length of the file.
            :param file: The file to check.
        """
        # Especially useful for large file who require more processing
        return 1

    def _consolidate_modalities(self, result: ProcessorResult) -> ProcessorResult:
        """
        Consolidates modalities in the processing result by updating file paths and copying files 
        (e.g., images) to a specified output directory.

        Args:
            result (ProcessorResult): The result containing samples with modalities to be consolidated.

        Returns:
            ProcessorResult: A new ProcessorResult object with updated modalities paths.
        """
        # For each sample modalities, we need to copy the modalities files to the output folder
        # and update the modalities paths in the sample
        def save_new_image(image_path: str, output_folder_path: str) -> str:
            try:
                image = Image.open(image_path)
                new_path = os.path.join(output_folder_path, "images", os.path.basename(image_path))
                image.save(new_path)
                return new_path
            except Exception as e:
                logger.error(f"Failed to save image {os.path.basename(image_path)}: {str(e)}")
                return None

        # Create a new list of samples with updated modalities paths
        new_samples = []
        for sample in result.samples:
            os.makedirs(os.path.join(self.config.custom_config.get("output_path", "output"), "images"), exist_ok=True)
            old_modalities = sample.modalities
            new_modalities = []
            for modality in old_modalities:
                modality_type = modality.type
                modality_value = modality.value
                if modality_type == "image":
                    new_modality_value = save_new_image(modality_value, self.config.custom_config.get("output_path", "output"))
                    new_modalities.append(MultimodalRawInput(modality_type, new_modality_value))
                else:
                    new_modalities.append(modality)
            new_sample = MultimodalSample(
                sample.text,
                new_modalities,
                sample.metadata)
            new_samples.append(new_sample)
        return ProcessorResult(new_samples)
