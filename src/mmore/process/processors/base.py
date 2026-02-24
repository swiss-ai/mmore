import datetime
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

import torch.multiprocessing as mp
from PIL import Image

from ...dashboard.backend.client import DashboardClient
from ...process.crawler import FileDescriptor, URLDescriptor
from ...process.execution_state import ExecutionState
from ...type import MultimodalRawInput, MultimodalSample

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Base configuration shared by all processors.

    Attributes:
        output_path: Directory for saving output files and extracted images.
            The processor will create an ``images/`` subdirectory here.
        extract_images: Whether to extract and save embedded images from
            documents (default: ``True``).
        attachment_tag: Placeholder text inserted in the extracted text at each
            location where an image was found (default: ``"<attachment>"``).
            Used by multimodal models such as Meditron to mark image positions.
        dashboard_backend_url: Optional URL of the mmore dashboard backend for
            live progress tracking. Set to ``None`` to disable (default).
    """

    output_path: str = ""
    extract_images: bool = True
    attachment_tag: str = "<attachment>"
    dashboard_backend_url: Optional[str] = None


@dataclass
class MediaProcessorConfig(ProcessorConfig):
    """Configuration for :class:`MediaProcessor` (audio and video files).

    Inherits all fields from :class:`ProcessorConfig`.

    Attributes:
        normal_model: HuggingFace model ID used for transcription in standard
            mode. Defaults to ``openai/whisper-large-v3-turbo``.
        fast_model: HuggingFace model ID used for transcription in fast mode
            (lower quality, faster). Defaults to ``openai/whisper-tiny``.
        frame_sample_rate: For video files, extract one frame every
            ``frame_sample_rate`` seconds. Defaults to ``10``.

    Example YAML (inside the ``processors:`` block of your config file)::

        processors:
          MediaProcessor:
            normal_model: openai/whisper-large-v3-turbo
            fast_model: openai/whisper-tiny
            frame_sample_rate: 10
    """

    normal_model: str = "openai/whisper-large-v3-turbo"
    fast_model: str = "openai/whisper-tiny"
    frame_sample_rate: int = 10


class ProcessorRegistry:
    """Registry for managing and accessing available processors.

    Attributes:
        _registry (List[type]): List of registered processor classes.
    """

    _registry: List[type] = []

    @classmethod
    def register(cls, processor_class: type, preload: bool = False) -> None:
        """Register a processor class.

        Args:
            processor_class: The processor class to register.
            preload: If ``True``, call ``processor_class.load_models()``
                immediately after registering (default: ``False``).
        """
        cls._registry.append(processor_class)
        if preload:
            processor_class.load_models()

    @classmethod
    def get_processors(cls) -> List[type]:
        """Return all registered processor classes."""
        return cls._registry

    @classmethod
    def get_by_name(cls, name: str) -> Optional[type]:
        """Look up a registered processor by class name.

        Args:
            name: The ``__name__`` of the processor class
                (e.g. ``"PDFProcessor"``).

        Returns:
            The matching processor class, or ``None`` if not found.
        """
        for processor in cls._registry:
            if processor.__name__ == name:
                return processor
        return None


class AutoProcessor:
    @classmethod
    def from_file(cls, file: FileDescriptor) -> Optional[type]:
        """Return the first registered processor that accepts *file*.

        Args:
            file: The file descriptor to match.

        Returns:
            A processor class, or ``None`` if no registered processor
            accepts the file.
        """
        for processor in ProcessorRegistry.get_processors():
            if processor.accepts(file):
                return processor

        logger.warning(f"No registered processor found for file {file}")
        return None


class Processor(ABC):
    """Abstract base class for all file processors.

    Subclasses must implement :meth:`accepts` and :meth:`process`.
    Override :meth:`process_fast` to provide a cheaper processing path.
    Override :meth:`process_batch` for custom batching logic (e.g. multi-GPU).

    Class attributes:
        CONFIG_CLASS: The :class:`ProcessorConfig` subclass used by this
            processor. The dispatcher instantiates this class with the
            per-processor settings from the config file.  Override in
            subclasses that need extra config fields (e.g.
            :class:`MediaProcessorConfig`).
        IMAGES_DIR: Sub-directory name (relative to ``output_path``) where
            extracted images are saved.
    """

    CONFIG_CLASS: Type[ProcessorConfig] = ProcessorConfig
    IMAGES_DIR: str = "images"

    def __init__(self, config: Optional[ProcessorConfig] = None) -> None:
        self.config: ProcessorConfig = (
            config if config is not None else ProcessorConfig()
        )
        self._pool = None
        self._owns_pool = False

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        """Return ``True`` if this processor can handle *file*.

        Args:
            file: The file descriptor to check.
        """
        raise NotImplementedError

    @abstractmethod
    def process(self, file_path: str) -> MultimodalSample:
        """Process a single file and return a :class:`MultimodalSample`.

        Args:
            file_path: Absolute path to the file.
        """
        pass

    def process_fast(self, file_path: str) -> MultimodalSample:
        """Process a single file in fast (lower-quality) mode.

        Falls back to :meth:`process` by default.  Override in processors
        that support a cheaper processing path.

        Args:
            file_path: Absolute path to the file.
        """
        return self.process(file_path)

    def __call__(
        self, files: List[Union[FileDescriptor, URLDescriptor]], fast: bool = False
    ) -> List[MultimodalSample]:
        """Process *files*, optionally in fast mode.

        Args:
            files: Files or URLs to process.
            fast: Use fast mode if ``True`` (default: ``False``).
        """
        if ExecutionState.get_should_stop_execution():
            logger.warning("ExecutionState says to stop, Processor execution aborted")
            return []
        files_paths = [file.file_path for file in files]
        res = self.process_batch(files_paths, fast, num_workers=os.cpu_count() or 1)
        new_state = self.ping_dashboard(files_paths)
        ExecutionState.set_should_stop_execution(new_state)
        return res

    def set_shared_pool(self, pool):
        """
        Injects a shared pool into the processor.
        """
        self._pool = pool
        self._owns_pool = False

    def process_batch(
        self, files_paths: List[str], fast_mode: bool = False, num_workers: int = 1
    ) -> List[MultimodalSample]:
        """Process a list of files, distributing work across *num_workers*.

        Override this method for processors that require custom batching
        (e.g. multi-GPU parallelism in :class:`PDFProcessor`).

        Args:
            files_paths: List of file paths to process.
            fast_mode: Use fast mode if ``True`` (default: ``False``).
            num_workers: Number of parallel worker processes (default: ``1``).
        """
        process_func = self.process_fast if fast_mode else self.process

        if self._pool is not None:
            try:
                return self._pool.map(process_func, files_paths)
            except Exception as e:
                logger.error(f"Error during pool execution: {e}")
                raise
        else:
            logger.info(
                f"⚠️ No shared pool found. Creating temporary pool with {num_workers} workers..."
            )
            with mp.Pool(processes=num_workers) as temp_pool:
                return temp_pool.map(process_func, files_paths)

    def __del__(self):
        if hasattr(self, "_owns_pool") and self._owns_pool and self._pool:
            self._pool.close()
            self._pool.join()

    def __getstate__(self):
        """Remove the pool before pickling (pools cannot be pickled)."""
        state = self.__dict__.copy()
        if "_pool" in state:
            del state["_pool"]
        return state

    def __setstate__(self, state):
        """Restore state after unpickling; workers don't need the pool."""
        self.__dict__.update(state)
        self._pool = None
        self._owns_pool = False

    @classmethod
    def get_file_len(cls, file: FileDescriptor) -> int:
        """Return a size estimate for *file*, used to balance dispatch batches.

        The default returns ``1`` (all files treated equally).  Override in
        processors where file size varies significantly (e.g. PDFs, where
        page count is a better proxy).

        Args:
            file: The file to estimate.
        """
        return 1

    def create_sample(
        self,
        texts: List[str],
        images: List[Image.Image],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MultimodalSample:
        """Build a :class:`MultimodalSample` from extracted text and images.

        Images are saved to ``{output_path}/images/`` as PNG files and
        referenced by path in the returned sample.

        Args:
            texts: List of text strings to join with newlines.
            images: List of PIL images extracted from the document.
            metadata: Optional dict of metadata (e.g. ``{"file_path": ...}``).
        """

        def _save_temp_image(image: Image.Image, base_path: str) -> Optional[str]:
            try:
                temp_dir = os.path.abspath(base_path)
                date_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, prefix=date_prefix, suffix=".png", dir=temp_dir
                )
                temp_file_path: str = temp_file.name
                image.save(temp_file_path, format="PNG")
                temp_file.close()
                return temp_file_path
            except Exception as e:
                logger.error(f"Failed to save temporary image: {e}")
                return None

        image_base_path = os.path.join(self.config.output_path, self.IMAGES_DIR)
        os.makedirs(image_base_path, exist_ok=True)

        sample = MultimodalSample(
            "\n".join(texts),
            [
                MultimodalRawInput("image", tmp_path)
                for img in images
                if (tmp_path := _save_temp_image(img, base_path=image_base_path))
            ],
            metadata if metadata is not None else {},
        )
        return sample

    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Return the size of *file_path* in bytes."""
        return os.path.getsize(file_path)

    def ping_dashboard(self, finished_file_paths: List[str]) -> bool:
        """Report completed files to the dashboard and check for a stop signal.

        Args:
            finished_file_paths: Paths of files just processed.

        Returns:
            ``True`` if execution should stop (dashboard signal), else ``False``.
        """
        if os.environ.get("RANK") is not None:
            worker_id = os.environ.get("RANK")
        else:
            worker_id = os.getpid()
        return DashboardClient(self.config.dashboard_backend_url).report(
            str(worker_id), finished_file_paths
        )

    @staticmethod
    def load_models() -> Any:
        """Pre-load models required by this processor.

        Called by :class:`ProcessorRegistry` when ``preload=True``.  Override
        in processors that benefit from eager model loading (e.g. to avoid
        repeated loads across workers).
        """
        pass
