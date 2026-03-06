import logging
import os
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union, cast

import torch
import torch.multiprocessing as mp
from dask.distributed import Client, as_completed
from tqdm import tqdm

from ..type import MultimodalSample
from .crawler import DispatcherReadyResult, FileDescriptor, URLDescriptor
from .execution_state import ExecutionState
from .processors.base import (
    AutoProcessor,
    Processor,
    ProcessorConfig,
    ProcessorRegistry,
)
from .processors.url_processor import URLProcessor

logger = logging.getLogger(__name__)


class ComputeDescriptor:
    @staticmethod
    def get_desc():
        num_gpus = 0
        gpu_size = None

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                gpu_size = torch.cuda.get_device_properties(0).total_memory
                logging.info(
                    f"Detected {num_gpus} GPUs with {gpu_size} bytes of memory."
                )

        return {
            "num_gpus": num_gpus,
            "gpu_size": gpu_size,
        }


@dataclass
class DispatcherConfig:
    """Configuration for the :class:`Dispatcher`.

    Output structure::

        output_path/
        ├── processors/
        │   ├── PDFProcessor/
        │   │   └── results.jsonl
        │   └── ...
        └── merged/
            └── merged_results.jsonl

    Attributes:
        output_path: Directory where processing results are saved.
        use_fast_processors: Use faster but lower-quality processing modes
            where available (default: ``False``).
        extract_images: Extract embedded images from documents
            (default: ``True``).
        distributed: Use distributed processing via Dask (default: ``False``).
        scheduler_file: Path to a Dask scheduler file (only needed when
            ``distributed=True``).
        dashboard_backend_url: Optional URL of the mmore dashboard backend
            for live progress tracking.
        batch_sizes: Per-processor batch size overrides, keyed by processor
            class name.  The value is the maximum number of *document pages*
            per dispatch batch.  Example::

                batch_sizes:
                  PDFProcessor: 4000
                  MediaProcessor: 40

        batch_multiplier: Multiply all batch sizes by this factor (default:
            ``1``).  Useful for scaling up on larger machines.
        processor_configs: Per-processor configuration overrides, keyed by
            processor class name.  Each value is a flat dict of keyword
            arguments passed to the processor's config class.  Example::

                processor_configs:
                  MediaProcessor:
                    normal_model: openai/whisper-large-v3-turbo
                    frame_sample_rate: 10

        file_type_processors: Override which processor handles a given file
            extension.  The key is the extension (including the leading dot),
            the value is a processor class name.  Example::

                file_type_processors:
                  .pdf: PDFProcessor
    """

    output_path: str
    use_fast_processors: bool = False
    extract_images: bool = True
    distributed: bool = False
    scheduler_file: Optional[str] = None
    dashboard_backend_url: Optional[str] = None
    batch_sizes: Dict[str, int] = field(default_factory=dict)
    batch_multiplier: int = 1
    processor_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    file_type_processors: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def from_dict(config: Dict) -> "DispatcherConfig":
        """Create a :class:`DispatcherConfig` from a plain dictionary.

        Accepts both the new format (flat dicts) and the legacy format where
        ``processor_config`` and ``process_batch_sizes`` were lists of
        single-key dicts.
        """
        # Handle legacy list-of-dicts format for batch sizes
        raw_batch_sizes = config.get("batch_sizes") or config.get("process_batch_sizes")
        if isinstance(raw_batch_sizes, list):
            batch_sizes: Dict[str, int] = {
                list(d.keys())[0]: int(list(d.values())[0]) for d in raw_batch_sizes
            }
        else:
            batch_sizes = {k: int(v) for k, v in (raw_batch_sizes or {}).items()}

        # Handle legacy list-of-dicts format for processor configs
        # Accept "processors" (new YAML key), "processor_configs" (old flat dict), or "processor_config" (legacy)
        raw_proc_configs = (
            config.get("processors")
            or config.get("processor_configs")
            or config.get("processor_config")
        )
        if isinstance(raw_proc_configs, dict):
            processor_configs: Dict[str, Dict[str, Any]] = {}
            for proc_name, proc_cfg in raw_proc_configs.items():
                if isinstance(proc_cfg, list):
                    # Legacy: list of single-key dicts
                    processor_configs[proc_name] = {
                        list(d.keys())[0]: list(d.values())[0] for d in proc_cfg
                    }
                else:
                    processor_configs[proc_name] = proc_cfg or {}
        else:
            processor_configs = {}

        return DispatcherConfig(
            output_path=config["output_path"],
            use_fast_processors=config.get("use_fast_processors", False),
            extract_images=config.get("extract_images", True),
            distributed=config.get("distributed", False),
            scheduler_file=config.get("scheduler_file"),
            dashboard_backend_url=config.get("dashboard_backend_url"),
            batch_sizes=batch_sizes,
            batch_multiplier=config.get("batch_multiplier", 1),
            processor_configs=processor_configs,
            file_type_processors=config.get("file_type_processors", {}),
        )

    @staticmethod
    def from_yaml(yaml_path: str) -> "DispatcherConfig":
        import yaml

        try:
            with open(yaml_path, "r") as file:
                config = yaml.safe_load(file)
            return DispatcherConfig.from_dict(config)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"[Dispatcher] Error processing file {yaml_path}")
            raise e

    def to_dict(self) -> Dict:
        return {
            "output_path": self.output_path,
            "use_fast_processors": self.use_fast_processors,
            "extract_images": self.extract_images,
            "distributed": self.distributed,
            "scheduler_file": self.scheduler_file,
            "dashboard_backend_url": self.dashboard_backend_url,
            "batch_sizes": self.batch_sizes,
            "batch_multiplier": self.batch_multiplier,
            "processor_configs": self.processor_configs,
            "file_type_processors": self.file_type_processors,
        }

    def __str__(self) -> str:
        return (
            f"DispatcherConfig("
            f"output_path={self.output_path}, "
            f"use_fast_processors={self.use_fast_processors}, "
            f"extract_images={self.extract_images}, "
            f"distributed={self.distributed}, "
            f"scheduler_file={self.scheduler_file}, "
            f"batch_sizes={self.batch_sizes}, "
            f"batch_multiplier={self.batch_multiplier})"
        )


def _build_processor_config(
    processor: Type[Processor],
    dispatcher_cfg: "DispatcherConfig",
) -> ProcessorConfig:
    """Instantiate the typed config for *processor* from *dispatcher_cfg*.

    The base fields (``output_path``, ``extract_images``,
    ``dashboard_backend_url``) come from the dispatcher config.  Any
    processor-specific overrides from ``dispatcher_cfg.processor_configs``
    are merged in as keyword arguments to the processor's ``CONFIG_CLASS``.

    Args:
        processor: The processor class whose config to build.
        dispatcher_cfg: The dispatcher-level configuration.
    """
    config_cls = getattr(processor, "CONFIG_CLASS", ProcessorConfig)
    proc_overrides = dispatcher_cfg.processor_configs.get(processor.__name__, {})

    return config_cls(
        output_path=dispatcher_cfg.output_path,
        extract_images=dispatcher_cfg.extract_images,
        dashboard_backend_url=dispatcher_cfg.dashboard_backend_url,
        **proc_overrides,
    )


class Dispatcher:
    """Routes crawled files to their appropriate processors and runs them.

    Args:
        result: The output of a :class:`~.crawler.Crawler` run.
        config: Dispatcher configuration.
        start_cluster: Unused; reserved for future use.
    """

    def __init__(
        self,
        result: DispatcherReadyResult,
        config: DispatcherConfig,
        start_cluster: bool = False,
    ):
        self.result = result
        self.config = config
        self.start_cluster = start_cluster
        self.intermediate_map: Dict[Optional[type], List] = {}

    def _bucket_files(self) -> None:
        """Assign each file to a processor.

        Checks ``self.config.file_type_processors`` first; falls back to
        :class:`AutoProcessor` auto-detection.
        """
        processor_map: Dict[Optional[type], List] = {
            processor: [] for processor in ProcessorRegistry.get_processors()
        }

        for file_path_list in self.result.file_paths.values():
            for file in file_path_list:
                ext = file.file_extension.lower()
                if ext in self.config.file_type_processors:
                    proc_name = self.config.file_type_processors[ext]
                    processor = ProcessorRegistry.get_by_name(proc_name)
                    if processor is None:
                        logger.warning(
                            f"Processor '{proc_name}' not found for extension '{ext}', "
                            "falling back to auto-detection."
                        )
                        processor = AutoProcessor.from_file(file)
                else:
                    processor = AutoProcessor.from_file(file)

                logger.debug(
                    f"Assigned file {file.file_path} to processor: {processor}"
                )
                if processor is not None:
                    if processor not in processor_map:
                        processor_map[processor] = []
                    processor_map[processor].append(file)

        url_processor = URLProcessor
        processor_map[url_processor].extend(self.result.urls)

        self.intermediate_map = processor_map

    def _dispatch_local(
        self, task_lists: List[Tuple[Type[Processor], List[FileDescriptor]]]
    ) -> Iterator[List[MultimodalSample]]:
        """Run each processor task sequentially using a shared worker pool.

        Processor instances are cached and reused across batches of the same
        type to avoid redundant initialisation overhead.
        """
        ExecutionState.initialize(distributed_mode=False)

        instantiated_processors: Dict[Type[Processor], Processor] = {}

        num_workers = os.cpu_count() or 1
        logger.info(f"🚀 Initializing Shared Global Pool with {num_workers} workers...")
        global_pool = mp.Pool(processes=num_workers)

        try:
            for processor, files in task_lists:
                if processor not in instantiated_processors:
                    proc_config = _build_processor_config(processor, self.config)
                    logger.info(f"Initializing processor: {processor.__name__}")
                    new_proc = processor(proc_config)
                    new_proc.set_shared_pool(global_pool)
                    instantiated_processors[processor] = new_proc

                proc = instantiated_processors[processor]
                logger.info(
                    f"Processing batch of {len(files)} files "
                    f"({sum(processor.get_file_len(f) for f in files)} pages) "
                    f"with {proc.__class__.__name__}"
                )
                res = proc(
                    cast(List[Union[FileDescriptor, URLDescriptor]], files),
                    self.config.use_fast_processors,
                )
                self.save_individual_processor_results(res, processor.__name__)
                yield res
        finally:
            logger.info("Closing Shared Global Pool")
            global_pool.close()
            global_pool.join()

    def _dispatch_distributed(
        self, task_lists: List[Tuple[Type[Processor], List[FileDescriptor]]]
    ) -> List[List[MultimodalSample]]:
        kwargs = {}
        if self.config.scheduler_file:
            absolute_scheduler_path = os.path.join(
                os.getcwd(), self.config.scheduler_file
            )
            if not os.path.exists(absolute_scheduler_path):
                logger.error(f"Scheduler file {absolute_scheduler_path} does not exist")
            kwargs["scheduler_file"] = absolute_scheduler_path

        client = Client(**kwargs)
        ExecutionState.initialize(distributed_mode=True, client=client)

        futures = []

        for processor, files in task_lists:
            logger.info(
                f"Dispatching in distributed mode {len(files)} files "
                f"({sum(processor.get_file_len(f) for f in files)} pages) "
                f"to {processor.__name__}"
            )

            proc_config = _build_processor_config(processor, self.config)

            def process_files(
                files, proc_config, processor_name, processor_class, use_fast
            ) -> Tuple[List[MultimodalSample], str]:
                client = Client(**kwargs)
                if ExecutionState._use_dask is None:
                    ExecutionState.initialize(distributed_mode=True, client=client)

                worker_count = os.cpu_count() or 1
                task_pool = mp.Pool(processes=worker_count)
                try:
                    proc_instance = processor_class(proc_config)
                    proc_instance.set_shared_pool(task_pool)
                    results = proc_instance(files, use_fast)
                    return results, processor_name
                finally:
                    task_pool.close()
                    task_pool.join()

            try:
                future = client.submit(
                    process_files,
                    files,
                    proc_config,
                    processor.__name__,
                    processor,
                    self.config.use_fast_processors,
                )
                futures.append(future)
            except Exception as e:
                logger.error(f"Error dispatching task to {processor.__name__}: {e}")

        results = []
        for future, (result, processor_name) in tqdm(
            as_completed(futures, with_results=True), total=len(futures)
        ):
            try:
                results.append(result)
                self.save_individual_processor_results(result, processor_name)
            except Exception as e:
                logger.error(f"Error gathering result: {e}")

        return results

    def dispatch(self) -> List[List[MultimodalSample]]:
        """Assign files to processors, batch them, and run all processing.

        Returns:
            A list of result lists, one per processor batch.
        """

        def batch_list(
            lst: List, obj_batch_size: int, processor: Type[Processor]
        ) -> List[List]:
            """Pack *lst* into bins using the best-fit decreasing algorithm."""
            items = [(obj, processor.get_file_len(obj)) for obj in lst]
            items = [item for item in items if item[1] != -1]
            items.sort(key=itemgetter(1), reverse=True)

            batches: List[List] = [[]]
            batch_sizes = [0]

            for obj, size in items:
                best_fit_idx = -1
                min_remaining = obj_batch_size

                for i, batch_size in enumerate(batch_sizes):
                    remaining = obj_batch_size - (batch_size + size)
                    if 0 <= remaining < min_remaining:
                        min_remaining = remaining
                        best_fit_idx = i

                if best_fit_idx >= 0:
                    batches[best_fit_idx].append(obj)
                    batch_sizes[best_fit_idx] += size
                else:
                    batches.append([obj])
                    batch_sizes.append(size)

            return batches

        self._bucket_files()

        task_lists = []
        for processor, file_list in self.intermediate_map.items():
            if processor is not None and len(file_list) > 0:
                default_batch = self.config.batch_sizes.get(processor.__name__, 100)
                effective_batch = self.config.batch_multiplier * default_batch
                batched_files = batch_list(file_list, effective_batch, processor)
                task_lists.extend([(processor, batch) for batch in batched_files])

        results: List[List[MultimodalSample]] = []
        if self.config.distributed:
            results = self._dispatch_distributed(task_lists)
        else:
            results = list(self._dispatch_local(task_lists))

        ExecutionState.shutdown()
        return results

    def __call__(self) -> List[List[MultimodalSample]]:
        return self.dispatch()

    def save_individual_processor_results(
        self, results: List[MultimodalSample], cls_name: str
    ) -> None:
        if not self.config.output_path:
            return

        processor_output_path = os.path.join(
            self.config.output_path, "processors", cls_name
        )
        os.makedirs(processor_output_path, exist_ok=True)
        output_file = os.path.join(processor_output_path, "results.jsonl")
        MultimodalSample.to_jsonl(output_file, results)

        logger.info(f"Results saved to {output_file}")
