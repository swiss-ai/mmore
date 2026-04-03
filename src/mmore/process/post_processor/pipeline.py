import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from ...type import MultimodalSample
from ..previous_results import (
    is_reusable_postprocess,
    load_previous_results,
    merge_results,
)
from ..utils import save_samples
from . import BasePostProcessor, BasePostProcessorConfig, load_postprocessor

logger = logging.getLogger(__name__)


@dataclass
class OutputConfig:
    output_path: str
    save_each_step: bool = False
    save_every: int = 100

    def __post_init__(self):
        dirname = os.path.dirname(self.output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)


@dataclass
class PPPipelineConfig:
    pp_modules: List[BasePostProcessorConfig]
    output: OutputConfig
    previous_results: Optional[str] = None


class PPPipeline:
    def __init__(
        self,
        *processors: BasePostProcessor,
        output_config: Optional[OutputConfig] = None,
        previous_results_path: Optional[str] = None,
    ):
        if output_config is None:
            output_config = OutputConfig(output_path="./results")

        self.post_processors = processors
        self.output_config = output_config
        self.previous_results_path = previous_results_path

        # Log the pipeline
        self._log_plan()

    def __add__(self, other):
        return PPPipeline(
            *self.post_processors,
            *other.post_processors,
            output_config=self.output_config,
            previous_results_path=self.previous_results_path,
        )

    def _log_plan(self):
        logger.info("-" * 50)
        logger.info("PP Pipeline:")
        for i, processor in enumerate(self.post_processors):
            logger.info(f"  > {i + 1}. {processor.name}")
        logger.info("-" * 50)

    @classmethod
    def from_config(cls, config: PPPipelineConfig):
        post_processors = [
            load_postprocessor(pp_config) for pp_config in config.pp_modules
        ]
        return cls(
            *post_processors,
            output_config=config.output,
            previous_results_path=config.previous_results,
        )

    def __call__(
        self, samples: List[MultimodalSample], **kwargs
    ) -> List[MultimodalSample]:
        return self.run(samples, **kwargs)

    def run(self, samples: List[MultimodalSample]) -> List[MultimodalSample]:
        """
        Run the post-processing pipeline on a list of multimodal samples.
        The post-processors are applied in sequence.

        Args:
            samples (List[MultimodalSample]): List of multimodal samples.

        Returns:
            List[MultimodalSample]: Post-processed multimodal samples.
        """
        if self.previous_results_path is not None:
            return self._run_incremental(samples)
        return self._run_full(samples)

    def _run_full(self, samples: List[MultimodalSample]) -> List[MultimodalSample]:
        """Run all processors on all samples (original behavior)."""
        output_dir = os.path.dirname(self.output_config.output_path) or "."
        for i, processor in enumerate(self.post_processors):
            tmp_save_path = None
            if self.output_config.save_each_step:
                tmp_save_path = os.path.join(
                    output_dir,
                    f"{i + 1}___{processor.name}.jsonl",
                )
            samples = processor.batch_process(
                samples,
                tmp_save_path=tmp_save_path,
                save_every=self.output_config.save_every,
            )

        processed_at = datetime.now().isoformat()
        for sample in samples:
            sample.metadata["processed_at"] = processed_at

        save_samples(samples, self.output_config.output_path)
        return samples

    def _run_incremental(
        self, samples: List[MultimodalSample]
    ) -> List[MultimodalSample]:
        """Run processors only on samples from new/changed source documents."""
        output_dir = os.path.dirname(self.output_config.output_path) or "."

        previous = load_previous_results(self.previous_results_path)

        # Group input samples by file_path
        groups: dict[str, List[MultimodalSample]] = {}
        for sample in samples:
            fp = sample.metadata.get("file_path", "__unknown__")
            groups.setdefault(fp, []).append(sample)

        current_fps = set(groups.keys())

        # For each group, get max(processed_at) from input samples
        reusable_fps = []
        to_process_fps = []
        for fp, group_samples in groups.items():
            pts = [
                s.metadata.get("processed_at")
                for s in group_samples
                if s.metadata.get("processed_at")
            ]
            if pts:
                input_processed_at = max(pts)
            else:
                # No processed_at on input → treat as new
                to_process_fps.append(fp)
                continue

            if is_reusable_postprocess(fp, input_processed_at, previous):
                reusable_fps.append(fp)
            else:
                to_process_fps.append(fp)

        n_deleted = len(set(previous.keys()) - set(groups.keys()))
        logger.info(
            f"PP incremental: {len(reusable_fps)} reused, "
            f"{len(to_process_fps)} to process, {n_deleted} deleted"
        )

        # Collect reused sample dicts from previous results
        reused: dict[str, List[dict]] = {
            fp: previous[fp] for fp in reusable_fps if fp in previous
        }

        if not to_process_fps:
            logger.info("No document changes detected, reusing all previous results")
            merged_dicts = merge_results(reused, [], current_fps)
            merged_samples = [MultimodalSample.from_dict(d) for d in merged_dicts]
            save_samples(merged_samples, self.output_config.output_path)
            return merged_samples

        # Collect samples to process
        to_process_set = set(to_process_fps)
        samples_to_process = [
            s
            for s in samples
            if s.metadata.get("file_path", "__unknown__") in to_process_set
        ]

        # Run through pipeline
        processed = samples_to_process
        for i, processor in enumerate(self.post_processors):
            tmp_save_path = None
            if self.output_config.save_each_step:
                tmp_save_path = os.path.join(
                    output_dir,
                    f"{i + 1}___{processor.name}_incremental.jsonl",
                )
            processed = processor.batch_process(
                processed,
                tmp_save_path=tmp_save_path,
                save_every=self.output_config.save_every,
            )

        # Enrich newly processed with processed_at
        processed_at = datetime.now().isoformat()
        for sample in processed:
            sample.metadata["processed_at"] = processed_at

        # Convert processed to dicts and merge with reused
        new_dicts = [s.to_dict() for s in processed]
        merged_dicts = merge_results(reused, new_dicts, current_fps)

        # Convert merged dicts back to MultimodalSample and save
        merged_samples = [MultimodalSample.from_dict(d) for d in merged_dicts]
        save_samples(merged_samples, self.output_config.output_path)
        return merged_samples
