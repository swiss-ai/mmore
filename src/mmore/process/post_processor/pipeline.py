import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from ...type import MultimodalSample
from ..utils import save_samples
from . import BasePostProcessor, BasePostProcessorConfig, load_postprocessor

logger = logging.getLogger(__name__)


@dataclass
class OutputConfig:
    output_path: str
    save_each_step: bool = False

    def __post_init__(self):
        dirname = os.path.dirname(self.output_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)


@dataclass
class PPPipelineConfig:
    pp_modules: List[BasePostProcessorConfig]
    output: OutputConfig


class PPPipeline:
    def __init__(
        self,
        *processors: BasePostProcessor,
        output_config: Optional[OutputConfig] = None,
    ):
        if output_config is None:
            output_config = OutputConfig(output_path="./results")

        self.post_processors = processors
        self.output_config = output_config

        # Log the pipeline
        self._log_plan()

    def __add__(self, other):
        return PPPipeline(
            *self.post_processors,
            *other.post_processors,
            output_config=self.output_config,
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
        return cls(*post_processors, output_config=config.output)

    def __call__(self, samples: List[MultimodalSample]) -> List[MultimodalSample]:
        return self.run(samples)

    def run(self, samples: List[MultimodalSample]) -> List[MultimodalSample]:
        """
        Run the post-processing pipeline on a list of multimodal samples.

        Args:
            samples (List[MultimodalSample]): List of multimodal samples.

        Returns:
            List[MultimodalSample]: Post-processed multimodal samples.
        """
        for i, processor in enumerate(self.post_processors):
            tmp_save_path = None
            if self.output_config.save_each_step:
                tmp_save_path = os.path.join(
                    os.path.dirname(self.output_config.output_path),
                    f"{i + 1}___{processor.name}.jsonl",
                )

            samples = processor.batch_process(samples, tmp_save_path=tmp_save_path)
            if self.output_config.save_each_step:
                self.save_results(samples, f"{i + 1}___{processor.name}.jsonl")
        self.save_results(samples, "final_pp.jsonl")
        return samples

    def save_results(self, samples: List[MultimodalSample], filename: str) -> None:
        """
        Save multimodal samples to a JSONL file.

        Args:
            samples (List[MultimodalSample]): List of multimodal samples.
            output_path (str): Path to save the samples.
        """

        save_samples(samples, os.path.join(self.output_config.output_path, filename))
