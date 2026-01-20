from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tqdm import tqdm

from ...type import MultimodalSample
from ..utils import save_samples


@dataclass
class BasePostProcessorConfig:
    type: str
    name: Optional[str] = None
    args: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name is None:
            self.name = self.type


class BasePostProcessor(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __call__(
        self, sample: MultimodalSample, **kwargs
    ) -> MultimodalSample | List[MultimodalSample]:
        return self.process(sample, **kwargs)

    @abstractmethod
    def process(self, sample: MultimodalSample, **kwargs) -> List[MultimodalSample]:
        """Abstract method for processing a sample.

        Args:
            sample (MultimodalSample): The sample to process.

        Returns:
            List[MultimodalSample]: The processed sample(s).
        """
        pass

    def batch_process(
        self,
        samples: List[MultimodalSample],
        tmp_save_path: Optional[str] = None,
        **kwargs,
    ) -> List[MultimodalSample]:
        """
        Process a batch of samples.
        Args:
            samples: a list of samples to process
            kwargs: additional arguments to pass to the process method

        Returns: a list of processed samples
        """
        res = []
        for s in tqdm(samples, desc=f"{self.name}"):
            new = self.process(s, **kwargs)
            if tmp_save_path and len(res) > 0 and len(res) % 100 == 0:
                save_samples(new, tmp_save_path, append_mode=True)

            res += new

        if tmp_save_path:
            save_samples(new, tmp_save_path, append_mode=True)

        return res
