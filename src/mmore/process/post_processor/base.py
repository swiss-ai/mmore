from abc import ABC, abstractmethod
from typing import Dict, List, Any, Literal, Optional, cast

from tqdm import tqdm
from dataclasses import dataclass, field

from ...type import MultimodalSample

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
        return f'{self.__class__.__name__}({self.name})'

    def __call__(self, sample: MultimodalSample, **kwargs) -> MultimodalSample | List[MultimodalSample]:
        return self.process(sample, **kwargs)

    @abstractmethod
    def process(self, sample: MultimodalSample, **kwargs) -> MultimodalSample | List[MultimodalSample]:
        """Abstract method for processing a sample.

        Args:
            sample (MultimodalSample): The sample to process.

        Returns:
            MultimodalSample | List[MultimodalSample]: The processed sample(s).
        """
        pass

    def batch_process(self, samples: List[MultimodalSample], **kwargs) -> List[MultimodalSample]:
        """
        Process a batch of samples.
        Args:
            samples: a list of samples to process
            kwargs: additional arguments to pass to the process method

        Returns: a list of processed samples
        """
        res = []

        for s in tqdm(samples, desc=f'{self.name}'):
            pp_s = self.process(s, **kwargs)
            if isinstance(pp_s, MultimodalSample):
                pp_s = [pp_s]
            res.extend(pp_s)

        return res