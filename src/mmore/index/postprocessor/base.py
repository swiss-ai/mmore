from abc import ABC, abstractmethod
from typing import List, Any, Literal

from tqdm import tqdm

from dataclasses import dataclass, field

from mmore.type import MultimodalSample

PP_TYPES = Literal[
    'chunker', 
    'lang_detector', 
    'modalities_counter', 
    'words_counter',
    'ner'
]

@dataclass
class BasePostProcessorConfig:
    type: PP_TYPES
    name: str = None
    args: Any = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.name is None:
            self.name = self.type

class BasePostProcessor(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

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
        res = []

        for s in tqdm(samples, desc=f'Processing samples ({self.name})'):
            pp_s = self.process(s, **kwargs)
            if isinstance(pp_s, MultimodalSample):
                pp_s = [pp_s]
            res.extend(pp_s)

        return res