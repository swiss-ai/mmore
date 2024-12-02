from abc import ABC, abstractmethod
from typing import List, Any

from mmore.type import MultimodalSample

class BasePostProcessor(ABC):
    def __init__(self, name: str):
        super().__init__()
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

        for s in samples:
            pp_s = self.process(s, **kwargs)
            if isinstance(pp_s, MultimodalSample):
                pp_s = [pp_s]
            res.extend(pp_s)

        return res