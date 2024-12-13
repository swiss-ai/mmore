from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Literal

from tqdm import tqdm

from dataclasses import dataclass, field

from mmore.type import MultimodalSample

FILTER_TYPES = Literal[
    'datatrove',
]

@dataclass
class BaseFilterConfig:
    type: FILTER_TYPES
    name: str = None
    args: Any = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.name is None:
            self.name = self.type

class BaseFilter(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'

    @abstractmethod
    def filter(self, sample: MultimodalSample) -> bool | Tuple[bool, str]:
        """Abstract method for processing a sample.

        Args:
            sample (MultimodalSample): The sample to process.

        Returns:
            bool: Whether the doc should be kept.
            str: If the document must be ignored, the reason.
        """
        pass

    def batch_filter(self, batch: List[MultimodalSample]) -> List[bool | Tuple[bool, str]]:
        """
        Overwrite this method to implement batched filtering. Batches have size `self.batch_size`, except possibly the last one.
        Args:
            batch: a list of Document to process

        Returns: a list, the same size as `batch`, containing the filter result for each document

        """
        return list(map(self.filter, tqdm(batch, desc=f'[FILTER] {self.name}')))