from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Literal, Optional, cast

from tqdm import tqdm

from dataclasses import dataclass, field

from .. import BasePostProcessor
from ....type import MultimodalSample

@dataclass
class BaseTaggerConfig:
    type: str
    _name: Optional[str] = None
    _metadata_key: Optional[str] = None
    args: Any = field(default_factory=lambda: {})

    def __post_init__(self):
        if self._name is None:
            self._name = self.type
        if self._metadata_key is None:
            self._metadata_key = self.type
    
    @property
    def name(self) -> str:
        return cast(str, self._name)
    
    @property
    def metadata_key(self) -> str:
        return cast(str, self._metadata_key)

class BaseTagger(BasePostProcessor):
    name: str
    metadata_key: str

    def __init__(self, name: str, metadata_key: str):
        self.name = name
        self.metadata_key = metadata_key

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'

    @abstractmethod
    def tag(self, sample: MultimodalSample) -> Any:
        """Abstract method for processing a sample.

        Args:
            sample (MultimodalSample): The sample to process.

        Returns:
            bool: Whether the doc should be kept.
            str: If the document must be ignored, the reason.
        """
        pass

    def batch_tag(self, batch: List[MultimodalSample]) -> List[Any]:
        """
        Overwrite this method to implement batched filtering. Batches have size `self.batch_size`, except possibly the last one.
        Args:
            batch: a list of Document to process

        Returns: a list, the same size as `batch`, containing the filter result for each document

        """
        return list(map(self.tag, tqdm(batch, desc=f'{self.name}')))
    
    def process(self, sample: MultimodalSample, **kwargs) -> MultimodalSample | List[MultimodalSample]:
        tag = self.tag(sample)
        sample.metadata[self.metadata_key] = tag
        return [sample]