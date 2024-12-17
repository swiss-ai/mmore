from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple, Literal, Union

from tqdm import tqdm

from dataclasses import dataclass, field

from mmore.type import MultimodalSample

from .base import BaseFilter

import nltk
nltk.download('punkt_tab', quiet=True)

from datatrove.pipeline.filters.base_filter import BaseFilter as DatatroveBaseFilter
from datatrove.pipeline.filters import (
    SamplerFilter,
    GopherRepetitionFilter,
    GopherQualityFilter,
    FineWebQualityFilter,
    C4QualityFilter,
    LanguageFilter,
    RegexFilter,
    FastTextClassifierFilter,
    LambdaFilter,
    UnigramLogProbFilter,
    URLFilter
)
from datatrove.data import Media, Document
from datatrove.pipeline.writers.jsonl import JsonlWriter

FILTERS_MAP = {
        'language': LanguageFilter,
        'gopher-repetition': GopherRepetitionFilter,
        'gopher-quality': GopherQualityFilter,
        'fineweb': FineWebQualityFilter,
        'c4': C4QualityFilter,
        'sampler': SamplerFilter,
        'regex': RegexFilter,
        'fasttext': FastTextClassifierFilter,
        'lambda': LambdaFilter,
        'unigram-logprob': UnigramLogProbFilter,
        'url': URLFilter,
}

def load_datatrove(filter_name: str, filter_args: Dict[str, Any]) -> DatatroveBaseFilter:
    if filter_name not in FILTERS_MAP:
        raise ValueError(f'Unsupported filter: {filter_name}')

    return FILTERS_MAP[filter_name](**filter_args)

@dataclass
class DatatroveFilterConfig:
    datatrove_name: str
    exclusion_writer: Union[str, JsonlWriter] = None
    datatrove_args: Dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self):
        if isinstance(self.exclusion_writer, str):
            self.exclusion_writer = JsonlWriter(self.exclusion_writer)

    @property
    def filter_args(self):
        return {'exclusion_writer': self.exclusion_writer, **self.datatrove_args}

class DatatroveFilter(BaseFilter):
    datatrove_filter: DatatroveBaseFilter
    
    def __init__(self, name: str, datatrove_filter: DatatroveBaseFilter):
        self.name = name
        self.datatrove_filter = datatrove_filter

    @classmethod
    def from_config(cls, config: DatatroveFilterConfig):
        datatrove_filter = load_datatrove(config.datatrove_name, config.filter_args)
        return cls(name=datatrove_filter.name, datatrove_filter=datatrove_filter)
    
    @staticmethod
    def sample_to_doc(sample: MultimodalSample) -> Document:
        return Document(
            text=sample.text,
            id=sample.id,
            media=[Media(type=modality.type, url=modality.value) for modality in sample.modalities],
            metadata=sample.metadata,
        )


    def filter(self, sample: MultimodalSample) -> bool | Tuple[bool, str]:
        """Abstract method for processing a sample.

        Args:
            sample (MultimodalSample): The sample to process.

        Returns:
            bool: Whether the doc should be kept.
            str: If the document must be ignored, the reason.
        """
        # Filter the document
        return self.datatrove_filter.filter(DatatroveFilter.sample_to_doc(sample))
    
    def batch_filter(self, batch):
        """Abstract method for processing a batch of samples.

        Args:
            batch (List[MultimodalSample]): The batch to process. 
        
        Returns:
            List[bool]: Whether each document should be kept.
        """
        batch = tqdm([DatatroveFilter.sample_to_doc(sample) for sample in batch], 
                     desc=f'{self.name}')
        return self.datatrove_filter.filter_batch(batch)