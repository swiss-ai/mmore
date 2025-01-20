from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple, Literal, Union

from tqdm import tqdm

from dataclasses import dataclass, field

from mmore.type import MultimodalSample

from .base import BaseFilter, BaseFilterConfig

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
        'filter_language': LanguageFilter,
        'filter_gopher-repetition': GopherRepetitionFilter,
        'filter_gopher-quality': GopherQualityFilter,
        'filter_fineweb': FineWebQualityFilter,
        'filter_c4': C4QualityFilter,
        'sampler': SamplerFilter,
        'filter_regex': RegexFilter,
        'filter_fasttext': FastTextClassifierFilter,
        'filter_lambda': LambdaFilter,
        'filter_unigram-logprob': UnigramLogProbFilter,
        'filter_url': URLFilter,
}
DATATROVE_FILTERS = list(FILTERS_MAP.keys())

def load_datatrove_filter(filter_name: str, filter_args: Dict[str, Any]) -> DatatroveBaseFilter:
    if filter_name not in FILTERS_MAP:
        raise ValueError(f'Unsupported filter: {filter_name}')
    if 'exclusion_writer' in filter_args and isinstance(filter_args['exclusion_writer'], str):
        filter_args['exclusion_writer'] = JsonlWriter(filter_args['exclusion_writer'])
    return FILTERS_MAP[filter_name](**filter_args)

class DatatroveFilter(BaseFilter):
    datatrove_filter: DatatroveBaseFilter
    
    def __init__(self, name: str, datatrove_filter: DatatroveBaseFilter):
        super().__init__(name)
        self.datatrove_filter = datatrove_filter

    @classmethod
    def from_config(cls, config: BaseFilterConfig) -> 'DatatroveFilter':
        datatrove_filter = load_datatrove_filter(config.type, config.args)
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
        res = self.datatrove_filter.filter(DatatroveFilter.sample_to_doc(sample))
        if isinstance(res, bool):
            return res
        else:
            return res[0]
        #return self.datatrove_filter.filter(DatatroveFilter.sample_to_doc(sample))
    
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