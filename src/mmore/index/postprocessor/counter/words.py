from typing import List
from ..base import BasePostProcessor

from mmore.type import MultimodalSample

class WordsCounter(BasePostProcessor):
    def __init__(self):
        super().__init__('🔤 Words Counter')
    
    def process(self, sample: MultimodalSample, **kwargs) -> MultimodalSample | List[MultimodalSample]:
        sample.metadata['words_count'] = len(sample.text.split())
        return [sample]