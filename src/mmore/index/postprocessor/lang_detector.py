from typing import List
from .base import BasePostProcessor

from mmore.type import MultimodalSample

from langdetect import detect

class LangDetector(BasePostProcessor):
    def __init__(self):
        super().__init__('lang_detector')
    
    def process(self, sample: MultimodalSample, **kwargs) -> MultimodalSample | List[MultimodalSample]:
        # Detect language
        lang = detect(sample.text)

        # Add metadata
        sample.metadata['lang'] = lang

        return [sample]