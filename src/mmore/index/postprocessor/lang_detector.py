from typing import List
from .base import BasePostProcessor

from mmore.type import MultimodalSample

from langdetect import detect

class LangDetector(BasePostProcessor):
    def __init__(self):
        super().__init__('🗣️ Lang Detector')
    
    def process(self, sample: MultimodalSample, **kwargs) -> MultimodalSample | List[MultimodalSample]:
        # Remove attachments flags (if any)
        text = sample.text.replace("<attachment>", "")

        # Detect language
        lang = detect(text)

        # Add metadata
        sample.metadata['lang'] = lang

        return [sample]