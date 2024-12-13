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
        try:
            # Detect language
            lang = detect(text)
        except:
            lang = "unknown"

        # Add metadata
        sample.metadata['lang'] = lang

        return [sample]