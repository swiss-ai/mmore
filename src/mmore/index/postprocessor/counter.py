from typing import List
from .base import BasePostProcessor

from mmore.type import MultimodalSample

class ModalitiesCounter(BasePostProcessor):
    def __init__(self):
        super().__init__('📸 Modalities Counter')
    
    def process(self, sample: MultimodalSample, **kwargs) -> MultimodalSample | List[MultimodalSample]:
        return [len(sample.modalities)]
    
class WordsCounter(BasePostProcessor):
    def __init__(self):
        super().__init__('🔤 Words Counter')
    
    def process(self, sample: MultimodalSample, **kwargs) -> MultimodalSample | List[MultimodalSample]:
        return [len(sample.text.replace('<attachment>', '').split())]