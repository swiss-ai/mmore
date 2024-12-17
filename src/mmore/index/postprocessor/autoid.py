from typing import List
from .base import BasePostProcessor

from mmore.type import MultimodalSample

import hashlib

class AutoID(BasePostProcessor):
    def __init__(self):
        super().__init__('#️⃣ Auto ID')

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def process(self, sample: MultimodalSample, **kwargs) -> MultimodalSample | List[MultimodalSample]:
        # Generate an ID for the sample
        sample.metadata['sample_id'] = AutoID.hash_text(sample.text)

        return [sample]