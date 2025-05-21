from collections import defaultdict
from typing import Dict, List
from mmore.type import MultimodalSample
from mmore.process.post_processor.base import BasePostProcessor
from langid import classify
import argostranslate.package
import argostranslate.translate

from dataclasses import dataclass
from enum import Enum

class MetaDataPosition:
    BEGINNING = "beginning"
    END = "end"


@dataclass
class MetaDataInfusorConfig:
    metadata_keys: List[str]
    content_template: str
    position: MetaDataPosition


class MetaDataInfusor(BasePostProcessor):
    def __init__(self, metadata_keys: List[str], content_template: str, position: MetaDataPosition):
        super().__init__(name="☕ Metadata Infusor")
        self.metadata_keys = metadata_keys
        self.content_template = content_template
        self.position = position

    @classmethod
    def from_config(cls, config: MetaDataInfusorConfig):
        metadata_infusor = MetaDataInfusor(
            metadata_keys=config.metadata_keys,
            content_template=config.content_template,
            position=config.position
        )
        return metadata_infusor

    def process(
        self, sample: MultimodalSample, **kwargs
    ) -> MultimodalSample | List[MultimodalSample]:

        format_mapping = defaultdict()
        for key in self.metadata_keys:
            value = sample.metadata.get(key, "")
            format_mapping[key] = value

        metadata_content = self.content_template.format_map(format_mapping)

        match self.position:
            case MetaDataPosition.BEGINNING:
                new_content = metadata_content + sample.text
            case MetaDataPosition.END:
                new_content = sample.text + metadata_content
            case _:
                new_content = metadata_content + sample.text

        return MultimodalSample(new_content, sample.modalities)


