from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List

from mmore.process.post_processor.base import BasePostProcessor
from mmore.type import MultimodalSample


class MetaDataPosition(Enum):
    BEGINNING = "beginning"
    END = "end"


@dataclass
class MetaDataInfusorConfig:
    metadata_keys: List[str]
    content_template: str
    position: str


class MetaDataInfusor(BasePostProcessor):
    def __init__(
        self,
        metadata_keys: List[str],
        content_template: str,
        position: MetaDataPosition,
    ):
        super().__init__(name="☕ Metadata Infusor")
        self.metadata_keys = metadata_keys
        self.content_template = content_template
        self.position = position

    @classmethod
    def from_config(cls, config: MetaDataInfusorConfig):
        metadata_infusor = MetaDataInfusor(
            metadata_keys=config.metadata_keys,
            content_template=config.content_template,
            position=MetaDataPosition(config.position),
        )
        return metadata_infusor

    def process(self, sample: MultimodalSample, **kwargs) -> List[MultimodalSample]:
        format_mapping = defaultdict()
        for key in self.metadata_keys:
            value = sample.metadata.get(key, "")
            format_mapping[key] = value

        metadata_content = self.content_template.format_map(format_mapping)

        match self.position:
            case MetaDataPosition.BEGINNING:
                new_content = metadata_content + "\n" + sample.text
            case MetaDataPosition.END:
                new_content = sample.text + "\n" + metadata_content
            case _:
                new_content = sample.text

        return [
            MultimodalSample(new_content, sample.modalities, sample.metadata, sample.id)
        ]
