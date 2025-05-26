from typing import List

from ....process.post_processor import BasePostProcessor
from ....type import MultimodalSample
from .extractor import NERExtractor, NERExtractorConfig


class NERecognizer(BasePostProcessor):
    def __init__(self, extractor: NERExtractor):
        super().__init__("🔎 NER")
        self._extractor = extractor

    @classmethod
    def from_config(cls, config: NERExtractorConfig):
        extractor = NERExtractor.from_config(config)
        return cls(extractor)

    def process(self, sample: MultimodalSample, **kwargs) -> List[MultimodalSample]:
        # Call the extractor to get the relation graph
        relation_graph = self._extractor.invoke(sample)

        # Convert the relation graph to a list of relations
        entities = [
            {"entity": e, **entity_desc}
            for e, entity_desc in relation_graph.nodes(data=True)
        ]

        # Add the relations to the sample metadata
        sample.metadata["ner"] = entities

        return [sample]
