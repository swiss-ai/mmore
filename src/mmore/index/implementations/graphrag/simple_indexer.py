"""A simple indexer that sequentially uses various components to index a document.

You could implement a more complex indexer that uses a pipeline, workflow engines
etc to index a document.

"""

from langchain_core.documents import Document

from mmore.types.graphrag.graphs.community import CommunityDetector

from .artifacts import IndexerArtifacts
from .artifacts_generation import (
    CommunitiesReportsArtifactsGenerator,
    EntitiesArtifactsGenerator,
    RelationshipsArtifactsGenerator,
    TextUnitsArtifactsGenerator,
)
from .graph_generation import GraphGenerator
from .text_unit_extractor import TextUnitExtractor


class SimpleIndexer:
    def __init__(
        self,
        text_unit_extractor: TextUnitExtractor,
        graph_generator: GraphGenerator,
        community_detector: CommunityDetector,
        entities_artifacts_generator: EntitiesArtifactsGenerator,
        relationships_artifacts_generator: RelationshipsArtifactsGenerator,
        communities_report_artifacts_generator: CommunitiesReportsArtifactsGenerator,
        text_units_artifacts_generator: TextUnitsArtifactsGenerator,
    ):
        self._text_unit_extractor = text_unit_extractor
        self._graph_generator = graph_generator
        self._community_detector = community_detector
        self._entities_artifacts_generator = entities_artifacts_generator
        self._relationships_artifacts_generator = relationships_artifacts_generator
        self._communities_report_artifacts_generator = (
            communities_report_artifacts_generator
        )
        self._text_units_artifacts_generator = text_units_artifacts_generator

    def run(self, documents: list[Document]) -> IndexerArtifacts:
        # Step 1 - Text Unit extraction
        df_base_text_units = self._text_unit_extractor.run(documents)

        # Step 2 - Generate graphs
        merged_graph, summarized_graph = self._graph_generator.run(df_base_text_units)

        print(len(summarized_graph.nodes(data=True)))

        # Step 3 - Detect communities in Graph
        community_detection_result = self._community_detector.run(summarized_graph)

        # Step 4 - Reports for detected Communities (depends on Step 2 & Step 3)
        df_communities_reports = self._communities_report_artifacts_generator.run(
            community_detection_result,
            summarized_graph,
        )

        for communityLevel in community_detection_result.communities:
            print(len(community_detection_result.communities[communityLevel].keys()))

        # Step 5 - Entities generation (depends on Step 2 & Step 3)
        df_entities = self._entities_artifacts_generator.run(
            community_detection_result,
            summarized_graph,
        )

        # Step 6 - Relationships generation (depends on Step 2)
        df_relationships = self._relationships_artifacts_generator.run(summarized_graph)

        # Step 7 - Text Units generation (depends on Steps 1, 5, 6)
        df_text_units = self._text_units_artifacts_generator.run(
            df_base_text_units,
            df_entities,
            df_relationships,
        )

        return IndexerArtifacts(
            entities=df_entities,
            relationships=df_relationships,
            text_units=df_text_units,
            communities_reports=df_communities_reports,
            summarized_graph=summarized_graph,
            merged_graph=merged_graph,
            communities=community_detection_result,
        )
