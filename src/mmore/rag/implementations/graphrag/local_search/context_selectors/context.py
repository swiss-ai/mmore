from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from langchain_core.vectorstores import VectorStore

from mmore.index.implementations.graphrag.artifacts import IndexerArtifacts
from mmore.types.graphrag.graphs.community import CommunityLevel

from .communities_reports import CommunitiesReportsSelector
from .entities import EntitiesSelector
from .relationships import RelationshipsSelectionResult, RelationshipsSelector
from .text_units import TextUnitsSelector


class ContextSelectionResult(NamedTuple):
    entities: pd.DataFrame
    text_units: pd.DataFrame
    relationships: RelationshipsSelectionResult
    communities_reports: pd.DataFrame


class ContextSelector:
    def __init__(
        self,
        entities_selector: EntitiesSelector,
        text_units_selector: TextUnitsSelector,
        relationships_selector: RelationshipsSelector,
        communities_reports_selector: CommunitiesReportsSelector,
    ):
        self._entities_selector = entities_selector
        self._text_units_selector = text_units_selector
        self._relationships_selector = relationships_selector
        self._communities_reports_selector = communities_reports_selector

    @staticmethod
    def build_default(
        entities_vector_store: VectorStore,
        entities_top_k: int,
        community_level: CommunityLevel,
    ) -> ContextSelector:
        return ContextSelector(
            entities_selector=EntitiesSelector(
                vector_store=entities_vector_store,
                top_k=entities_top_k,
            ),
            text_units_selector=TextUnitsSelector(),
            relationships_selector=RelationshipsSelector(),
            communities_reports_selector=CommunitiesReportsSelector(
                community_level=community_level
            ),
        )

    def run(
        self,
        query: str,
        artifacts: IndexerArtifacts,
    ):
        # Step 1
        # Select the entities to be used in the local search
        selected_entities = self._entities_selector.run(query, artifacts.entities)

        # Step 2
        # Select the text units to be used in the local search
        selected_text_units = self._text_units_selector.run(
            df_entities=selected_entities,
            df_relationships=artifacts.relationships,
            df_text_units=artifacts.text_units,
        )

        # Step 3
        # Select the relationships to be used in the local search
        selected_relationships = self._relationships_selector.run(
            df_entities=selected_entities,
            df_relationships=artifacts.relationships,
        )

        # Step 4
        # Select the communities to be used in the local search
        selected_communities_reports = self._communities_reports_selector.run(
            df_entities=selected_entities,
            df_reports=artifacts.communities_reports,
        )

        return ContextSelectionResult(
            entities=selected_entities,
            text_units=selected_text_units,
            relationships=selected_relationships,
            communities_reports=selected_communities_reports,
        )
