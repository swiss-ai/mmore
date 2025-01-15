from __future__ import annotations

import logging

from langchain_core.documents import Document

from mmore.rag.implementations.graphrag.local_search.context_selectors import (
    ContextSelectionResult,
)
from mmore.types.graphrag.tokens import TokenCounter

from .communities_reports import CommunitiesReportsContextBuilder
from .entities import EntitiesContextBuilder
from .relationships import RelationshipsContextBuilder
from .text_units import TextUnitsContextBuilder

_LOGGER = logging.getLogger(__name__)


class ContextBuilder:
    def __init__(
        self,
        entities_context_builder: EntitiesContextBuilder,
        realtionships_context_builder: RelationshipsContextBuilder,
        text_units_context_builder: TextUnitsContextBuilder,
        communities_reports_context_builder: CommunitiesReportsContextBuilder,
    ):
        self._entities_context_builder = entities_context_builder
        self._relationships_context_builder = realtionships_context_builder
        self._text_units_context_builder = text_units_context_builder
        self._communities_reports_context_builder = communities_reports_context_builder

    @staticmethod
    def build_default(token_counter: TokenCounter) -> ContextBuilder:
        return ContextBuilder(
            entities_context_builder=EntitiesContextBuilder(
                token_counter=token_counter,
            ),
            realtionships_context_builder=RelationshipsContextBuilder(
                token_counter=token_counter,
            ),
            text_units_context_builder=TextUnitsContextBuilder(
                token_counter=token_counter,
            ),
            communities_reports_context_builder=CommunitiesReportsContextBuilder(
                token_counter=token_counter,
            ),
        )

    def __call__(self, result: ContextSelectionResult) -> list[Document]:
        entities_document = self._entities_context_builder(result.entities)
        relationships_document = self._relationships_context_builder(
            result.relationships
        )
        text_units_document = self._text_units_context_builder(result.text_units)
        communities_reports_document = self._communities_reports_context_builder(
            result.communities_reports
        )

        documents = [
            entities_document,
            relationships_document,
            text_units_document,
            communities_reports_document,
        ]

        if _LOGGER.isEnabledFor(logging.DEBUG):
            import tableprint

            rows = []
            tableprint.banner("Context Token Usage")
            for name, doc in zip(
                ["Entities", "Relationships", "Text Units", "Communities Reports"],
                [
                    entities_document,
                    relationships_document,
                    text_units_document,
                    communities_reports_document,
                ],
                strict=True,
            ):
                rows.append([name, doc.metadata["token_count"]])

            tableprint.table(rows, ["Context", "Token Count"])

        return documents
