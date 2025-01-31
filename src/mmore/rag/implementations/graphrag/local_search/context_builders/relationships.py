import logging

import pandas as pd
from langchain_core.documents import Document

from mmore.rag.implementations.graphrag.local_search.context_selectors.relationships import (
    RelationshipsSelectionResult,
)
from mmore.types.graphrag.tokens import TokenCounter

_LOGGER = logging.getLogger(__name__)


class RelationshipsContextBuilder:
    def __init__(
        self,
        *,
        include_weight: bool = True,
        context_name: str = "Relationships",
        column_delimiter: str = "|",
        max_tokens: int = 8000,
        token_counter: TokenCounter,
    ):
        self._include_weight = include_weight
        self._context_name = context_name
        self._column_delimiter = column_delimiter
        self._max_tokens = max_tokens
        self._token_counter = token_counter

    def __call__(
        self,
        selected_relationships: RelationshipsSelectionResult,
    ) -> Document:
        all_context_text = f"-----{self._context_name}-----" + "\n"
        header = ["id", "source", "target", "description"]
        if self._include_weight:
            header.append("weight")

        all_context_text += self._column_delimiter.join(header) + "\n"
        all_token_count = self._token_counter.count_tokens(all_context_text)

        def _build_context_text(
            relationships: pd.DataFrame,
            context_text: str,
            token_count: int,
        ) -> tuple[str, int]:
            for relationship in relationships.itertuples():
                new_context = [
                    str(relationship.human_readable_id),
                    relationship.source,
                    relationship.target,
                    relationship.description,
                ]
                if self._include_weight:
                    new_context.append(str(relationship.weight))

                new_context_text = self._column_delimiter.join(new_context) + "\n"
                new_token_count = self._token_counter.count_tokens(new_context_text)

                if token_count + new_token_count > self._max_tokens:
                    _LOGGER.warning(
                        f"Stopping relationships context build at {token_count} tokens..."  # noqa: E501
                    )
                    return context_text, token_count

                context_text += new_context_text
                token_count += new_token_count

            return context_text, token_count

        all_context_text, all_token_count = _build_context_text(
            selected_relationships.in_network_relationships,
            all_context_text,
            all_token_count,
        )

        if all_token_count < self._max_tokens:
            all_context_text, all_token_count = _build_context_text(
                selected_relationships.out_network_relationships,
                all_context_text,
                all_token_count,
            )

        return Document(
            page_content=all_context_text,
            metadata={"token_count": all_token_count},
        )
