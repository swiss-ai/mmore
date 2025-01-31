import logging

import pandas as pd
from langchain_core.documents import Document

from mmore.types.graphrag.tokens import TokenCounter

_LOGGER = logging.getLogger(__name__)


class EntitiesContextBuilder:
    def __init__(
        self,
        *,
        include_rank: bool = True,
        context_name: str = "Entities",
        rank_heading: str = "number of relationships",
        column_delimiter: str = "|",
        max_tokens: int = 8000,
        token_counter: TokenCounter,
    ):
        self._include_rank = include_rank
        self._context_name = context_name
        self._rank_heading = rank_heading
        self._column_delimiter = column_delimiter
        self._max_tokens = max_tokens
        self._token_counter = token_counter

    def __call__(self, entities: pd.DataFrame) -> Document:
        context_text = f"-----{self._context_name}-----" + "\n"
        header = ["id", "entity", "description"]
        if self._include_rank:
            header.append(self._rank_heading)

        context_text += self._column_delimiter.join(header) + "\n"
        token_count = self._token_counter.count_tokens(context_text)

        for entity in entities.itertuples():
            new_context = [
                str(entity.human_readable_id),
                entity.title,
                entity.description,
            ]
            if self._include_rank:
                new_context.append(str(entity.degree))

            new_context_text = self._column_delimiter.join(new_context) + "\n"

            new_token_count = self._token_counter.count_tokens(new_context_text)
            if token_count + new_token_count > self._max_tokens:
                _LOGGER.warning(
                    f"Stopping entities context build at {token_count} tokens ..."
                )
                break

            context_text += new_context_text
            token_count += new_token_count

        return Document(
            page_content=context_text,
            metadata={"token_count": token_count},
        )
