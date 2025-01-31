import logging

import pandas as pd
from langchain_core.documents import Document

from mmore.types.graphrag.tokens import TokenCounter

_LOGGER = logging.getLogger(__name__)


class CommunitiesReportsContextBuilder:
    def __init__(
        self,
        *,
        context_name: str = "Reports",
        column_delimiter: str = "|",
        max_tokens: int = 8000,
        token_counter: TokenCounter,
    ):
        self._context_name = context_name
        self._column_delimiter = column_delimiter
        self._max_tokens = max_tokens
        self._token_counter = token_counter

    def __call__(self, communities_reports: pd.DataFrame) -> Document:
        context_text = f"-----{self._context_name}-----" + "\n"
        header = ["id", "title", "content"]

        context_text += self._column_delimiter.join(header) + "\n"
        token_count = self._token_counter.count_tokens(context_text)

        for report in communities_reports.itertuples():
            new_context = [
                str(report.community_id),
                report.title,
                report.content,
            ]

            new_context_text = self._column_delimiter.join(new_context) + "\n"
            new_token_count = self._token_counter.count_tokens(new_context_text)

            if token_count + new_token_count > self._max_tokens:
                _LOGGER.warning(
                    f"Stopping communities context build at {token_count} tokens ..."
                )
                break

            context_text += new_context_text
            token_count += new_token_count

        return Document(
            page_content=context_text,
            metadata={"token_count": token_count},
        )
