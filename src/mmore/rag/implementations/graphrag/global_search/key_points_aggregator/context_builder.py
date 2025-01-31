import logging

from langchain_core.documents import Document

from mmore.rag.implementations.graphrag.global_search.key_points_generator.utils import (
    KeyPointsResult,
)
from mmore.utils.graphrag.token_counter import TokenCounter

_REPORT_TEMPLATE = """
--- {analyst} ---

Importance Score: {score}

{content}

"""

_LOGGER = logging.getLogger(__name__)


class KeyPointsContextBuilder:
    def __init__(
        self,
        token_counter: TokenCounter,
        max_tokens: int = 8000,
    ):
        self._token_counter = token_counter
        self._max_tokens = max_tokens

    def __call__(self, key_points: dict[str, KeyPointsResult]) -> list[Document]:
        documents: list[Document] = []
        total_tokens = 0
        max_token_limit_reached = False
        for k, v in key_points.items():
            if max_token_limit_reached:
                break
            for p in v.points:
                report = _REPORT_TEMPLATE.format(
                    analyst=k,
                    score=p.score,
                    content=p.description,
                )
                report_token = self._token_counter.count_tokens(report)
                if total_tokens + report_token > self._max_tokens:
                    _LOGGER.debug("Reached max tokens for key points aggregation ...")
                    max_token_limit_reached = True
                    break
                total_tokens += report_token
                documents.append(
                    Document(
                        page_content=report,
                        metadata={
                            "rank": p.score,
                            "analyst": k,
                            "token_count": report_token,
                        },
                    )
                )

        # we now sort the documents based on the
        # importance score of the key points
        sorted_documents = sorted(
            documents,
            key=lambda x: x.metadata["rank"],
            reverse=True,
        )

        if _LOGGER.isEnabledFor(logging.DEBUG):
            import tableprint

            rows = []
            tableprint.banner("KP Aggregation Context Token Usage")
            for doc in sorted_documents:
                rows.append([doc.metadata["analyst"], doc.metadata["token_count"]])  # noqa: PERF401

            tableprint.table(rows, ["Analyst", "Token Count"])

        return sorted_documents
