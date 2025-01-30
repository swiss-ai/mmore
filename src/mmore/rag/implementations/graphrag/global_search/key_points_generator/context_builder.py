import logging

from langchain_core.documents import Document

from mmore.index.implementations.graphrag.artifacts import IndexerArtifacts
from mmore.rag.implementations.graphrag.global_search.community_report import CommunityReport
from mmore.rag.implementations.graphrag.global_search.community_weight_calculator import (
    CommunityWeightCalculator,
)
from mmore.types.graphrag.graphs.community import CommunityId, CommunityLevel
from mmore.utils.graphrag.token_counter import TokenCounter

_REPORT_TEMPLATE = """
--- Report {report_id} ---

Title: {title}
Weight: {weight}
Rank: {rank}
Report:

{content}

"""

_LOGGER = logging.getLogger(__name__)


class CommunityReportContextBuilder:
    def __init__(
        self,
        community_level: CommunityLevel,
        weight_calculator: CommunityWeightCalculator,
        artifacts: IndexerArtifacts,
        token_counter: TokenCounter,
        max_tokens: int = 8000,
    ):
        self._community_level = community_level
        self._weight_calculator = weight_calculator
        self._artifacts = artifacts
        self._token_counter = token_counter
        self._max_tokens = max_tokens

    def _filter_communities(self) -> list[CommunityReport]:
        df_entities = self._artifacts.entities
        df_reports = self._artifacts.communities_reports

        reports_weight: dict[CommunityId, float] = self._weight_calculator(
            df_entities,
            df_reports,
        )

        df_reports_filtered = df_reports[df_reports["level"] <= self._community_level]

        reports = []
        for _, row in df_reports_filtered.iterrows():
            reports.append(
                CommunityReport(
                    id=row["community_id"],
                    weight=reports_weight[row["community_id"]],
                    title=row["title"],
                    summary=row["summary"],
                    rank=row["rating"],
                    content=row["content"],
                )
            )

        return reports

    def __call__(self) -> list[Document]:
        reports = self._filter_communities()

        documents: list[Document] = []
        report_str_accumulated: list[str] = []
        token_count = 0
        for report in reports:
            # we would try to combine multiple
            # reports into a single document
            # as long as we do not exceed the token limit

            report_str = _REPORT_TEMPLATE.format(
                report_id=report.id,
                title=report.title,
                weight=report.weight,
                rank=report.rank,
                content=report.content,
            )

            report_str_token_count = self._token_counter.count_tokens(report_str)

            if token_count + report_str_token_count > self._max_tokens:
                _LOGGER.debug("Reached max tokens for a community report call ...")
                # we cut a new document here
                documents.append(
                    Document(
                        page_content="\n".join(report_str_accumulated),
                        metadata={"token_count": token_count},
                    )
                )
                # reset the token count and the accumulated string
                token_count = 0
                report_str_accumulated = []
            else:
                token_count += report_str_token_count
                report_str_accumulated.append(report_str)

        if report_str_accumulated:
            documents.append(
                Document(
                    page_content="\n".join(report_str_accumulated),
                    metadata={"token_count": token_count},
                )
            )

        if _LOGGER.isEnabledFor(logging.DEBUG):
            import tableprint

            rows = []
            tableprint.banner("KP Generation Context Token Usage")
            for index, doc in enumerate(documents):
                rows.append([f"Report {index}", doc.metadata["token_count"]])

            tableprint.table(rows, ["Reports", "Token Count"])

        return documents
