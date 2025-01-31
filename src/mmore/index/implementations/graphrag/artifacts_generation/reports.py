import logging

import networkx as nx
import pandas as pd
from langchain_core.exceptions import OutputParserException
from tqdm import tqdm

from mmore.index.implementations.graphrag.report_generation import (
    CommunityReportGenerator,
    CommunityReportWriter,
)
from mmore.types.graphrag.graphs.community import (
    Community,
    CommunityDetectionResult,
)

_LOGGER = logging.getLogger(__name__)


def _get_entities(community: Community, graph: nx.Graph) -> list[str]:
    return [graph.nodes[n.name]["id"] for n in community.nodes]


class CommunitiesReportsArtifactsGenerator:
    def __init__(
        self,
        report_generator: CommunityReportGenerator,
        report_writer: CommunityReportWriter,
    ):
        self._report_generator = report_generator
        self._report_writer = report_writer

    def run(
        self,
        detection_result: CommunityDetectionResult,
        graph: nx.Graph,
    ) -> pd.DataFrame:
        reports = []

        # TODO: Parallelize all this
        for level in detection_result.communities:
            communities = detection_result.communities_at_level(level)
            c_pbar = tqdm(communities)
            for c in c_pbar:
                c_pbar.set_description_str(
                    f"Generating report for level={level} commnuity_id={c.id}"
                )

                try:
                    report = self._report_generator.invoke(community=c, graph=graph)
                except OutputParserException:
                    _LOGGER.exception(
                        f"Failed to generate report for level={level} community_id={c.id}"
                    )
                    continue

                report_str = self._report_writer.write(report)
                entities = _get_entities(c, graph)

                reports.append(
                    dict(
                        level=level,
                        community_id=c.id,
                        entities=entities,
                        title=report.title,
                        summary=report.summary,
                        rating=report.rating,
                        rating_explanation=report.rating_explanation,
                        content=report_str,
                    )
                )

        return pd.DataFrame.from_records(reports)
