import logging

import networkx as nx
import pandas as pd
from langchain_core.exceptions import OutputParserException
from tqdm import tqdm

from mmore.index.implementations.graphrag.report_generation import (
    CommunityReportWriter,
)
from mmore.index.implementations.graphrag.report_generation.vllm_generator import vLLMCommunityReportGenerator 
from mmore.types.graphrag.graphs.community import (
    Community,
    CommunityDetectionResult,
)

from typing import Any, Dict, List

_LOGGER = logging.getLogger(__name__)


class vLLMCommunitiesReportsArtifactsGenerator:
    def __init__(
        self,
        report_generator: vLLMCommunityReportGenerator,
        report_writer: CommunityReportWriter,
    ):
        self._report_generator = report_generator
        self._report_writer = report_writer

    def _process_communities(
        self,
        communities: List[Community],
        graph: nx.Graph,
        level: int,
    ) -> List[Dict[str, Any]]:
        """Process all communities at a given level using vLLM."""
        try:
            reports = self._report_generator.invoke(communities, graph)
            if not isinstance(reports, list):
                reports = [reports]

            results = []
            for community, report in zip(communities, reports):
                try:
                    report_str = self._report_writer.write(report)
                    entities = [graph.nodes[n.name]["id"] for n in community.nodes]

                    results.append({
                        'level': level,
                        'community_id': community.id,
                        'entities': entities,
                        'title': report.title,
                        'summary': report.summary,
                        'rating': report.rating,
                        'rating_explanation': report.rating_explanation,
                        'content': report_str,
                    })
                except Exception as e:
                    _LOGGER.exception(
                        f"Failed to process report for level={level} community_id={community.id}: {str(e)}"
                    )
                    
        except Exception as e:
            _LOGGER.exception(f"Processing failed for level {level}: {str(e)}")
            return []

        return results

    def run(
        self,
        detection_result: CommunityDetectionResult,
        graph: nx.Graph,
    ) -> pd.DataFrame:
        reports = []

        with tqdm(total=len(detection_result.communities)) as pbar:
            for level in detection_result.communities:
                communities = detection_result.communities_at_level(level)
                pbar.set_description(f"Processing level {level}")
                
                # Process all communities at this level
                level_results = self._process_communities(
                    communities=communities,
                    graph=graph,
                    level=level
                )
                
                reports.extend(level_results)
                pbar.update(1)

        return pd.DataFrame.from_records(reports)
