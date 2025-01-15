"""vLLM-based Community Report Generator module."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import networkx as nx
from langchain_core.runnables.config import RunnableConfig

from mmore.types.graphrag.graphs.community import Community
from mmore.types.graphrag.prompts import IndexingPromptBuilder
from .generator import CommunityReportGenerator
from .prompt_builder import CommunityReportGenerationPromptBuilder
from .utils import CommunityReportResult

_LOGGER = logging.getLogger(__name__)

class vLLMCommunityReportGenerator(CommunityReportGenerator):
    def __init__(
        self,
        prompt_builder: IndexingPromptBuilder,
        llm: vLLMWrapper,
        *,
        chain_config: RunnableConfig | None = None,
    ):
        """Initialize vLLM-based community report generator.

        Args:
            prompt_builder: The prompt builder for constructing report generation prompts
            llm: The vLLM-based language model
            chain_config: Optional configuration for the chain
        """
        self._prompt_template, self._output_parser = prompt_builder.build()
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._chain_config = chain_config

    @staticmethod
    def build_default(
        llm: vLLMWrapper,
        *,
        chain_config: RunnableConfig | None = None,
    ) -> vLLMCommunityReportGenerator:
        return vLLMCommunityReportGenerator(
            prompt_builder=CommunityReportGenerationPromptBuilder(),
            llm=llm,
            chain_config=chain_config,
        )

    def _prepare_community_prompts(
        self, 
        communities: List[Community], 
        graph: nx.Graph
    ) -> List[str]:
        """Prepare prompts for community report generation."""
        prompts = []
        for community in communities:
            chain_input = self._prompt_builder.prepare_chain_input(
                community=community,
                graph=graph,
            )
            
            formatted_prompt = self._prompt_template.format(**chain_input)
            prompts.append(formatted_prompt)
        return prompts

    def _process_vllm_outputs(
        self, 
        outputs: List[str]
        ) -> List[CommunityReportResult]:
        """Process vLLM outputs into community reports."""
        reports = []
        for output in outputs:

            report = self._output_parser.parse(output)
            reports.append(report)
                
        return reports

    def invoke(
        self, 
        community: Community | List[Community], 
        graph: nx.Graph
    ) -> CommunityReportResult | List[CommunityReportResult]:
        """
        Generate reports for communities in parallel using vLLM.
        
        Args:
            community: Single community or list of communities to generate reports for
            graph: The graph containing the community data
            
        Returns:
            Single report or list of reports depending on input
        """
        
        communities = [community] if not isinstance(community, list) else community
        
        prompts = self._prepare_community_prompts(communities, graph)
        
        _LOGGER.info(f"Processing {len(prompts)} communities in parallel...")


        
        outputs = self._llm(prompts, max_tokens=2048)

        for i in range(len(outputs)):
            outputs[i] = "{" + outputs[i].split("{", 1)[1]
            if "```" in outputs[i]:
                outputs[i] = outputs[i].split("```")[0]
            
            outputs[i] = outputs[i].replace(", {}", "")


        reports = self._process_vllm_outputs(outputs)
        
        return reports[0] if not isinstance(community, list) else reports