from __future__ import annotations

import networkx as nx
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables.config import RunnableConfig

from mmore.types.graphrag.graphs.community import Community
from mmore.types.graphrag.prompts import IndexingPromptBuilder

from .prompt_builder import CommunityReportGenerationPromptBuilder
from .utils import CommunityReportResult


class CommunityReportGenerator:
    def __init__(
        self,
        prompt_builder: IndexingPromptBuilder,
        llm: LanguageModelLike,
        *,
        chain_config: RunnableConfig | None = None,
    ):
        prompt, output_parser = prompt_builder.build()
        self._chain = prompt | llm | output_parser
        self._prompt_builder = prompt_builder
        self._chain_config = chain_config

    @staticmethod
    def build_default(
        llm: LanguageModelLike,
        *,
        chain_config: RunnableConfig | None = None,
    ) -> CommunityReportGenerator:
        return CommunityReportGenerator(
            prompt_builder=CommunityReportGenerationPromptBuilder(),
            llm=llm,
            chain_config=chain_config,
        )

    def invoke(self, community: Community, graph: nx.Graph) -> CommunityReportResult:
        chain_input = self._prompt_builder.prepare_chain_input(
            community=community,
            graph=graph,
        )

        return self._chain.invoke(input=chain_input, config=self._chain_config)
