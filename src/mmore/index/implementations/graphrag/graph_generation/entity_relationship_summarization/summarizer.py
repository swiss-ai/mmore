from __future__ import annotations

import networkx as nx
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables.config import RunnableConfig
from tqdm import tqdm

from mmore.types.graphrag.prompts import IndexingPromptBuilder

from .prompt_builder import SummarizeDescriptionPromptBuilder


class EntityRelationshipDescriptionSummarizer:
    def __init__(
        self,
        prompt_builder: IndexingPromptBuilder,
        llm: LanguageModelLike,
        *,
        chain_config: RunnableConfig | None = None,
    ):
        prompt, output_parser = prompt_builder.build()
        self._summarize_chain = prompt | llm | output_parser
        self._prompt_builder = prompt_builder
        self._chain_config = chain_config

    @staticmethod
    def build_default(
        llm: LanguageModelLike,
        *,
        chain_config: RunnableConfig | None = None,
    ) -> EntityRelationshipDescriptionSummarizer:
        return EntityRelationshipDescriptionSummarizer(
            prompt_builder=SummarizeDescriptionPromptBuilder(),
            llm=llm,
            chain_config=chain_config,
        )

    def invoke(self, graph: nx.Graph) -> nx.Graph:
        for node_name, node in tqdm(
            graph.nodes(data=True), desc="Summarizing entities descriptions"
        ):
            if len(node["description"]) == 1:
                node["description"] = node["description"][0]
                continue

            chain_input = self._prompt_builder.prepare_chain_input(
                entity_name=node_name, description_list=node["description"]
            )

            node["description"] = self._summarize_chain.invoke(
                input=chain_input,
                config=self._chain_config,
            )

        for from_node, to_node, edge in tqdm(
            graph.edges(data=True), desc="Summarizing relationship descriptions"
        ):
            if len(edge["description"]) == 1:
                edge["description"] = edge["description"][0]
                continue

            chain_input = self._prompt_builder.prepare_chain_input(
                entity_name=f"{from_node} -> {to_node}",
                description_list=edge["description"],
            )

            edge["description"] = self._summarize_chain.invoke(
                input=chain_input,
                config=self._chain_config,
            )

        return graph
