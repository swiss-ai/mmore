from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing_extensions import Unpack

from mmore.types.graphrag.graphs.community import Community
from mmore.types.graphrag.prompts import IndexingPromptBuilder

from ._default_prompts import MILOU_PROMPT
from ._output_parser import CommunityReportOutputParser
from .utils import get_info


class CommunityReportGenerationPromptBuilder(IndexingPromptBuilder):
    def __init__(
        self,
        *,
        prompt: str | None = None,
    ):
        self._prompt: str | None
        if prompt is None:
            self._prompt = MILOU_PROMPT
        else:
            self._prompt = prompt

    def build(self) -> tuple[ChatPromptTemplate, BaseOutputParser]:
        prompt_template = ChatPromptTemplate.from_template(self._prompt)

        return prompt_template, CommunityReportOutputParser()

    def prepare_chain_input(self, **kwargs: Unpack[dict[str, Any]]) -> dict[str, str]:
        community: Community = kwargs.get("community", None)
        graph: nx.Graph = kwargs.get("graph", None)

        if community is None:
            raise ValueError("community is required")

        if graph is None:
            raise ValueError("graph is required")

        entities, relationships = get_info(community, graph)
        for entity in entities:
            for key, value in entity.items():
                entity[key] = str(value).replace("\"", "")
        
        for relationship in relationships:
            for key, value in relationship.items():
                relationship[key] = str(value).replace("\"", "")

        entities_table = pd.DataFrame.from_records(entities).to_csv(
            index=False,
        )

        relationships_table = pd.DataFrame.from_records(relationships).to_csv(
            index=False,
        )

        input_text = f"""
        -----Entities-----
        {entities_table}

        -----Relationships-----
        {relationships_table}
        """

        return dict(input_text=input_text)
