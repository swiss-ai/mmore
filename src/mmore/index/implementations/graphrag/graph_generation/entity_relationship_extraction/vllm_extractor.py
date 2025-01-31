"""Entity Relationship Extractor module."""

from __future__ import annotations

import logging

import networkx as nx
import pandas as pd
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompt_values import StringPromptValue
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_core.outputs import LLMResult

from tqdm import tqdm

from mmore.index.implementations.graphrag.vllm_model import vLLMWrapper

from mmore.types.graphrag.prompts import IndexingPromptBuilder

from .prompt_builder import EntityExtractionPromptBuilder
from .extractor import EntityRelationshipExtractor

from typing import Any, Dict, List

_LOGGER = logging.getLogger(__name__)


class vLLMEntityRelationshipExtractor(EntityRelationshipExtractor):
    def __init__(
        self,
        prompt_builder: IndexingPromptBuilder,
        llm: vLLMWrapper,
        *,
        chain_config: RunnableConfig | None = None,
    ):
        """Extracts entities and relationships from text units using a language model.

        Args:
            prompt_builder (PromptBuilder): The prompt builder object used to construct the prompt for the language model.
            llm (LanguageModelLike): The language model used for entity and relationship extraction.
            chain_config (RunnableConfig, optional): The configuration object for the extraction chain. Defaults to None.

        """
        prompt, self._output_parser = prompt_builder.build()
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._chain_config = chain_config
        self._prompt_template = prompt

    @staticmethod
    def build_default(
        llm: vLLMWrapper,
        *,
        chain_config: RunnableConfig | None = None,
    ) -> vLLMEntityRelationshipExtractor:
        """Builds and returns an instance of EntityRelationshipExtractor with default parameters.

        Parameters:
            llm (LanguageModelLike): The language model used for entity relationship extraction.
            chain_config (RunnableConfig, optional): The configuration object for the extraction chain. Defaults to None.

        Returns:
            EntityRelationshipExtractor: An instance of EntityRelationshipExtractor with default parameters.
        """
        return vLLMEntityRelationshipExtractor(
            prompt_builder=EntityExtractionPromptBuilder(),
            llm=llm,
            chain_config=chain_config,
        )
    
    def _prepare_batch_prompts(self, text_units: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare all prompts for batch processing."""
        prompts = []
        for _, row in text_units.iterrows():
            chain_input = self._prompt_builder.prepare_chain_input(text_unit=row['text_unit'])

            formatted_prompt = self._prompt_template.format_messages(**chain_input)
            prompts.append({
                'prompt': formatted_prompt,
                'metadata': {
                    'document_id': row['document_id'],
                    'text_id': row['id']
                }
            })
        return prompts
    
    def _process_vllm_outputs(self, outputs: List[LLMResult], metadata: List[Dict]) -> List[nx.Graph]:
        """Process vLLM outputs into graphs."""
        graphs = []
        for output, meta in zip(outputs, metadata):
            try:                
                graph = self._output_parser.parse_result(output.generations[0])
                
                for node in graph.nodes():
                    graph.nodes[node]['text_unit_ids'] = [meta['text_id']]
                for edge in graph.edges():
                    graph.edges[edge]['text_unit_ids'] = [meta['text_id']]
                
                graphs.append(graph)
                
            except Exception as e:
                _LOGGER.error(f"Error processing output for text unit {meta['text_id']}: {str(e)}")
                graphs.append(nx.Graph())
                
        return graphs

    def invoke(self, text_units: pd.DataFrame) -> List[nx.Graph]:
        """
        Extract entities and relationships from text units in parallel using vLLM.
        
        Args:
            text_units: DataFrame with columns: document_id, id, text_unit
            
        Returns:
            List of networkx Graphs representing extracted entities and relationships
        """
        prompt_batch = self._prepare_batch_prompts(text_units)
        
        prompts = [item['prompt'] for item in prompt_batch]
        metadata = [item['metadata'] for item in prompt_batch]

        _LOGGER.info(f"Processing {len(prompts)} text units in parallel...")

        outputs = self._llm.generate(prompts, max_tokens=2048)
        
        graphs = self._process_vllm_outputs(outputs.flatten(), metadata)
        
        return graphs