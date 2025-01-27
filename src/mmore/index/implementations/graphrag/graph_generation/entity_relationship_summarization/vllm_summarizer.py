from __future__ import annotations

import logging
import networkx as nx
from langchain_core.runnables.config import RunnableConfig

from typing import Any, List, Dict

from mmore.index.implementations.graphrag.vllm_model import vLLMWrapper

from mmore.types.graphrag.prompts import IndexingPromptBuilder

from .prompt_builder import SummarizeDescriptionPromptBuilder

_LOGGER = logging.getLogger(__name__)


class vLLMEntityRelationshipDescriptionSummarizer:
    def __init__(
        self,
        prompt_builder: IndexingPromptBuilder,
        llm: vLLMWrapper,
        *,
        chain_config: RunnableConfig | None = None,
    ):
        """Summarizes entity and relationship descriptions using a language model."""
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
    ) -> vLLMEntityRelationshipDescriptionSummarizer:
        return vLLMEntityRelationshipDescriptionSummarizer(
            prompt_builder=SummarizeDescriptionPromptBuilder(),
            llm=llm,
            chain_config=chain_config,
        )

    def _prepare_node_prompts(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Prepare prompts for node description summarization."""
        prompts = []
        for node_name, node in graph.nodes(data=True):
            if len(node["description"]) == 1:
                node["description"] = node["description"][0]
                continue
            
            chain_input = self._prompt_builder.prepare_chain_input(
                entity_name=node_name, 
                description_list=node["description"]
            )
            
            formatted_prompt = self._prompt_template.format(**chain_input)
            prompts.append({
                'prompt': formatted_prompt,
                'metadata': {
                    'type': 'node',
                    'name': node_name
                }
            })
        return prompts
    
    def _prepare_edge_prompts(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Prepare prompts for edge description summarization."""
        prompts = []
        for from_node, to_node, edge in graph.edges(data=True):
            if len(edge["description"]) == 1:
                edge["description"] = edge["description"][0]
                continue
                
            chain_input = self._prompt_builder.prepare_chain_input(
                entity_name=f"{from_node} -> {to_node}",
                description_list=edge["description"]
            )
            
            formatted_prompt = self._prompt_template.format(**chain_input)
            prompts.append({
                'prompt': formatted_prompt,
                'metadata': {
                    'type': 'edge',
                    'from_node': from_node,
                    'to_node': to_node
                }
            })
        return prompts
    
    def _process_vllm_outputs(
        self, 
        graph: nx.Graph, 
        outputs: List[str], 
        metadata: List[Dict]
    ) -> nx.Graph:
        """Process vLLM outputs and update the graph."""
        for output, meta in zip(outputs, metadata):
            summary = self._output_parser.parse(output)

            if "\n" in summary:
                summary = summary.split("\n")[0]
                
            if meta['type'] == 'node':
                graph.nodes[meta['name']]['description'] = summary
            else:  # edge
                graph.edges[meta['from_node'], meta['to_node']]['description'] = summary
                    
                
                
        return graph

    def invoke(self, graph: nx.Graph) -> nx.Graph:
        """
        Summarize entity and relationship descriptions in parallel using vLLM.
        
        Args:
            graph: Input graph with multiple descriptions per node/edge
            
        Returns:
            Graph with summarized descriptions
        """

        node_prompts = self._prepare_node_prompts(graph)
        edge_prompts = self._prepare_edge_prompts(graph)
        
        all_prompts = node_prompts + edge_prompts

        for prompt in all_prompts:
            _LOGGER.info(f"Prompt: {prompt['prompt']})")

        if not all_prompts:
            return graph
            
        _LOGGER.info(f"Processing {len(all_prompts)} descriptions in parallel...")
        
        # Extract prompts and metadata
        prompts = [item['prompt'] for item in all_prompts]
        metadata = [item['metadata'] for item in all_prompts]
        
        # Get vLLM outputs
        outputs = self._llm(prompts, max_tokens=512)
        
        # Process outputs and update graph
        graph = self._process_vllm_outputs(graph, outputs, metadata)
        
        return graph
