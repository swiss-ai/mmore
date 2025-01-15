from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from mmore.rag.implementations.graphrag.global_search.key_points_generator.utils import (
    KeyPointsResult,
    KeyPointInfo,
)
from mmore.types.graphrag.prompts import PromptBuilder
from .generator import KeyPointsGenerator
from .context_builder import CommunityReportContextBuilder

_LOGGER = logging.getLogger(__name__)

def _format_docs(documents: list[Document]) -> str:
    context_data = [d.page_content for d in documents]
    context_data_str: str = "\n".join(context_data)
    return context_data_str

class vLLMKeyPointsGenerator(KeyPointsGenerator):
    def __init__(
        self,
        llm: vLLMWrapper,
        prompt_builder: PromptBuilder,
        context_builder: CommunityReportContextBuilder,
    ):
        """Initialize vLLM-based key points generator.

        Args:
            llm: The vLLM-based language model
            prompt_builder: Prompt builder for constructing prompts
            context_builder: Context builder for community reports
        """
        self._llm = llm
        self._context_builder = context_builder
        self._prompt_template, self._output_parser = prompt_builder.build()

    def _prepare_batch_prompts( 
        self, 
        query: str,
        documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """Prepare prompts for batch processing."""
        prompts = []
        for idx, doc in enumerate(documents):
            context_data = _format_docs([doc])
            chain_input = {"global_query": query, "context_data": context_data}
            
            formatted_prompt = self._prompt_template.format(**chain_input)
            prompts.append({
                'prompt': formatted_prompt,
                'metadata': {
                    'analyst_idx': idx,
                }
            })
        return prompts

    def _process_vllm_outputs(
    self, 
    outputs: List[str], 
    metadata: List[Dict]
    ) -> Dict[str, KeyPointsResult]:
        """Process vLLM outputs into key points results."""
        results = {}
        for output, meta in zip(outputs, metadata):
            try:
                analyst_name = f"Analyst-{meta['analyst_idx'] + 1}"
                if "--- Response ---" in output:
                    json_str = output.split("--- Response ---")[1].strip()
                elif "Reponse: " in output:
                    json_str = output.split("Reponse: ")[1].strip()
                elif "### Response" in output:
                    json_str = output.split("### Response")[1].strip()
                else:
                    # If no separator, try to use the whole output
                    json_str = output.strip()

                parsed_json = self._output_parser.parse(json_str)
            
                results[analyst_name] = parsed_json

            except Exception as e:
                _LOGGER.error(
                    f"Error processing output for analyst {meta['analyst_idx']}: {str(e)}\n"
                    f"Raw output: {output}"
                )
                results[f"Analyst-{meta['analyst_idx'] + 1}"] = KeyPointsResult(points=[])
                raise e
            
        return results

    def __call__(self) -> Runnable:
        """Create a runnable that generates key points using vLLM."""
        def generate_key_points(query: str) -> Dict[str, KeyPointsResult]:
            documents = self._context_builder()
            
            all_prompts = self._prepare_batch_prompts(query, documents)

            
            _LOGGER.info(f"Processing {len(all_prompts)} documents in parallel...")
            
            prompts = [item['prompt'] for item in all_prompts]
            metadata = [item['metadata'] for item in all_prompts]

            for idx, prompt in enumerate(prompts):
                _LOGGER.info(f"Prompt {idx}: {prompt}")
            
            outputs = self._llm(prompts, max_tokens=512)
            
            for idx, output in enumerate(outputs):
                _LOGGER.info(f"Output {idx}: {output}")

            return self._process_vllm_outputs(outputs, metadata)
            
        return RunnableLambda(generate_key_points)

