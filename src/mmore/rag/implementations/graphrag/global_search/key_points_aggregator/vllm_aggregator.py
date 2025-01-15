from __future__ import annotations

import logging
from typing import Any, Dict
import operator
from functools import partial

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from mmore.rag.implementations.graphrag.global_search.key_points_generator.utils import KeyPointsResult
from mmore.types.graphrag.prompts import PromptBuilder
from .aggregator import KeyPointsAggregator
from .context_builder import KeyPointsContextBuilder

_LOGGER = logging.getLogger(__name__)


def _format_docs(documents: list[Document]) -> str:
    context_data = [d.page_content for d in documents]
    context_data_str: str = "\n".join(context_data)
    return context_data_str

def _kp_result_to_docs(
    key_points: dict[str, KeyPointsResult],
    context_builder: KeyPointsContextBuilder,
) -> list[Document]:
    return context_builder(key_points)

class vLLMKeyPointsAggregator(KeyPointsAggregator):
    def __init__(
        self,
        llm: vLLMWrapper,
        prompt_builder: PromptBuilder,
        context_builder: KeyPointsContextBuilder,
        *,
        output_raw: bool = False,
    ):
        """Initialize vLLM-based key points aggregator.

        Args:
            llm: The vLLM-based language model
            prompt_builder: Prompt builder for constructing prompts
            context_builder: Context builder for key points
            output_raw: Whether to output raw LLM response
        """
        self._llm = llm
        self._context_builder = context_builder
        self._output_raw = output_raw
        self._prompt_template, self._output_parser = prompt_builder.build()

    def _prepare_aggregate_prompt(
        self,
        report_data: str,
        global_query: str
    ) -> Dict[str, Any]:
        """Prepare prompt for aggregation."""
        
        chain_input = {
            "report_data": report_data,
            "global_query": global_query
        }
        
        formatted_prompt = self._prompt_template.format(**chain_input)
        return {
            'prompt': formatted_prompt,
            'metadata': {
                'query': global_query
            }
        }

    def _process_vllm_output(
        self, 
        output: str
    ) -> str | KeyPointsResult:
        """Process vLLM output into final result."""
        if self._output_raw:
            return output
            
        try:
            return self._output_parser.parse(output)
        except Exception as e:
            _LOGGER.error(f"Error processing aggregation output: {str(e)}")
            return KeyPointsResult(
                points=[],
                confidence=0.0,
            )

    def __call__(self) -> Runnable:
        """Create a runnable that aggregates key points using vLLM."""
        kp_lambda = partial(
            _kp_result_to_docs,
            context_builder=self._context_builder,
        )

        def aggregate_key_points(input_data: Dict) -> str | KeyPointsResult:
            report_data = input_data["report_data"]
            global_query = input_data["global_query"]
            
            prompt_data = self._prepare_aggregate_prompt(report_data, global_query)

            _LOGGER.info(f"Aggregating key points for query: {global_query} \n Prompt : \n {prompt_data['prompt']}")
            
            output = self._llm([prompt_data['prompt']], max_tokens=512)[0]
            
            return self._process_vllm_output(output)
            
        search_chain: Runnable = {
            "report_data": operator.itemgetter("report_data")
            | RunnableLambda(kp_lambda)
            | _format_docs,
            "global_query": operator.itemgetter("global_query"),
        } | RunnableLambda(aggregate_key_points)

        return search_chain