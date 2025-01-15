"""vLLM-based Local Search module optimized for single query processing."""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from mmore.types.graphrag.prompts import PromptBuilder
from .search import LocalSearch

_LOGGER = logging.getLogger(__name__)

def _format_docs(documents: list[Document]) -> str:
    context_data = [d.page_content for d in documents]
    context_data_str: str = "\n".join(context_data)
    return context_data_str


class vLLMLocalSearch(LocalSearch):
    def __init__(
        self,
        llm: LanguageModelLike,
        prompt_builder: PromptBuilder,
        retriever: BaseRetriever,
        *,
        output_raw: bool = False,
    ):
        """Initialize vLLM-based local search for single query processing.

        Args:
            llm: The vLLM-based language model
            prompt_builder: Prompt builder for constructing prompts
            retriever: Document retriever component
            output_raw: Whether to output raw LLM response
        """
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._retriever = retriever
        self._output_raw = output_raw
        prompt, self._output_parser = prompt_builder.build()
        self._prompt_template = prompt

    def _prepare_search_prompt(
        self,
        query: str,
        context: str
    ) -> str:
        """Prepare prompt for search processing."""
        chain_input = {
            "local_query": query,
            "context_data": context
        }
        return self._prompt_template.format(**chain_input)

    def _process_vllm_output(
        self, 
        output: str
    ) -> Any:
        """Process vLLM output into final result."""
        if self._output_raw:
            return output
            
        try:
            return self._output_parser.parse(output)
        except Exception as e:
            _LOGGER.error(f"Error processing output: {str(e)}")
            return None

    def __call__(self) -> Runnable:
        """Create a runnable that performs local search using vLLM."""
        def search_documents(query: str) -> Any:
            
            docs = self._retriever.get_relevant_documents(query)
            context = _format_docs(docs)
            
            formatted_prompt = self._prepare_search_prompt(query, context)
            
            output = self._llm([formatted_prompt])[0]
            
            return self._process_vllm_output(output)
            
        search_chain: Runnable = RunnableLambda(search_documents)
        return search_chain