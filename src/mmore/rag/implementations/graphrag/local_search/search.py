from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

from mmore.types.graphrag.prompts import PromptBuilder


def _format_docs(documents: list[Document]) -> str:
    context_data = [d.page_content for d in documents]
    context_data_str: str = "\n".join(context_data)
    return context_data_str


class LocalSearch:
    def __init__(
        self,
        llm: LanguageModelLike,
        prompt_builder: PromptBuilder,
        retriever: BaseRetriever,
        *,
        output_raw: bool = False,
    ):
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._retriever = retriever
        self._output_raw = output_raw

    def __call__(self) -> Runnable:
        prompt, output_parser = self._prompt_builder.build()

        base_chain = prompt | self._llm

        if not self._output_raw:
            base_chain = base_chain | output_parser

        search_chain: Runnable = {
            "context_data": self._retriever | _format_docs,
            "local_query": RunnablePassthrough(),
        } | base_chain

        return search_chain
