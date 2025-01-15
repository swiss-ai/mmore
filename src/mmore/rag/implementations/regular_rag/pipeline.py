"""
Example implementation: 
RAG pipeline.
Integrates Milvus retrieval with HuggingFace text generation.
"""

from typing import Union, List, Dict, Optional, Any

from dataclasses import dataclass, field

from src.mmore.rag import BaseRAGPipeline, BaseRAGConfig

from langchain.chains.base import Chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain_core.language_models.chat_models import BaseChatModel

from src.mmore.rag.retriever import Retriever, RetrieverConfig
from src.mmore.rag.llm import LLM, LLMConfig
from src.mmore.rag.types import QuotedAnswer, CitedAnswer

from src.mmore.utils import load_config

DEFAULT_PROMPT = """\
Use the following context to answer the questions. If none of the context answer the question, just say you don't know.

Context:
{context}
"""


@dataclass
class RAGConfig(BaseRAGConfig):
    """Configuration for RAG pipeline."""
    retriever: RetrieverConfig
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(llm_name='gpt2'))
    system_prompt: str = DEFAULT_PROMPT


class RAGPipeline(BaseRAGPipeline):
    """Main RAG pipeline combining retrieval and generation."""

    retriever: Retriever
    llm: BaseChatModel
    prompt_template: str

    def __init__(
            self,
            retriever: Retriever,
            prompt_template: str,
            llm: BaseChatModel,
    ):
        # Get modules
        self.retriever = retriever
        self.prompt = prompt_template
        self.llm = llm

        # Build the rag chain
        self.rag_chain = RAGPipeline._build_chain(self.retriever, RAGPipeline.format_docs, self.prompt, self.llm)

    def __str__(self):
        return str(self.rag_chain)

    @classmethod
    def from_config(cls, config: str | RAGConfig):
        if isinstance(config, str):
            config = load_config(config, RAGConfig)

        retriever = Retriever.from_config(config.retriever)
        llm = LLM.from_config(config.llm)
        chat_template = ChatPromptTemplate.from_messages(
            [
                ("system", config.system_prompt),
                ("human", "{input}")
            ]
        )

        return cls(retriever, chat_template, llm)

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Format documents for prompt."""
        return "\n\n".join(f"[{doc.metadata['rank']}] {doc.page_content}" for doc in docs)
        # return "\n\n".join(f"[#{doc.metadata['rank']}, sim={doc.metadata['similarity']:.2f}] {doc.page_content}" for doc in docs)

    @staticmethod
    # TODO: Add non RAG Pipeline (i.e. retriever is None)
    def _build_chain(retriever, format_docs, prompt, llm) -> Chain:
        structured_llm = llm
        # structured_llm = llm.with_structured_output(CitedAnswer)
        # structured_llm = llm.with_structured_output(QuotedAnswer)

        rag_chain_from_docs = (
                prompt
                | structured_llm
                | StrOutputParser()
        )

        return (
            RunnablePassthrough
            .assign(docs=retriever)
            .assign(context=lambda x: format_docs(x["docs"]))
            .assign(answer=rag_chain_from_docs)
        )

    # TODO: Define query input/output formats clearly and pass them here (or in build chain idk)
    # TODO: Streaming (low priority)
    def __call__(self, queries: Dict[str, Any] | List[Dict[str, Any]], return_dict: bool = False) -> List[
        Dict[str, str | List[str]]]:
        if isinstance(queries, Dict):
            queries = [queries]

        results = self.rag_chain.batch(queries)

        if return_dict:
            return results
        else:
            return [result['answer'] for result in results]
