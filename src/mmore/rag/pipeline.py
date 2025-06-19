"""
Example implementation:
RAG pipeline.
Integrates Milvus retrieval with HuggingFace text generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from ..utils import load_config
from .llm import LLM, LLMConfig
from .retriever import Retriever, RetrieverConfig
from .types import MMOREInput, MMOREOutput

DEFAULT_PROMPT = """\
Use the following context to answer the questions. If none of the context answer the question, just say you don't know.

Context:
{context}
"""


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    retriever: RetrieverConfig
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(llm_name="gpt2"))
    system_prompt: str = DEFAULT_PROMPT


class RAGPipeline:
    """Main RAG pipeline combining retrieval and generation."""

    retriever: Retriever
    llm: BaseChatModel
    prompt_template: Union[str, ChatPromptTemplate]

    def __init__(
        self,
        retriever: Retriever,
        prompt_template: Union[str, ChatPromptTemplate],
        llm: BaseChatModel,
    ):
        # Get modules
        self.retriever = retriever
        self.prompt = prompt_template
        self.llm = llm

        # Build the rag chain
        self.rag_chain = RAGPipeline._build_chain(
            self.retriever, RAGPipeline.format_docs, self.prompt, self.llm
        )

    def __str__(self):
        return str(self.rag_chain)

    @classmethod
    def from_config(cls, config: str | RAGConfig):
        if isinstance(config, str):
            config = load_config(config, RAGConfig)

        retriever = Retriever.from_config(config.retriever)
        llm = LLM.from_config(config.llm)
        chat_template = ChatPromptTemplate.from_messages(
            [("system", config.system_prompt), ("human", "{input}")]
        )

        return cls(retriever, chat_template, llm)

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Format documents for prompt."""
        return "\n\n".join(
            f"[{doc.metadata['rank']}] {doc.page_content}" for doc in docs
        )
        # return "\n\n".join(f"[#{doc.metadata['rank']}, sim={doc.metadata['similarity']:.2f}] {doc.page_content}" for doc in docs)

    @staticmethod
    def _build_chain(retriever, format_docs, prompt, llm) -> Runnable:
        validate_input = RunnableLambda(
            lambda x: MMOREInput.model_validate(x).model_dump()
        )

        def make_output(x):
            """Validate the output of the LLM and keep only the actual answer of the assistant"""
            res_dict = MMOREOutput.model_validate(x).model_dump()
            res_dict["answer"] = res_dict["answer"].split("<|assistant|>\n")[-1]

            return res_dict

        validate_output = RunnableLambda(make_output)

        rag_chain_from_docs = prompt | llm | StrOutputParser()

        core_chain = (
            RunnablePassthrough.assign(docs=retriever)
            .assign(context=lambda x: format_docs(x["docs"]))
            .assign(answer=rag_chain_from_docs)
        )

        return validate_input | core_chain | validate_output

    def __call__(
        self, queries: Dict[str, Any] | List[Dict[str, Any]], return_dict: bool = False
    ) -> List[Dict[str, str | List[str]]]:
        if isinstance(queries, Dict):
            queries_list = [queries]
        else:
            queries_list = queries

        results = self.rag_chain.batch(queries_list)

        if return_dict:
            return results
        else:
            return [result["answer"] for result in results]
