"""
Example implementation:
RAG pipeline.
Integrates Milvus retrieval with text-only and vision-enabled generation backends.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, cast

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)

from ..utils import load_config
from .judge import JUDGE_OUTPUT_KEYS, JudgeConfig, LLMJudge, retrieve_with_judge
from .judge.llm import judge_llm_from_config
from .llm import LLM, LLMConfig
from .model.vision import (
    BaseMultimodalLLM,
    aggregate_image_paths,
    get_multimodal_llm,
    load_images_from_paths,
)
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
    max_images_per_request: int = 20
    judge: Optional[JudgeConfig] = None


class RAGPipeline:
    """Main RAG pipeline combining retrieval and generation."""

    retriever: Retriever
    llm: Optional[BaseChatModel]
    prompt_template: Union[str, ChatPromptTemplate]

    @staticmethod
    def _validate_generation_backend(
        use_vision: bool,
        llm: Optional[BaseChatModel],
        multimodal_llm: Optional[BaseMultimodalLLM],
    ) -> None:
        if use_vision:
            if multimodal_llm is None:
                raise ValueError(
                    "Vision mode requires a multimodal LLM. Set rag.llm.use_vision: true "
                    "and build the pipeline with RAGPipeline.from_config(), or pass "
                    "multimodal_llm=get_multimodal_llm(llm_config). For local models use "
                    "provider: HF and a Qwen-VL llm_name; install transformers, torch, and "
                    "qwen-vl-utils."
                )
            return
        if llm is None:
            raise ValueError(
                "Text-only RAG requires an LLM. Set rag.llm.use_vision: false and provide "
                "llm_name (or pass llm=LLM.from_config(...))."
            )

    def __init__(
        self,
        retriever: Retriever,
        prompt_template: Union[str, ChatPromptTemplate],
        llm: Optional[BaseChatModel],
        use_vision: bool = False,
        multimodal_llm: Optional[BaseMultimodalLLM] = None,
        max_images_per_request: int = 20,
        judge: Optional[LLMJudge] = None,
    ):
        self._validate_generation_backend(use_vision, llm, multimodal_llm)
        # Get modules
        self.retriever = retriever
        self.prompt = prompt_template
        self.llm = llm
        self.use_vision = use_vision
        self.multimodal_llm = multimodal_llm
        self.max_images_per_request = max_images_per_request
        self.judge = judge

        # Build the rag chain
        self.rag_chain = RAGPipeline._build_chain(
            self.retriever,
            RAGPipeline.format_docs,
            self.prompt,
            self.llm,
            use_vision=self.use_vision,
            multimodal_llm=self.multimodal_llm,
            max_images_per_request=self.max_images_per_request,
            judge=self.judge,
        )

    def __str__(self):
        return str(self.rag_chain)

    @classmethod
    def from_config(cls, config: str | RAGConfig):
        if isinstance(config, str):
            config = load_config(config, RAGConfig)

        retriever = Retriever.from_config(config.retriever)
        if config.llm.use_vision:
            llm: Optional[BaseChatModel] = None
            multimodal_llm = get_multimodal_llm(config.llm)
            multimodal_llm._load()
        else:
            llm = LLM.from_config(config.llm)
            multimodal_llm = None
        judge = (
            LLMJudge(llm=judge_llm_from_config(config.judge.llm), config=config.judge)
            if config.judge
            else None
        )
        chat_template = ChatPromptTemplate.from_messages(
            [("system", config.system_prompt), ("human", "{input}")]
        )

        return cls(
            retriever,
            chat_template,
            llm,
            use_vision=config.llm.use_vision,
            multimodal_llm=multimodal_llm,
            max_images_per_request=config.max_images_per_request,
            judge=judge,
        )

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Format documents for prompt."""
        return "\n\n".join(
            f"[{doc.metadata['rank']}] {doc.page_content}" for doc in docs
        )

    @staticmethod
    def _build_chain(
        retriever,
        format_docs,
        prompt,
        llm,
        use_vision=False,
        multimodal_llm=None,
        max_images_per_request=20,
        judge=None,
    ) -> Runnable:
        validate_input = RunnableLambda(
            lambda x: MMOREInput.model_validate(x).model_dump()
        )

        def make_output(x):
            """Validate the output of the LLM and keep only the actual answer of the assistant"""
            if use_vision and multimodal_llm is not None:
                image_paths = aggregate_image_paths(x["docs"])[:max_images_per_request]
            else:
                image_paths = []
            out = {
                "input": x["input"],
                "docs": x["docs"],
                "answer": x["answer"],
                "image_paths": image_paths,
            }
            res_dict = MMOREOutput.model_validate(out).model_dump()
            res_dict["answer"] = res_dict["answer"].split("<|im_start|>assistant\n")[-1]
            # Expose formatted context and judge correction logs in the API response (context is not on MMOREOutput).
            for key in ("context", *JUDGE_OUTPUT_KEYS):
                if key in x:
                    res_dict[key] = x[key]

            return res_dict

        validate_output = RunnableLambda(make_output)

        if use_vision and multimodal_llm is not None:

            def answer_with_vision(x: Dict[str, Any]) -> str:
                # Aggregate and load images linked to retrieved chunks.
                image_paths = aggregate_image_paths(x["docs"])
                images = load_images_from_paths(
                    image_paths, max_images=max_images_per_request
                )
                # Preserve chat roles instead of flattening the prompt into a
                # single text blob, so vision mode stays aligned with text-only mode.
                rendered_messages = prompt.invoke(
                    {"context": x["context"], "input": x["input"]}
                ).to_messages()

                system_parts: List[str] = []
                user_parts: List[str] = []
                for message in rendered_messages:
                    content = (
                        message.text()
                        if hasattr(message, "text")
                        else str(message.content)
                    )
                    if getattr(message, "type", None) == "system":
                        system_parts.append(content)
                    else:
                        user_parts.append(content)

                system_prompt = "\n\n".join(part for part in system_parts if part)
                prompt_text = "\n\n".join(part for part in user_parts if part)

                return multimodal_llm.invoke_with_images(
                    text=prompt_text,
                    images=images,
                    system_prompt=system_prompt or None,
                )

            rag_chain_from_docs: Runnable = RunnableLambda(answer_with_vision)
        else:
            rag_chain_from_docs = prompt | llm | StrOutputParser()

        # Only retrieval differs (retriever vs judge); format context and generate answer unchanged.
        if judge is not None:
            # retrieve with judge
            def retrieval_with_judge(state: Dict[str, Any]) -> Dict[str, Any]:
                return retrieve_with_judge(retriever, judge, state)

            retrieval_step: Runnable = RunnableLambda(retrieval_with_judge)
        else:
            # retrieve without judge
            retrieval_step = RunnablePassthrough.assign(docs=retriever)

        core_chain = retrieval_step.assign(
            context=lambda x: format_docs(x["docs"])
        ).assign(answer=rag_chain_from_docs)

        return validate_input | core_chain | validate_output

    def __call__(
        self, queries: Dict[str, Any] | List[Dict[str, Any]], return_dict: bool = False
    ) -> List[Dict[str, str | List[str]]]:
        if isinstance(queries, Dict):
            queries_list = [queries]
        else:
            queries_list = queries

        batch_config = (
            {"max_concurrency": 1} if self.use_vision and self.multimodal_llm else None
        )
        if batch_config:
            results = self.rag_chain.batch(
                queries_list, config=cast(RunnableConfig, batch_config)
            )
        else:
            results = self.rag_chain.batch(queries_list)

        if return_dict:
            return results
        else:
            return [result["answer"] for result in results]
