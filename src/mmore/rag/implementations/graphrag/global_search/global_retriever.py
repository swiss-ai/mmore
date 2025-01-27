from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel

from mmore.rag.base_retriever import RetrieverConfig

from mmore.rag.implementations.graphrag.global_search.key_points_generator.utils import KeyPointsResult

from mmore.types.graphrag.graphs.community import CommunityLevel
from mmore.rag.implementations.graphrag.global_search.community_weight_calculator import CommunityWeightCalculator
from mmore.utils.graphrag.token_counter import TiktokenCounter
from mmore.utils.graphrag.artifacts import load_artifacts
from mmore.utils import load_config


from mmore.rag.implementations.graphrag.global_search.key_points_generator.prompt_builder import KeyPointsGeneratorPromptBuilder


from mmore.rag.implementations.graphrag.global_search.key_points_generator.context_builder import CommunityReportContextBuilder
from mmore.rag.implementations.graphrag.global_search.key_points_aggregator.context_builder import KeyPointsContextBuilder


_LOGGER = logging.getLogger(__name__)

@dataclass
class GraphRAGGlobalRetrieverConfig(RetrieverConfig):
    artifact_path: str | Path
    community_level: int = 3
    show_references: bool = False
    repeat_instructions: bool = False

    def __post_init__(self):
        if isinstance(self.artifact_path, str):
            self.artifact_path = Path(self.artifact_path)

def _format_docs(documents: list[Document]) -> str:
    context_data = [d.page_content for d in documents]
    context_data_str: str = "\n".join(context_data)
    return context_data_str

class GraphRAGGlobalRetriever(BaseRetriever):
    """Retriever for key points using vLLM-based language model."""
    llm : BaseChatModel
    community_report_context_builder: CommunityReportContextBuilder
    keypoints_context_builder: KeyPointsContextBuilder
    prompt_template: BasePromptTemplate
    output_parser: BaseOutputParser

    @classmethod
    def from_config(cls, config: str | GraphRAGGlobalRetrieverConfig, llm: BaseChatModel): 
        """Create a new instance of the retriever from a configuration object."""

        if isinstance(config, str):
            config = load_config(config, GraphRAGGlobalRetrieverConfig)

        artifacts = load_artifacts(config.artifact_path)
        
        prompt_builder=KeyPointsGeneratorPromptBuilder(
            show_references=config.show_references,
            repeat_instructions=config.repeat_instructions
        )

        report_context_builder = CommunityReportContextBuilder(
            community_level=CommunityLevel(config.community_level),
            weight_calculator=CommunityWeightCalculator(),
            artifacts=artifacts,
            token_counter=TiktokenCounter(),
            max_tokens=8000,
        )
        prompt_template, output_parser = prompt_builder.build()

        key_points_context_builder = KeyPointsContextBuilder(TiktokenCounter(), 8000)


        return cls(llm=llm, prompt_template=prompt_template, output_parser=output_parser, community_report_context_builder=report_context_builder, keypoints_context_builder=key_points_context_builder)


    def _prepare_prompt(
        self, 
        query: str,
        document: Document
    ) -> Dict[str, Any]:
        """Prepare prompt for a single document."""
        context_data = _format_docs([document])
        chain_input = {"global_query": query, "context_data": context_data}
        
        return self.prompt_template.format(**chain_input) 

    def _process_vllm_output(self, output: str) -> KeyPointsResult:
        """Process vLLM output into a KeyPointsResult object."""
        try:
            if "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" in output:
                output = output.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[1].strip()
            if "--- Response ---" in output:
                json_str = output.split("--- Response ---")[1].strip()
            elif "Reponse: " in output:
                json_str = output.split("Reponse: ")[1].strip()
            elif "### Response" in output:
                json_str = output.split("### Response")[1].strip()
            else:
                json_str = output.strip()

            return self.output_parser.parse(json_str)

        except Exception as e:
            _LOGGER.error(f"Error processing output: {str(e)}\nRaw output: {json_str}")
            return KeyPointsResult(points = [])

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents with key points relevant to the query.
        
        Args:
            query: The query string
            
        Returns:
            List of Document objects containing relevant key points
        """
        context_documents = self.community_report_context_builder()
        
        all_prompts = [self._prepare_prompt(query, doc) for doc in context_documents]
        
        _LOGGER.info(f"Processing {len(all_prompts)} documents...")
        
        for idx, prompt in enumerate(all_prompts):
            _LOGGER.debug(f"Prompt {idx}: {prompt}")

        outputs = [self.llm.invoke(prompt).content for prompt in all_prompts]
        
        for idx, output in enumerate(outputs):
            _LOGGER.debug(f"Output {idx}: {output}")

        result_keypoints = [self._process_vllm_output(output) for output in outputs]

        result_keypoints = dict(zip([f"Analyst-{i + 1}" for i in range(len(result_keypoints))], result_keypoints))
        
        return self.keypoints_context_builder(result_keypoints)