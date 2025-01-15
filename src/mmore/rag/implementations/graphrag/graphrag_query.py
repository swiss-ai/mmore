from typing import List, Dict, Any, Literal
from dataclasses import dataclass
from pathlib import Path

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma.vectorstores import Chroma as ChromaVectorStore
from langchain_core.language_models import LanguageModelLike

from end2end.utils import load_config

from end2end.rag.langchain_graphrag.query.global_search import GlobalSearch
from end2end.rag.langchain_graphrag.query.global_search.community_weight_calculator import (
    CommunityWeightCalculator,
)
from end2end.rag.langchain_graphrag.query.global_search.key_points_aggregator.vllm_aggregator import vLLMKeyPointsAggregator
from end2end.rag.langchain_graphrag.query.global_search.key_points_aggregator import (
    KeyPointsAggregatorPromptBuilder,
    KeyPointsContextBuilder,
)
from end2end.rag.langchain_graphrag.query.global_search.key_points_generator.vllm_generator import vLLMKeyPointsGenerator
from end2end.rag.langchain_graphrag.query.global_search.key_points_generator import (
    CommunityReportContextBuilder,
    KeyPointsGeneratorPromptBuilder,
)
from end2end.rag.langchain_graphrag.query.local_search.vllm_search import vLLMLocalSearch
from end2end.rag.langchain_graphrag.query.local_search import (
    LocalSearch,
    LocalSearchPromptBuilder,
    LocalSearchRetriever,
)
from end2end.rag.langchain_graphrag.query.local_search.context_builders import (
    ContextBuilder,
)
from end2end.rag.langchain_graphrag.query.local_search.context_selectors import (
    ContextSelector,
)
from end2end.rag.langchain_graphrag.types.graphs.community import CommunityLevel
from end2end.rag.langchain_graphrag.utils import TiktokenCounter
from end2end.rag.langchain_graphrag.common import make_embedding_instance, load_artifacts, get_artifacts_dir_name
from end2end.rag.models.vllm_model import vLLMWrapper
from end2end.rag.langchain_graphrag.indexing.artifacts import IndexerArtifacts

@dataclass
class GraphRAGQueryConfig:
    """Configuration for GraphRAG querying."""
    llm_type: str
    llm_model: str
    embedding_type: str
    embedding_model: str
    output_dir: str = "./artifacts"
    cache_dir: str = "./cache"
    search_type: Literal["global", "local", "hybrid"] = "hybrid"
    community_level: int = 2
    collection_name: str = "multimeditron"
    show_references: bool = True
    repeat_instructions: bool = False
    output_raw: bool = False

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

class GraphRAGQuerying:
    """Main GraphRAG pipeline combining retrieval and generation."""
    
    def __init__(
            self, 
            llm_model: str, 
            llm_type: str,
            embedding_type: str,
            embedding_model: str,
            output_dir: Path, 
            cache_dir: Path,
            search_type: Literal["global", "local", "hybrid"], 
            community_level: int, 
            collection_name: str,
            show_references: bool, 
            repeat_instructions: bool, 
            output_raw: bool, 
            llm: LanguageModelLike = None
            ):
        
        if llm is None:
            llm = vLLMWrapper(llm_model)

        self.search_type = search_type

        artifacts = load_artifacts(output_dir / get_artifacts_dir_name(llm_model))
        
        self.global_search = self._setup_global_search(llm, community_level, show_references, repeat_instructions, output_raw, artifacts) if search_type in ["global", "hybrid"] else None
        self.local_search = self._setup_local_search(llm, community_level, show_references, repeat_instructions, output_raw, collection_name, output_dir, embedding_type, embedding_model, cache_dir, artifacts) if search_type in ["local", "hybrid"] else None

    def _setup_global_search(
            self, 
            llm: LanguageModelLike, 
            community_level: int, 
            show_references: bool, 
            repeat_instructions: bool, 
            output_raw: bool,
            artifacts: IndexerArtifacts
            ) -> GlobalSearch:
        """Initialize global search components"""
        
        report_context_builder = CommunityReportContextBuilder(
            community_level=CommunityLevel(community_level),
            weight_calculator=CommunityWeightCalculator(),
            artifacts=artifacts,
            token_counter=TiktokenCounter(),
            max_tokens=8000,
        )

        kp_generator = vLLMKeyPointsGenerator(
            llm=llm,
            prompt_builder=KeyPointsGeneratorPromptBuilder(
                show_references=show_references,
                repeat_instructions=repeat_instructions
            ),
            context_builder=report_context_builder,
        )

        kp_aggregator = vLLMKeyPointsAggregator(
            llm=llm,
            prompt_builder=KeyPointsAggregatorPromptBuilder(
                show_references=show_references,
                repeat_instructions=repeat_instructions,
            ),
            context_builder=KeyPointsContextBuilder(
                token_counter=TiktokenCounter(),
                max_tokens=24000,
            ),
            output_raw=output_raw,
        )

        return GlobalSearch(
            kp_generator=kp_generator,
            kp_aggregator=kp_aggregator,
            generation_chain_config={"tags": ["kp-generation"]},
            aggregation_chain_config={"tags": ["kp-aggregation"]},
        )

    def _setup_local_search(
            self, 
            llm: LanguageModelLike, 
            community_level: int, 
            show_references: bool, 
            repeat_instructions: bool, 
            output_raw: bool,
            collection_name: str, 
            output_dir: Path, 
            embedding_type: str, 
            embedding_model: str, 
            cache_dir: Path,
            artifacts: IndexerArtifacts
            ) -> LocalSearch:
        """Initialize local search components"""

        entities_vector_store = ChromaVectorStore(
            collection_name=f"entity-{collection_name}",
            persist_directory=str(output_dir / "vector_stores"),
            embedding_function=make_embedding_instance(
                embedding_type=embedding_type,
                model=embedding_model,
                cache_dir=cache_dir,
            ),
        )

        context_selector = ContextSelector.build_default(
            entities_vector_store=entities_vector_store,
            entities_top_k=10,
            community_level=CommunityLevel(community_level),
        )

        context_builder = ContextBuilder.build_default(
            token_counter=TiktokenCounter(),
        )

        retriever = LocalSearchRetriever(
            context_selector=context_selector,
            context_builder=context_builder,
            artifacts=artifacts,
        )

        return vLLMLocalSearch(
            prompt_builder=LocalSearchPromptBuilder(
                show_references=show_references,
                repeat_instructions=repeat_instructions,
            ),
            llm=llm, 
            retriever=retriever,
            output_raw=output_raw,
        )

    def _execute_query(self, query: str) -> str:
        """Execute query using configured search method(s)"""
        results = []
        
        if self.search_type in ["global", "hybrid"]:
            global_result = self.global_search.invoke(query)
            results.append(global_result)
            
        if self.search_type in ["local", "hybrid"]:
            local_result = self.local_search().invoke(query, config={"tags": ["local-search"]})
            results.append(local_result)
            
        if len(results) > 1:
            # Combine results for hybrid search
            # TODO something better ?
            combined = "\nGlobal Search:\n" + results[0] + "\nLocal Search:\n" + results[1]
            return combined
            
        return results[0]

    def __call__(self, queries: Dict[str, Any] | List[Dict[str, Any]], 
                 return_dict: bool = False) -> List[Dict[str, str]]:
        if isinstance(queries, dict):
            queries = [queries]
            
        results = []
        for query in queries:
            answer = self._execute_query(query['input'])
            result = {
                'input': query['input'],
                'docs': query.get('docs', None),
                'answer': answer
            }
            results.append(result)
            
        if return_dict:
            return results
        return [result['answer'] for result in results]

    @classmethod
    def from_config(cls, config_path: str, llm: LanguageModelLike = None):
        """Create querying instance from config file"""
        config = load_config(config_path, GraphRAGQueryConfig)
        return cls(
            config.llm_model, 
            config.llm_type, 
            config.embedding_type, 
            config.embedding_model, 
            config.output_dir, 
            config.cache_dir, 
            config.search_type, 
            config.community_level, 
            config.collection_name, 
            config.show_references, 
            config.repeat_instructions, 
            config.output_raw, 
            llm
        )

    def stream(self, query: str):
        """Support streaming responses"""
        if self.config.search_type == "global":
            yield from self.global_search.stream(query)
        elif self.config.search_type == "local":
            yield from self.local_search().stream(query, config={"tags": ["local-search"]})
        else:
            # For hybrid, stream both sequentially
            yield "Global Search Results:\n"
            yield from self.global_search.stream(query)
            yield "\n\nLocal Search Results:\n"
            yield from self.local_search().stream(query, config={"tags": ["local-search"]})