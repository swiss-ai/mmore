from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from langchain_core.language_models import LanguageModelLike
from langchain_core.language_models.chat_models import BaseChatModel



from mmore.types.type import MultimodalSample
from mmore.utils import load_config
from mmore.utils.graphrag.artifacts import save_artifacts, get_artifacts_dir_name

from mmore.rag.llm import LLMConfig, LLM
from mmore.rag.model.dense.base import DenseModel, DenseModelConfig


from mmore.index.implementations.graphrag import SimpleIndexer, TextUnitExtractor
from mmore.index.implementations.graphrag.graph_generation import (
    GraphGenerator,
    GraphsMerger,
)
from mmore.index.implementations.graphrag.graph_generation.entity_relationship_extraction.vllm_extractor import vLLMEntityRelationshipExtractor
from mmore.index.implementations.graphrag.graph_generation.entity_relationship_summarization.vllm_summarizer import vLLMEntityRelationshipDescriptionSummarizer

from mmore.index.implementations.graphrag.graph_clustering.leiden_community_detector import (
    HierarchicalLeidenCommunityDetector,
)
from mmore.index.implementations.graphrag.artifacts_generation import (
    vLLMCommunitiesReportsArtifactsGenerator,
    EntitiesArtifactsGenerator,
    RelationshipsArtifactsGenerator,
    TextUnitsArtifactsGenerator,
)
from mmore.index.implementations.graphrag.report_generation.vllm_generator import vLLMCommunityReportGenerator
from mmore.index.implementations.graphrag.report_generation import CommunityReportWriter

from mmore.index import BaseIndexerConfig, BaseIndexer



from langchain_chroma.vectorstores import Chroma as ChromaVectorStore

import os


@dataclass
class GraphRAGIndexerConfig(BaseIndexerConfig):
    llm: LLMConfig
    dense_model: DenseModelConfig
    output_dir: str = "./temp/artifacts"
    chunk_size: int = 1200
    chunk_overlap: int = 100

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

class GraphRAGIndexer(BaseIndexer):
    """Adapter class that implements GraphRAG indexing for the Meditron pipeline"""
    indexer: SimpleIndexer
    output_dir: Path
    llm_model: str
    
    def __init__(
            self, 
            llm: BaseChatModel,
            dense_model: DenseModel,
            collection_name: str, 
            output_dir: Path, 
            chunk_size: int, 
            chunk_overlap: int, 
            ):
        
        self.llm_model = llm._llm_type

        self.output_dir = output_dir
        
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        text_unit_extractor = TextUnitExtractor(text_splitter=text_splitter)

        entity_extractor = vLLMEntityRelationshipExtractor.build_default(
            llm=llm,
            chain_config={"tags": ["er-extraction"]},
        )
        
        entity_summarizer = vLLMEntityRelationshipDescriptionSummarizer.build_default(
            llm=llm,
            chain_config={"tags": ["er-description-summarization"]},
        )
        
        graph_generator = GraphGenerator(
            er_extractor=entity_extractor,
            graphs_merger=GraphsMerger(),
            er_description_summarizer=entity_summarizer,
        )
        

        vector_store = ChromaVectorStore(
            collection_name=f"entity-{collection_name}",
            persist_directory=str(output_dir / "vector_stores"),
            embedding_function=dense_model,
        )

        communities_report_artifacts_generator = vLLMCommunitiesReportsArtifactsGenerator(
            report_generator=vLLMCommunityReportGenerator.build_default(
                llm=llm,
                chain_config={"tags": ["community-report"]},
            ),
            report_writer=CommunityReportWriter(),
        )
        
        self.indexer = SimpleIndexer(
            text_unit_extractor=text_unit_extractor,
            graph_generator=graph_generator,
            community_detector=HierarchicalLeidenCommunityDetector(),
            entities_artifacts_generator=EntitiesArtifactsGenerator(entities_vector_store=vector_store),
            relationships_artifacts_generator=RelationshipsArtifactsGenerator(),
            text_units_artifacts_generator=TextUnitsArtifactsGenerator(),
            communities_report_artifacts_generator=communities_report_artifacts_generator,
        )

    def index_documents(
        self,
        documents: List[MultimodalSample],
    ):
        """Index documents using GraphRAG approach while maintaining compatibility with Meditron pipeline"""
        
        # Convert MultimodalSamples to Documents
        processed_docs = []
        for doc in documents:            
            # TODO (milouordo): Handle multimodal samples and metadata
            langchain_doc = Document(doc.text)
            processed_docs.append(langchain_doc)

        artifacts = self.indexer.run(processed_docs)
        
        artifacts_dir = self.output_dir / get_artifacts_dir_name(self.llm_model)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        save_artifacts(artifacts, artifacts_dir)
        artifacts.report()
        
        return artifacts

    @classmethod
    def from_config(cls, config: str | GraphRAGIndexerConfig, collection_name: str = 'med_docs'):
        """Create indexer instance from config file"""
        if isinstance(config, str):
            config = load_config(config, GraphRAGIndexerConfig)

        dense_model = DenseModel.from_config(config.dense_model)
        llm = LLM.from_config(config.llm)
        
        return cls(
            llm=llm,
            dense_model=dense_model,
            collection_name=collection_name,
            output_dir=config.output_dir,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    @classmethod
    def from_documents(
        cls,
        config: str | GraphRAGIndexerConfig,
        documents: List[MultimodalSample],
        collection_name: str = 'med_docs',
        partition_name: Optional[str] = None,
    ):
        """Create and run indexer directly from documents"""
        indexer = cls.from_config(config, collection_name=collection_name)
        indexer.index_documents(
            documents=documents,
        )
        return indexer