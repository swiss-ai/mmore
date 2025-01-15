from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

from mmore.types.type import MultimodalSample
from mmore.utils import load_config
from mmore.utils.graphrag.artifacts import save_artifacts, get_artifacts_dir_name
from langchain_core.language_models import LanguageModelLike


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


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma.vectorstores import Chroma as ChromaVectorStore


@dataclass
class GraphRAGIndexerConfig:
    llm_type: str
    llm_model: str
    embedding_type: str
    embedding_model: str
    collection_name: str = "multimeditron"
    output_dir: str = "./temp/artifacts"
    cache_dir: str ="./temp/cache"
    chunk_size: int = 1200
    chunk_overlap: int = 100

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

class GraphRAGIndexer:
    """Adapter class that implements GraphRAG indexing for the Meditron pipeline"""
    indexer: SimpleIndexer
    output_dir: Path
    llm_model: str
    
    def __init__(
            self, 
            llm_model: str,
            llm_type: str,
            embedding_type: str, 
            embedding_model: str, 
            collection_name: str, 
            output_dir: Path, 
            cache_dir: Path, 
            chunk_size: int, 
            chunk_overlap: int, 
            llm: LanguageModelLike = None, 
            ):
        
        if llm is None:
            self.llm = vLLMWrapper(llm_model)

        self.llm_model = llm_model
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
        
        community_detector = HierarchicalLeidenCommunityDetector()

        entities_collection = f"entity-{collection_name}"
        vector_store = ChromaVectorStore(
            collection_name=entities_collection,
            persist_directory=str(output_dir / "vector_stores"),
            embedding_function=make_embedding_instance(
                embedding_type,
                embedding_model, 
                cache_dir),
        )

        generators = {
            'entities': EntitiesArtifactsGenerator(entities_vector_store=vector_store),
            'relationships': RelationshipsArtifactsGenerator(),
            'text_units': TextUnitsArtifactsGenerator(),
            'communities': vLLMCommunitiesReportsArtifactsGenerator(
                report_generator=vLLMCommunityReportGenerator.build_default(
                    llm=llm,
                    chain_config={"tags": ["community-report"]},
                ),
                report_writer=CommunityReportWriter(),
            )
        }
        
        self.indexer = SimpleIndexer(
            text_unit_extractor=text_unit_extractor,
            graph_generator=graph_generator,
            community_detector=community_detector,
            entities_artifacts_generator=generators['entities'],
            relationships_artifacts_generator=generators['relationships'],
            text_units_artifacts_generator=generators['text_units'],
            communities_report_artifacts_generator=generators['communities'],
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
    def from_config(cls, config_path: str, llm: LanguageModelLike = None):
        """Create indexer instance from config file"""
        config = load_config(config_path, GraphRAGIndexerConfig)
        return cls(
            llm_model=config.llm_model,
            llm_type=config.llm_type,
            embedding_type=config.embedding_type,
            embedding_model=config.embedding_model,
            collection_name=config.collection_name,
            output_dir=config.output_dir,
            cache_dir=config.cache_dir,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            llm=llm,
        )

    @classmethod
    def from_documents(
        cls,
        config_path: str,
        documents: List[MultimodalSample],
        collection_name: str = 'med_docs',
        partition_name: Optional[str] = None,
        llm: LanguageModelLike = None
    ):
        """Create and run indexer directly from documents"""
        indexer = cls.from_config(config_path, llm)
        indexer.index_documents(
            documents=documents,
        )
        return indexer