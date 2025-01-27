from dataclasses import dataclass
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from mmore.rag.base_retriever import RetrieverConfig

from mmore.utils import load_config
from mmore.utils.graphrag.artifacts import load_artifacts

from mmore.index.implementations.graphrag.artifacts import IndexerArtifacts

from .context_builders import ContextBuilder
from .context_selectors import ContextSelector

from mmore.types.graphrag.graphs.community import CommunityLevel
from mmore.utils.graphrag.token_counter import TiktokenCounter

from mmore.rag.model.dense.base import DenseModel, DenseModelConfig

from langchain_chroma.vectorstores import Chroma as ChromaVectorStore

@dataclass
class GraphRAGLocalRetrieverConfig(RetrieverConfig):
    artifact_path: str | Path
    output_dir: str | Path
    dense_model: DenseModelConfig
    collection_name: str = "my_docs"
    community_level: int = 3

    def __post_init__(self):
        if isinstance(self.artifact_path, str):
            self.artifact_path = Path(self.artifact_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


class GraphRAGLocalRetriever(BaseRetriever):
    context_selector: ContextSelector
    context_builder: ContextBuilder
    artifacts: IndexerArtifacts

    @classmethod
    def from_config(cls, config: str | GraphRAGLocalRetrieverConfig):
        if isinstance(config, str):
            config = load_config(config, GraphRAGLocalRetrieverConfig)

        artifacts = load_artifacts(config.artifact_path)


        entities_vector_store = ChromaVectorStore(
            collection_name=f"entity-{config.collection_name}",
            persist_directory=str(config.output_dir / "vector_stores"),
            embedding_function=DenseModel.from_config(config.dense_model),
        )

        context_selector = ContextSelector.build_default(
            entities_vector_store=entities_vector_store,
            entities_top_k=10,
            community_level=CommunityLevel(config.community_level),
        )

        context_builder = ContextBuilder.build_default(
            token_counter=TiktokenCounter(),
        )

        return cls(
            context_selector=context_selector,
            context_builder=context_builder,
            artifacts=artifacts,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa: ARG002
    ) -> list[Document]:
        context_selection_result = self.context_selector.run(
            query=query,
            artifacts=self.artifacts,
        )

        return self.context_builder(context_selection_result)
