from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from mmore.index.implementations.graphrag.artifacts import IndexerArtifacts

from .context_builders import ContextBuilder
from .context_selectors import ContextSelector


class LocalSearchRetriever(BaseRetriever):
    context_selector: ContextSelector
    context_builder: ContextBuilder
    artifacts: IndexerArtifacts

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
