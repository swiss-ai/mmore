import networkx as nx
import numpy as np
import pandas as pd
from langchain_core.vectorstores import VectorStore

from mmore.types.graphrag.graphs.community import (
    CommunityDetectionResult,
    CommunityId,
)
from mmore.types.graphrag.graphs.embedding import GraphEmbeddingGenerator


def _make_entity_to_communities_map(
    detection_result: CommunityDetectionResult,
) -> dict[str, list[CommunityId]]:
    entity_to_communities: dict[str, list[CommunityId]] = {}
    for level in detection_result.communities:
        communities = detection_result.communities_at_level(level)
        for c in communities:
            for node in c.nodes:
                entity_to_communities.setdefault(node.name, []).append(c.id)
    return entity_to_communities


class EntitiesArtifactsGenerator:
    def __init__(
        self,
        entities_vector_store: VectorStore,
        graph_embedding_generator: GraphEmbeddingGenerator | None = None,
    ):
        self._graph_embedding_generator = graph_embedding_generator
        self._entities_vector_store = entities_vector_store

    def _unpack_nodes(
        self,
        graph: nx.Graph,
        entity_to_commnunities_map: dict[str, list[CommunityId]],
        graph_embeddings: dict[str, np.ndarray] | None,
    ) -> pd.DataFrame:
        records = [
            {
                "title": label,
                **(node_data or {}),
                "communities": entity_to_commnunities_map.get(label),
                "graph_embedding": graph_embeddings.get(label)
                if graph_embeddings
                else None,
            }
            for label, node_data in graph.nodes(data=True)
        ]
        return pd.DataFrame.from_records(records)

    def run(
        self,
        detection_result: CommunityDetectionResult,
        graph: nx.Graph,
    ) -> pd.DataFrame:
        # Step 1 (Optional)
        # Generate graph embeddings
        graph_embeddings = (
            self._graph_embedding_generator.run(graph)
            if self._graph_embedding_generator
            else None
        )

        # Step 2
        # Extract the information to embed from the graph
        # and put in the vectorstore
        texts_to_embed = []
        texts_metadata = []
        texts_ids = []
        for name, node_data in graph.nodes(data=True):
            text_description = node_data.get("description")
            texts_ids.append(node_data.get("id"))
            texts_to_embed.append(f"{name}:{text_description}")

            # Bug in langchain vectorstore retrival that
            # does not populate Document.id field.
            #
            # Hence add entity_id as an additional field
            # in the metadata
            texts_metadata.append(
                dict(
                    name=name,
                    description=text_description,
                    degree=node_data.get("degree"),
                    entity_id=node_data.get(
                        "id"
                    ),  # TODO: Remove once langchain is fixed
                )
            )
        """
        self._entities_vector_store.add_texts(
            texts_to_embed,
            metadatas=texts_metadata,
            ids=texts_ids,
        )
        """

        entity_to_commnunities_map = _make_entity_to_communities_map(detection_result)

        # Step 3
        # Make a dataframe
        return self._unpack_nodes(
            graph,
            entity_to_commnunities_map,
            graph_embeddings,
        )
