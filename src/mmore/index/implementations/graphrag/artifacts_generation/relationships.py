import networkx as nx
import pandas as pd
from langchain_core.vectorstores import VectorStore


class RelationshipsArtifactsGenerator:
    def __init__(
        self,
        relationships_vector_store: VectorStore | None = None,
    ):
        self._relationships_vector_store = relationships_vector_store

    def _unpack_edges(self, graph: nx.Graph) -> pd.DataFrame:
        records = [
            {
                "source": source,
                "target": target,
                "source_id": graph.nodes[source].get("id"),
                "target_id": graph.nodes[target].get("id"),
                **(edge_data or {}),
            }
            for source, target, edge_data in graph.edges(data=True)
        ]
        return pd.DataFrame.from_records(records)

    def _embed_relationships(self, graph: nx.Graph) -> None:
        # Extract the information to embed from the graph
        # and put in the vectorstore
        texts_to_embed = []
        texts_metadata = []
        texts_ids = []
        for source, target, edge_data in graph.edges(data=True):
            text_description = edge_data.get("description")
            texts_ids.append(edge_data.get("id"))
            texts_to_embed.append(text_description)

            # Bug in langchain vectorstore retrival that
            # does not populate Document.id field.
            #
            # Hence add relationship_id as an additional field
            # in the metadata
            texts_metadata.append(
                dict(
                    source=source,
                    target=target,
                    description=text_description,
                    rank=edge_data.get("rank"),
                    relationship_id=edge_data.get(
                        "id"
                    ),  # TODO: Remove once langchain is fixed
                )
            )

        assert self._relationships_vector_store is not None

        self._relationships_vector_store.add_texts(
            texts_to_embed,
            metadatas=texts_metadata,
            ids=texts_ids,
        )

    def run(self, graph: nx.Graph) -> pd.DataFrame:
        if self._relationships_vector_store:
            self._embed_relationships(graph)

        return self._unpack_edges(graph)
