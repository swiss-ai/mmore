"""Graph embedding generation using node2vec."""

import graspologic as gl
import networkx as nx
import numpy as np

from mmore.index.implementations.graphrag._graph_utils import stable_largest_connected_component
from mmore.types.graphrag.  graphs.embedding import GraphEmbeddingGenerator


class Node2VectorGraphEmbeddingGenerator(GraphEmbeddingGenerator):
    def __init__(
        self,
        *,
        use_lcc: bool = True,
        dimensions: int = 1536,
        num_walks: int = 10,
        walk_length: int = 40,
        window_size: int = 2,
        num_iter: int = 3,
        random_seed: int = 86,
    ):
        self._use_lcc = use_lcc
        self._dimensions = dimensions
        self._num_walks = num_walks
        self._walk_length = walk_length
        self._window_size = window_size
        self._num_iter = num_iter
        self._random_seed = random_seed

    def run(
        self,
        graph: nx.Graph,
    ) -> dict[str, np.ndarray]:
        if self._use_lcc:
            graph = stable_largest_connected_component(graph)

        lcc_tensors = gl.embed.node2vec_embed(
            graph=graph,
            dimensions=self._dimensions,
            window_size=self._window_size,
            iterations=self._num_iter,
            num_walks=self._num_walks,
            walk_length=self._walk_length,
            random_seed=self._random_seed,
        )

        embeddings = lcc_tensors[0]
        nodes = lcc_tensors[1]

        pairs = zip(nodes, embeddings, strict=True)
        sorted_pairs = sorted(pairs, key=lambda x: x[0])

        return dict(sorted_pairs)
