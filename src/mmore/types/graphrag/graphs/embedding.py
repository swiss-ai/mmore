from typing import Protocol

import networkx as nx
import numpy as np


class GraphEmbeddingGenerator(Protocol):
    def run(self, graph: nx.Graph) -> dict[str, np.ndarray]: ...
