from typing import cast

import networkx as nx
from graspologic.partition import (
    HierarchicalCluster,
    HierarchicalClusters,
    hierarchical_leiden,
)

from mmore.index.implementations.graphrag._graph_utils import stable_largest_connected_component
from mmore.types.graphrag.graphs.community import (
    Community,
    CommunityDetectionResult,
    CommunityDetector,
    CommunityId,
    CommunityLevel,
    CommunityNode,
)


class HierarchicalLeidenCommunityDetector(CommunityDetector):
    def __init__(
        self,
        *,
        use_lcc: bool = True,
        max_cluster_size: int = 10,
        seed: int = 0xDEADBEEF,
    ):
        self._use_lcc = use_lcc
        self._max_cluster_size = max_cluster_size
        self._seed = seed

    def run(self, graph: nx.Graph) -> CommunityDetectionResult:
        if self._use_lcc:
            graph = stable_largest_connected_component(graph)

        community_mapping: HierarchicalClusters = hierarchical_leiden(
            graph,
            max_cluster_size=self._max_cluster_size,
            random_seed=self._seed,
        )

        communities: dict[CommunityLevel, dict[CommunityId, Community]] = {}

        partition: HierarchicalCluster
        for partition in community_mapping:
            partition_level = cast(CommunityLevel, partition.level)
            partition_cluster = cast(CommunityId, partition.cluster)

            communities_at_level = communities.get(partition_level, {})
            community = communities_at_level.get(
                partition_cluster,
                Community(id=partition_cluster, nodes=[]),
            )
            community.nodes.append(
                CommunityNode(
                    name=partition.node,
                    parent_cluster=cast(CommunityId, partition.parent_cluster),
                    is_final_cluster=partition.is_final_cluster,
                )
            )

            communities_at_level[partition_cluster] = community
            communities[partition_level] = communities_at_level

        return CommunityDetectionResult(communities=communities)
