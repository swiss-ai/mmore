from dataclasses import dataclass
from typing import NewType, Protocol

import networkx as nx

CommunityId = NewType("CommunityId", int)
CommunityLevel = NewType("CommunityLevel", int)


@dataclass
class CommunityNode:
    name: str
    parent_cluster: CommunityId | None
    is_final_cluster: bool


@dataclass
class Community:
    id: CommunityId
    nodes: list[CommunityNode]


@dataclass
class CommunityDetectionResult:
    communities: dict[CommunityLevel, dict[CommunityId, Community]]

    def communities_at_level(self, level: CommunityLevel) -> list[Community]:
        return list(self.communities[level].values())


class CommunityDetector(Protocol):
    def run(self, graph: nx.Graph) -> CommunityDetectionResult: ...
