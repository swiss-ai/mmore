import itertools
from typing import Sequence, TypedDict

import networkx as nx
from pydantic import BaseModel, Field

from mmore.types.graphrag.graphs.community import Community


class Entity(TypedDict):
    id: str
    name: str
    type: str
    description: str
    degree: int


class Relationship(TypedDict):
    id: str
    source: str
    target: str
    description: str
    rank: int


def entity_from_graph(name: str, graph: nx.Graph) -> Entity:
    node = graph.nodes[name]
    return Entity(
        id=node["human_readable_id"],
        name=name,
        type=node["type"],
        description=node["description"],
        degree=node["degree"],
    )


def relationship_from_graph(
    pair: tuple[str, str],
    graph: nx.Graph,
) -> Relationship:
    n1, n2 = pair
    edge = graph.edges[n1, n2]
    return Relationship(
        id=edge["human_readable_id"],
        source=n1,
        target=n2,
        description=edge["description"],
        rank=edge["rank"],
    )


class CommunityFinding(BaseModel):
    summary: str = Field(description="Insight summary")
    explanation: str = Field(description="Insight explanation")


class CommunityReportResult(BaseModel):
    title: str = Field(description="Title of the report")
    summary: str = Field(description="Summary of the report")
    rating: float = Field(description="Impact severity rating of the report")
    rating_explanation: str = Field(
        description="Single sentence explanation of the IMPACT severity rating"
    )
    findings: list[CommunityFinding] = Field(description="Detailed findings")


def get_info(
    community: Community,
    graph: nx.Graph,
) -> tuple[Sequence[Entity], Sequence[Relationship]]:
    nodes = [n.name for n in community.nodes]
    entities = [entity_from_graph(n, graph) for n in nodes]

    node_pairs = itertools.combinations(entities, 2)

    pairs_with_edges = []
    for n1, n2 in node_pairs:
        if not graph.has_edge(n1["name"], n2["name"]):
            continue
        pairs_with_edges.append((n1["name"], n2["name"]))

    relationships = [relationship_from_graph(p, graph) for p in pairs_with_edges]

    return entities, relationships
