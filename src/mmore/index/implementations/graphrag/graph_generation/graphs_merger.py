from enum import Enum
from random import Random
from typing import Any

import networkx as nx

from mmore.utils.graphrag.uuid import gen_uuid


class AttributesToMerge(str, Enum):
    text_unit_ids = "text_unit_ids"
    description = "description"
    weight = "weight"


def merge_attributes(
    *,
    target_node: dict[str, Any],
    source_node: dict[str, Any],
    attribs: list[AttributesToMerge],
):
    for attrib in attribs:
        # I am expecting the attributes are not missing
        target_attrib = target_node.get(attrib)
        source_attrib = source_node.get(attrib)
        if attrib == AttributesToMerge.weight:
            target_node[attrib] = int(target_attrib) + int(source_attrib)
        else:
            target_node[attrib].extend(source_attrib)
            target_node[attrib] = sorted(set(target_node[attrib]))


def merge_nodes(*, target_graph: nx.Graph, sub_graph: nx.Graph):
    for node in sub_graph.nodes:
        if node not in target_graph.nodes:
            target_graph.add_node(node, **(sub_graph.nodes[node] or {}))
        else:
            merge_attributes(
                target_node=target_graph.nodes[node],
                source_node=sub_graph.nodes[node],
                attribs=[
                    AttributesToMerge.text_unit_ids,
                    AttributesToMerge.description,
                ],
            )


def merge_edges(*, target_graph: nx.Graph, sub_graph: nx.Graph):
    for source, target, edge_data in sub_graph.edges(data=True):
        if not target_graph.has_edge(source, target):
            target_graph.add_edge(source, target, **(edge_data or {}))
        else:
            merge_attributes(
                target_node=target_graph.edges[(source, target)],
                source_node=edge_data,
                attribs=[
                    AttributesToMerge.text_unit_ids,
                    AttributesToMerge.description,
                    AttributesToMerge.weight,
                ],
            )


class GraphsMerger:
    def __init__(self, seed: int = 0xF001):
        self._seed = seed

    def __call__(
        self,
        graphs: list[nx.Graph],
    ) -> nx.Graph:
        merged_graph: nx.Graph = nx.Graph()
        for g in graphs:
            merge_nodes(target_graph=merged_graph, sub_graph=g)
            merge_edges(target_graph=merged_graph, sub_graph=g)

        # add degree as an attribute
        for node_degree in merged_graph.degree:
            merged_graph.nodes[str(node_degree[0])]["degree"] = int(node_degree[1])

        # add source degree, target degree and rank as attributes
        # to the edges
        for source, target in merged_graph.edges():
            source_degree = merged_graph.nodes[source]["degree"]
            target_degree = merged_graph.nodes[target]["degree"]
            merged_graph.edges[source, target]["source_degree"] = source_degree
            merged_graph.edges[source, target]["target_degree"] = target_degree
            merged_graph.edges[source, target]["rank"] = source_degree + target_degree

        random = Random(self._seed)  # noqa: S311

        # add ids to nodes
        for index, node in enumerate(merged_graph.nodes()):
            merged_graph.nodes[node]["human_readable_id"] = index
            merged_graph.nodes[node]["id"] = str(gen_uuid(random))

        # add ids to edges
        for index, edge in enumerate(merged_graph.edges()):
            merged_graph.edges[edge]["human_readable_id"] = index
            merged_graph.edges[edge]["id"] = str(gen_uuid(random))

        return merged_graph
