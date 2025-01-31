"""Default OutputParser for Entity Relationship Extraction."""

import html
import numbers
import re
from collections.abc import Mapping
from typing import Any

import networkx as nx
from langchain_core.output_parsers import BaseOutputParser

_ENTITY_ATTRIBUTES_LENGTH = 4
_RELATIONSHIP_ATTRIBUTES_LENGTH = 5


def _clean_str(input_str: Any) -> str:
    """Remove HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input_str, str):
        return input_str

    result = html.unescape(input_str.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def _unpack_descriptions(data: Mapping) -> list[str]:
    return data.get("description", [])


class EntityExtractionOutputParser(BaseOutputParser[nx.Graph]):
    """OutputParser for extracting entities and relationships.

    This parses the output from the LLM based on the conventions & formats
    used in the default prompt.

    Note:
        You would not use this class directly.


    Attributes:
        tuple_delimiter (str): Delimiter used to separate attributes within a record.
        record_delimiter (str): Delimiter used to separate records.
    """

    tuple_delimiter: str
    record_delimiter: str

    def _process_entity(self, record_attributes: list[str], graph: nx.Graph) -> None:
        if (record_attributes[0] != '"entity"') or (
            len(record_attributes) < _ENTITY_ATTRIBUTES_LENGTH
        ):
            return

        # add this record as a node in the G
        entity_name = _clean_str(record_attributes[1].upper())
        entity_type = _clean_str(record_attributes[2].upper())
        entity_description = _clean_str(record_attributes[3])

        if entity_name in graph.nodes():
            node = graph.nodes[entity_name]
            node["description"] = list(
                {
                    *_unpack_descriptions(node),
                    entity_description,
                }
            )

            node["entity_type"] = (
                entity_type if entity_type != "" else node["entity_type"]
            )
        else:
            graph.add_node(
                entity_name,
                type=entity_type,
                description=[entity_description],
            )

    def _process_relationship(
        self,
        record_attributes: list[str],
        graph: nx.Graph,
    ) -> None:
        if (
            record_attributes[0] != '"relationship"'
            or len(record_attributes) < _RELATIONSHIP_ATTRIBUTES_LENGTH
        ):
            return

        # add this record as edge
        source = _clean_str(record_attributes[1].upper())
        target = _clean_str(record_attributes[2].upper())
        edge_description = _clean_str(record_attributes[3])

        weight = (
            float(record_attributes[-1])
            if isinstance(record_attributes[-1], numbers.Number)
            else 1.0
        )
        if source not in graph.nodes():
            graph.add_node(
                source,
                type="",
                description=[""],
            )
        if target not in graph.nodes():
            graph.add_node(
                target,
                type="",
                description=[""],
            )
        if graph.has_edge(source, target):
            edge_data = graph.get_edge_data(source, target)
            if edge_data is not None:
                weight += edge_data["weight"]
                edge_descriptions = list(
                    {
                        *_unpack_descriptions(edge_data),
                        edge_description,
                    }
                )
        else:
            edge_descriptions = [edge_description]

        graph.add_edge(source, target, weight=weight, description=edge_descriptions)

    def _process_record(self, graph: nx.Graph, record: str) -> None:
        record = re.sub(r"^\(|\)$", "", record.strip())
        record_attributes = record.split(self.tuple_delimiter)

        self._process_entity(record_attributes, graph)
        self._process_relationship(record_attributes, graph)

    def parse(self, text: str) -> nx.Graph:
        """Parses the given text and returns a networkx Graph object.

        Parameters:
            text (str): The text to be parsed.

        Returns:
            The parsed graph object.
        """
        graph: nx.Graph = nx.Graph()
        records = [r.strip() for r in text.split(self.record_delimiter)]
        for record in records:
            self._process_record(graph, record)
        return graph

    @property
    def _type(self) -> str:
        return "entity_extraction_output_parser"
