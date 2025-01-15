from typing import NamedTuple

import networkx as nx
import pandas as pd
import tableprint

from mmore.types.graphrag.graphs.community import CommunityDetectionResult


class IndexerArtifacts(NamedTuple):
    entities: pd.DataFrame
    relationships: pd.DataFrame
    text_units: pd.DataFrame
    communities_reports: pd.DataFrame
    merged_graph: nx.Graph | None = None
    summarized_graph: nx.Graph | None = None
    communities: CommunityDetectionResult | None = None

    def _entity_info(self, top_k: int) -> None:
        tableprint.banner("Entities")

        rows = [
            ["Count", len(self.entities)],
            ["Number of types", len(self.entities["type"].unique())],
        ]
        tableprint.table(rows)

        # entity types
        tableprint.banner("Entity Types")
        rows = []
        for entity_type, count in self.entities["type"].value_counts().items():
            rows.append([entity_type, count])
        tableprint.table(rows, ["Type", "Count"])

        # entities for which the generated type is empty string
        empty_type_entities = self.entities[self.entities["type"] == ""]
        if not empty_type_entities.empty:
            tableprint.banner("Entities with Empty Type")
            tableprint.dataframe(empty_type_entities[["title", "degree"]])

        # k most connected entities
        by_degree = self.entities.sort_values("degree", ascending=False)[:top_k]

        # print k most connected entities
        tableprint.banner(f"{top_k} Most Connected Entities")
        tableprint.dataframe(by_degree[["title", "degree"]])

        # are there entities with degree 0
        zero_degree_entities = self.entities[self.entities["degree"] == 0]
        if not zero_degree_entities.empty:
            tableprint.banner("Disconnected Entities")
            tableprint.dataframe(zero_degree_entities[["title", "degree"]])

    def _relationships_info(self, top_k: int) -> None:
        tableprint.banner("Relationships")
        rows = [["Count", len(self.relationships)]]
        tableprint.table(rows)

        # k highly ranked relationships
        by_rank = self.relationships.sort_values("rank", ascending=False)[:top_k]

        # print 5 most connected entities
        tableprint.banner(f"{top_k} Top Ranked Relationships")
        tableprint.dataframe(by_rank[["source", "target", "rank"]])

    def _text_units_info(self) -> None:
        tableprint.banner("Text Units")
        rows = [["Count", len(self.text_units)]]
        tableprint.table(rows)

    def _communities_reports_info(self) -> None:
        tableprint.banner("Communities Reports")

        levels = self.communities_reports["level"].unique()

        rows = []
        for level in levels:
            communities = self.communities_reports[
                self.communities_reports["level"] == level
            ]
            row = [level, len(communities)]
            rows.append(row)

        tableprint.table(rows, ["Level", "Number of Communities"])

    def report(
        self,
        top_k_entities: int = 5,
        top_k_relationships: int = 5,
    ) -> None:
        self._text_units_info()
        self._entity_info(top_k=top_k_entities)
        self._relationships_info(top_k=top_k_relationships)
        self._communities_reports_info()
