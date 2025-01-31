import itertools
import logging
from typing import NamedTuple

import pandas as pd

_LOGGER = logging.getLogger(__name__)


class RelationshipsSelectionResult(NamedTuple):
    in_network_relationships: pd.DataFrame
    out_network_relationships: pd.DataFrame


def _find_in_network_relationships(
    df_entities: pd.DataFrame,
    df_relationships: pd.DataFrame,
    source_column_name: str = "source_id",
    target_column_name: str = "target_id",
    entity_column_name: str = "id",
) -> pd.DataFrame:
    entities_ids = df_entities[entity_column_name].tolist()
    entities_pairs = list(itertools.combinations(entities_ids, 2))

    def filter_in_network_relationships(source: str, target: str) -> bool:
        check_1 = (source, target) in entities_pairs
        check_2 = (target, source) in entities_pairs
        return check_1 == True or check_2 == True  # noqa: E712

    df_relationships["is_in_network"] = df_relationships.apply(
        lambda x: filter_in_network_relationships(
            x[source_column_name], x[target_column_name]
        ),
        axis=1,
    )

    df_relationships = df_relationships[df_relationships["is_in_network"] == True]  # noqa: E712

    df_relationships.drop(columns=["is_in_network"], inplace=True)

    # sort the relationships by rank
    df_relationships = df_relationships.sort_values(
        by="rank", ascending=False
    ).reset_index(drop=True)

    if _LOGGER.isEnabledFor(logging.DEBUG):
        import tableprint

        how_many = len(df_relationships)

        tableprint.banner(f"Selected {how_many} In-Network Relationships")
        tableprint.dataframe(df_relationships[["source", "target", "rank"]])

    return df_relationships


def _find_out_network_relationships(
    df_entities: pd.DataFrame,
    df_relationships: pd.DataFrame,
    top_k: int = 10,
    source_column_name: str = "source_id",
    target_column_name: str = "target_id",
    entity_column_name: str = "id",
) -> pd.DataFrame:
    entities_ids = df_entities[entity_column_name].tolist()

    # top_k is budget for out-network relationships
    relationship_budget = top_k * len(entities_ids)

    def filter_out_network_relationships(source: str, target: str) -> bool:
        if source in entities_ids and target not in entities_ids:
            return True
        if target in entities_ids and source not in entities_ids:  # noqa: SIM103
            return True

        return False

    df_relationships["is_out_network"] = df_relationships.apply(
        lambda x: filter_out_network_relationships(
            x[source_column_name], x[target_column_name]
        ),
        axis=1,
    )

    df_relationships = df_relationships[df_relationships["is_out_network"] == True]  # noqa: E712

    df_relationships.drop(columns=["is_out_network"], inplace=True)

    # now we need to prioritize based on which external
    # entities have the most connection with the selected entities
    # we will do this by counting the number of relationships
    # each external entity has with the selected entities
    source_external_entities = df_relationships[
        ~df_relationships[source_column_name].isin(entities_ids)
    ][source_column_name]

    target_external_entities = df_relationships[
        ~df_relationships[target_column_name].isin(entities_ids)
    ][target_column_name]

    df_relationships = (
        df_relationships.merge(
            source_external_entities.value_counts(),
            how="left",
            left_on=source_column_name,
            right_on=source_column_name,
        )
        .fillna(0)
        .rename(columns={"count": "source_count"})
    )

    df_relationships = (
        df_relationships.merge(
            target_external_entities.value_counts(),
            how="left",
            left_on=target_column_name,
            right_on=target_column_name,
        )
        .fillna(0)
        .rename(columns={"count": "target_count"})
    )

    df_relationships["links"] = (
        df_relationships["source_count"] + df_relationships["target_count"]
    )

    df_relationships = df_relationships.sort_values(
        by=["links", "rank"],
        ascending=[False, False],
    ).reset_index(drop=True)

    # time to use the budget
    df_relationships = df_relationships.head(relationship_budget)

    if _LOGGER.isEnabledFor(logging.DEBUG):
        import tableprint

        how_many = len(df_relationships)

        tableprint.banner(f"Selected {how_many} Out-Network Relationships")
        tableprint.dataframe(df_relationships[["source", "target", "rank", "links"]])

    return df_relationships


class RelationshipsSelector:
    def __init__(self, top_k_out_network: int = 5):
        self._top_k_out_network = top_k_out_network

    def run(
        self,
        df_entities: pd.DataFrame,
        df_relationships: pd.DataFrame,
    ) -> RelationshipsSelectionResult:
        in_network_relationships = _find_in_network_relationships(
            df_entities,
            df_relationships.copy(deep=True),
        )

        out_network_relationships = _find_out_network_relationships(
            df_entities,
            df_relationships.copy(deep=True),
            top_k=self._top_k_out_network,
        )

        return RelationshipsSelectionResult(
            in_network_relationships,
            out_network_relationships,
        )
