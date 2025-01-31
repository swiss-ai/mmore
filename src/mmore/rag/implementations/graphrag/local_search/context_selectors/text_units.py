"""Build the TextUnit context for the LocalSearch algorithm."""

import logging
from typing import TypedDict

import pandas as pd

_LOGGER = logging.getLogger(__name__)


class SelectedTextUnit(TypedDict):
    id: str
    short_id: str
    entity_score: float
    relationship_score: int
    text_unit: str


def compute_relationship_score(
    df_relationships: pd.DataFrame,
    df_text_relationships: pd.DataFrame,
    entity_title: str,
) -> int:
    relationships_subset = df_relationships[
        df_relationships["id"].isin(df_text_relationships)
    ]

    source_count = (relationships_subset["source"] == entity_title).sum()
    target_count = (relationships_subset["target"] == entity_title).sum()

    return source_count + target_count


class TextUnitsSelector:
    def run(
        self,
        df_entities: pd.DataFrame,
        df_relationships: pd.DataFrame,
        df_text_units: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the TextUnit context for the LocalSearch algorithm."""
        selected_text_units: dict[str, SelectedTextUnit] = {}

        def _process_text_unit_id(text_unit_id: str) -> SelectedTextUnit:
            df_texts_units_subset = df_text_units[df_text_units["id"] == text_unit_id]
            text_relationship_ids = df_texts_units_subset["relationship_ids"].explode()

            relationship_score = compute_relationship_score(
                df_relationships,
                text_relationship_ids,
                entity.title,
            )

            text_unit = df_texts_units_subset["text_unit"].iloc[0]
            short_id = df_texts_units_subset.index.to_numpy()[0]

            return SelectedTextUnit(
                id=text_unit_id,
                short_id=short_id,
                entity_score=entity.score,
                relationship_score=relationship_score,
                text_unit=text_unit,
            )

        def _process_entity(entity) -> None:  # noqa: ANN001
            for text_unit_id in entity.text_unit_ids:
                if text_unit_id in selected_text_units:
                    continue
                selected_text_units[text_unit_id] = _process_text_unit_id(text_unit_id)

        for entity in df_entities.itertuples():
            _process_entity(entity)

        df_selected_text_units = pd.DataFrame.from_records(
            list(selected_text_units.values())
        )

        # sort it by
        # descending order of entity_score
        # and then descending order of relationship_score
        df_selected_text_units = df_selected_text_units.sort_values(
            by=["entity_score", "relationship_score"],
            ascending=[False, False],
        ).reset_index(drop=True)

        if _LOGGER.isEnabledFor(logging.DEBUG):
            import tableprint

            tableprint.banner("Selected Text units")
            tableprint.dataframe(
                df_selected_text_units[["id", "entity_score", "relationship_score"]]
            )

        return df_selected_text_units
