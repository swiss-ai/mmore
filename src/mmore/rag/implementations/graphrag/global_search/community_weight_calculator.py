"""Compute the weight of the community."""

import pandas as pd

from mmore.types.graphrag.graphs.community import CommunityId


class CommunityWeightCalculator:
    def __init__(self, *, should_normalize: bool = True):
        self._should_normalize = should_normalize

    def __call__(
        self,
        df_entities: pd.DataFrame,
        df_reports: pd.DataFrame,
    ) -> dict[CommunityId, float]:
        result: dict[CommunityId, float] = {}

        for _, row in df_reports.iterrows():
            entities = row["entities"]
            # get rows from entities dataframe where ids are in entities
            df_entities_filtered = df_entities[df_entities["id"].isin(entities)]
            # get the text_units from df_entities_filtered
            text_units = df_entities_filtered["text_unit_ids"].explode().unique()
            result[row["community_id"]] = len(text_units)

        if self._should_normalize:
            max_weight = max(result.values())
            for community_id in result:
                result[community_id] = result[community_id] / max_weight

        return result
