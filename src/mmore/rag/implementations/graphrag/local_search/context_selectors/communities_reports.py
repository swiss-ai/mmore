"""Select the communities to be used in the local search."""

import logging

import pandas as pd

from mmore.types.graphrag.graphs.community import CommunityId, CommunityLevel

_LOGGER = logging.getLogger(__name__)


class CommunitiesReportsSelector:
    def __init__(
        self,
        community_level: CommunityLevel,
        *,
        must_have_selected_entities: bool = True,
    ):
        self._community_level = community_level
        self._must_have_selected_entities = must_have_selected_entities

    def run(
        self,
        df_entities: pd.DataFrame,
        df_reports: pd.DataFrame,
    ) -> pd.DataFrame:
        # Filter the communities based on the community level
        df_reports_filtered = df_reports[
            df_reports["level"] <= self._community_level
        ].copy(deep=True)

        # get the communities we have
        selected_communities = df_reports_filtered["community_id"].unique()

        # we will rank the communities based on the
        # number of selected entities that belong to a community
        community_to_entities_count: dict[CommunityId, int] = {}
        for entity in df_entities.itertuples():
            if entity.communities is None:
                continue
            for community in entity.communities:
                if community in selected_communities:
                    community_to_entities_count[community] = (
                        community_to_entities_count.get(community, 0) + 1
                    )

        df_reports_filtered["selected_entities_count"] = df_reports_filtered[
            "community_id"
        ].apply(lambda community_id: community_to_entities_count.get(community_id, 0))

        # sort the communities based on the number of selected entities
        # and rank of the community
        selected_reports = df_reports_filtered.sort_values(
            by=["selected_entities_count", "rating"],
            ascending=[False, False],
        ).reset_index(drop=True)

        if self._must_have_selected_entities:
            selected_reports = selected_reports[
                selected_reports["selected_entities_count"] > 0
            ]

        if _LOGGER.isEnabledFor(logging.DEBUG):
            import tableprint

            tableprint.banner("Selected Reports")
            tableprint.dataframe(
                selected_reports[
                    ["community_id", "level", "selected_entities_count", "rating"]
                ]
            )

        return selected_reports
