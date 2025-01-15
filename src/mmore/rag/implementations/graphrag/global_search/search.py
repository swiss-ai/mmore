import logging
from typing import Iterator

from langchain_core.runnables import RunnableConfig

from .key_points_aggregator import KeyPointsAggregator
from .key_points_generator import KeyPointsGenerator
from .key_points_generator.utils import (
    KeyPointsResult,
)

_LOGGER = logging.getLogger(__name__)


class GlobalSearch:
    def __init__(
        self,
        kp_generator: KeyPointsGenerator,
        kp_aggregator: KeyPointsAggregator,
        *,
        generation_chain_config: RunnableConfig | None = None,
        aggregation_chain_config: RunnableConfig | None = None,
    ):
        self._kp_generator = kp_generator
        self._kp_aggregator = kp_aggregator
        self._generation_chain_config = generation_chain_config
        self._aggregation_chain_config = aggregation_chain_config

    def _get_key_points(self, query: str) -> dict[str, KeyPointsResult]:
        generation_chain = self._kp_generator()
        response = generation_chain.invoke(
            query,
            config=self._generation_chain_config,
        )

        if _LOGGER.getEffectiveLevel() == logging.INFO:
            for k, v in response.items():
                _LOGGER.info(f"{k} - {len(v.points)}")

        return response

    def invoke(self, query: str) -> str:
        aggregation_chain = self._kp_aggregator()
        response = self._get_key_points(query)

        return aggregation_chain.invoke(
            input=dict(report_data=response, global_query=query),
            config=self._aggregation_chain_config,
        )

    def stream(self, query: str) -> Iterator:
        aggregation_chain = self._kp_aggregator()
        response = self._get_key_points(query)

        return aggregation_chain.stream(
            input=dict(report_data=response, global_query=query),
            config=self._aggregation_chain_config,
        )
