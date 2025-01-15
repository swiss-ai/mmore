from typing import Any, Protocol

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from typing_extensions import Unpack


class PromptBuilder(Protocol):
    def build(self) -> tuple[BasePromptTemplate, BaseOutputParser]: ...


class IndexingPromptBuilder(PromptBuilder, Protocol):
    def prepare_chain_input(
        self, **kwargs: Unpack[dict[str, Any]]
    ) -> dict[str, str]: ...
