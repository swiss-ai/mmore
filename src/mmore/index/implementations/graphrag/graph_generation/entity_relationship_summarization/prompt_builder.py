from pathlib import Path
from typing import Any

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from typing_extensions import Unpack

from mmore.types.graphrag.prompts import IndexingPromptBuilder

from ._default_prompts import DEFAULT_PROMPT, MILOU_PROMPT


class SummarizeDescriptionPromptBuilder(IndexingPromptBuilder):
    def __init__(
        self,
        *,
        prompt: str | None = None,
        prompt_path: Path | None = None,
    ):
        self._prompt: str | None
        if prompt is None and prompt_path is None:
            self._prompt = MILOU_PROMPT
        else:
            self._prompt = prompt

        self._prompt_path = prompt_path

    def build(self) -> tuple[BasePromptTemplate, BaseOutputParser]:
        if self._prompt:
            prompt_template = PromptTemplate.from_template(self._prompt)
        else:
            assert self._prompt_path is not None
            prompt_template = PromptTemplate.from_file(self._prompt_path)

        return prompt_template, StrOutputParser()

    def prepare_chain_input(self, **kwargs: Unpack[dict[str, Any]]) -> dict[str, str]:
        entity_name = kwargs.get("entity_name", None)
        description_list = kwargs.get("description_list", None)
        if entity_name is None:
            raise ValueError("entity_name is required")
        if description_list is None:
            raise ValueError("description_list is required")

        return dict(description_list=description_list, entity_name=entity_name)
