from pathlib import Path

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

from mmore.types.graphrag.prompts import PromptBuilder

from ._system_prompt import REDUCE_SYSTEM_PROMPT


class KeyPointsAggregatorPromptBuilder(PromptBuilder):
    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        system_prompt_path: Path | None = None,
        show_references: bool = True,
        repeat_instructions: bool = True,
    ):
        self._system_prompt: str | None
        if system_prompt is None and system_prompt_path is None:
            self._system_prompt = REDUCE_SYSTEM_PROMPT
        else:
            self._system_prompt = system_prompt

        self._system_prompt_path = system_prompt_path
        self._show_references = show_references
        self._repeat_instructions = repeat_instructions

    def build(self) -> tuple[BasePromptTemplate, BaseOutputParser]:
        if self._system_prompt_path:
            prompt = Path.read_text(self._system_prompt_path)
        else:
            assert self._system_prompt is not None
            prompt = self._system_prompt

        system_template = SystemMessagePromptTemplate.from_template(
            prompt,
            partial_variables=dict(
                response_type="Multiple Paragraphs",
                show_references=self._show_references,
                repeat_instructions=self._repeat_instructions,
            ),
            template_format="mustache",
        )

        template = ChatPromptTemplate(
            [system_template, ("user", "{{global_query}}")],
            template_format="mustache",
        )
        return template, StrOutputParser()
