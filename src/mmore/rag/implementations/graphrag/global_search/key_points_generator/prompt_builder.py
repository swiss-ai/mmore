from pathlib import Path

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

from mmore.types.graphrag.prompts import PromptBuilder

from ._output_parser import KeyPointsOutputParser
from ._system_prompt import MILOU_PROMPT


class KeyPointsGeneratorPromptBuilder(PromptBuilder):
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
            self._system_prompt = MILOU_PROMPT
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
            template_format="mustache",
            partial_variables=dict(
                show_references=self._show_references,
                repeat_instructions=self._repeat_instructions,
            ),
        )

        template = ChatPromptTemplate(
            [system_template, ("user", "{{global_query}}")],
            template_format="mustache",
        )
        return template, KeyPointsOutputParser()
