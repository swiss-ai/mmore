"""Default PromptBuilder for Entity Relationship Extraction."""

from pathlib import Path
from typing import Any

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from typing_extensions import Unpack

from mmore.types.graphrag.prompts import IndexingPromptBuilder

from ._default_prompts import MED_ER_EXTRACTION_PROMPT as DEFAULT_ER_EXTRACTION_PROMPT
from ._output_parser import EntityExtractionOutputParser

_DEFAULT_TUPLE_DELIMITER = "<|>"
_DEFAULT_RECORD_DELIMITER = "##"
_DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
_DEFAULT_ENTITY_TYPES = ["disease", "symptom", "cause", "test"]




class EntityExtractionPromptBuilder(IndexingPromptBuilder):
    """PromptBuilder for Entity Relationship extraction.

    This implementation assumes that the prompt is a template string with the following placeholders:

    - entity_types: A comma-separated list of entity types. Default is "organization,person,geo,event".

    - tuple_delimiter: The delimiter for tuples. Default is "<|>".

    - record_delimiter: The delimiter for records. Default is "##".

    - completion_delimiter: The delimiter for completions. Default is "<|COMPLETE|>".


    There is a default template embedded in the package which is same
    as that of the official implementation.

    You can supply your own prompt as long as you keep the placeholders and as
    have examples specified in the same output format as the default prompt.

    If you want a prompt with different placeholders or output formats for
    entity extraction, you can create a custom implementation of the protocol
    `PromptBuilder` and use it in the `EntityRelationshipExtractor` class.

    """

    def __init__(
        self,
        *,
        prompt: str | None = None,
        prompt_path: Path | None = None,
        entity_types: list[str] = _DEFAULT_ENTITY_TYPES,
        tuple_delimiter: str = _DEFAULT_TUPLE_DELIMITER,
        record_delimiter: str = _DEFAULT_RECORD_DELIMITER,
        completion_delimiter: str = _DEFAULT_COMPLETION_DELIMITER,
    ):
        """Initializes the PromptBuilder object.

        If neither prompt nor prompt_path is provided, the default prompt
        provided with in the package (same as official implementation)  will be used.

        Args:
            prompt (str | None, optional): The prompt string.
            prompt_path (Path | None, optional): The path to the prompt file.
            entity_types (list[str], optional): The list of entity types.
            tuple_delimiter (str, optional): The delimiter for tuples.
            record_delimiter (str, optional): The delimiter for records.
            completion_delimiter (str, optional): The delimiter for completions.
        """  # noqa: D202

        self._prompt: str | None
        if prompt is None and prompt_path is None:
            self._prompt = DEFAULT_ER_EXTRACTION_PROMPT
        else:
            self._prompt = prompt

        self._prompt_path = prompt_path

        self._entity_types = entity_types
        self._tuple_delimiter = tuple_delimiter
        self._record_delimiter = record_delimiter
        self._completion_delimiter = completion_delimiter

    def build(self) -> tuple[BasePromptTemplate, BaseOutputParser]:
        """Build the template and output parser.

        Note:
            You would not directly use this method. It is used by the
            `EntityRelationshipExtractor` class.

        Returns:
            A tuple containing the built `BasePromptTemplate` object
            and the `EntityExtractionOutputParser` object.
        """
        if self._prompt:
            prompt_template = PromptTemplate.from_template(self._prompt)
        else:
            assert self._prompt_path is not None
            prompt_template = PromptTemplate.from_file(self._prompt_path)

        return (
            prompt_template.partial(
                completion_delimiter=self._completion_delimiter,
                tuple_delimiter=self._tuple_delimiter,
                record_delimiter=self._record_delimiter,
                entity_types=",".join(self._entity_types),
            ),
            EntityExtractionOutputParser(
                tuple_delimiter=self._tuple_delimiter,
                record_delimiter=self._record_delimiter,
            ),
        )

    def prepare_chain_input(self, **kwargs: Unpack[dict[str, Any]]) -> dict[str, str]:  # noqa: D417
        """Prepares the input for the extraction chain.

        Note:
            You would not directly use this method.
            It is used by the `EntityRelationshipExtractor` class.

        Args:
            text_unit: The text unit from which entities and relationships are extracted.
        """
        text_unit: str = kwargs.get("text_unit", None)
        if text_unit is None:
            raise ValueError("text_unit is required")

        return dict(input_text=text_unit)
