from typing import Any

from langchain.output_parsers import PydanticOutputParser

from .utils import KeyPointsResult


class KeyPointsOutputParser(PydanticOutputParser):
    def __init__(self, **kwargs: dict[str, Any]):
        super().__init__(pydantic_object=KeyPointsResult, **kwargs)
