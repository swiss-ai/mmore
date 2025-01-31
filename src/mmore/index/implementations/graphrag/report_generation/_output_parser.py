from typing import Any

from langchain.output_parsers import PydanticOutputParser

from .utils import CommunityReportResult


class CommunityReportOutputParser(PydanticOutputParser):
    def __init__(self, **kwargs: dict[str, Any]):
        super().__init__(pydantic_object=CommunityReportResult, **kwargs)
