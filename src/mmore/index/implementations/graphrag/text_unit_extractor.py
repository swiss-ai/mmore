"""A module to extract text units from the document."""

import uuid
from typing import TypedDict

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from tqdm import tqdm


class _TextUnit(TypedDict):
    id: str
    document_id: str
    text_unit: str


class TextUnitExtractor:
    def __init__(self, text_splitter: TextSplitter):
        self._text_splitter = text_splitter

    def run(self, documents: list[Document]) -> pd.DataFrame:
        response: list[_TextUnit] = []

        # TODO: Parallize this
        for document in tqdm(documents, desc="Processing documents ..."):
            text_units = self._text_splitter.split_text(document.page_content)

            document_id = document.id if document.id else str(uuid.uuid4())

            for t in tqdm(text_units, desc="Extracting text units ..."):
                response.append(  # noqa: PERF401
                    _TextUnit(
                        document_id=document_id,
                        id=str(uuid.uuid4()),
                        text_unit=t,
                    )
                )

        return pd.DataFrame.from_records(response)
