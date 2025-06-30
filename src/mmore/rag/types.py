from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel, Field

# ------------------------------- Simple Input ------------------------------- #


class MMOREInput(BaseModel):
    """Input for the chat endpoint."""

    input: str = Field(
        ...,
        description="The user input.",
    )
    collection_name: str = Field(
        ...,
        description="The collection",
    )
    document_ids: List[str] = Field(
        default_factory=list,  # Set default to an empty list if not provided
        description="List of document IDs, defaults to an empty list if not provided.",
    )


# ------------------------------- Simple Output ------------------------------ #


class MMOREOutput(BaseModel):
    """Base Answer, outputs the query, documents and answer"""

    input: str
    docs: List[Document]
    answer: str


# -------------------------------- CitedAnswer ------------------------------- #


class CitedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer.",
    )


# ------------------------------- QuotedAnswer ------------------------------- #


class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )
