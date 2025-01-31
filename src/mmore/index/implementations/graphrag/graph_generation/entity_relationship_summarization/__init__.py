"""Entity Relationship Description Summarization Module."""

from .prompt_builder import SummarizeDescriptionPromptBuilder
from .summarizer import EntityRelationshipDescriptionSummarizer

__all__ = [
    "SummarizeDescriptionPromptBuilder",
    "EntityRelationshipDescriptionSummarizer",
]
