"""
PRF Module abstract class for Query Expansion in RAG Retriever.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from src.mmore.utils import load_config
from src.mmore.rag.llm import LLM, LLMConfig
from src.mmore.rag.retriever import Retriever, RetrieverConfig


import logging
logger = logging.getLogger(__name__)

class PRF(BaseQueryExpansion):
    """Handles Pseudo Relevance Feedback for RAG retriever."""
    LLM: BaseChatModel

    def __init__(self, PRFConfig: PRFConfig, retrieverConfig: RetrieverConfig):
        super().__init__(PRFConfig)
        self.LLM = LLM.from_config(PRFConfig.llm)
        self.retriever = Retriever.from_config(retrieverConfig)

    def expand_query(self, query: str, collection_name: str, partition_name: str) -> str:
        """
        Expand the query with similar terms.
        It retrieves the top 3 documents from the initial query and prompts an LLM.

        Args:
            query (str): The original query.

        Returns:
            str: The expanded query.
        """
        # Retrieve the top 3 documents
        results = self.retriever.retrieve(
            query=query,
            collection_name=collection_name,  
            partition_names=partition_names,       
            k=3                         
        )

        # Extract the text content from the results
        top_docs_text = [
            result["entity"]["text"] for result in results[0]
        ]

        # Generate the prompt
        prompt = (
            "Write a passage that answers the given query based on the context:\n"
            f"Context:\n{top_docs_text[0]}\n{top_docs_text[1]}\n{top_docs_text[2]}\n"
            f"Query: {query}\n"
            "Passage:"
        )
        
        # Generate the expanded query
        response = self.LLM.generate(prompt, max_length=100)

        # Repeat the original query 5 times before appending the response
        expanded_query = f"{query} " * 5 + response.strip()

        return expanded_query