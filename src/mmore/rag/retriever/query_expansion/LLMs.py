"""
LLMs Module abstract class for Query Expansion in RAG Retriever.
"""
from langchain_core.language_models.chat_models import BaseChatModel

from src.mmore.utils import load_config
from src.mmore.rag.llm import LLM, LLMConfig
from src.mmore.rag.retriever import Retriever, RetrieverConfig


import logging
logger = logging.getLogger(__name__)

class LLMs(BaseQueryExpansion):
    """Handles Language Model based Query Expansion for RAG retriever."""
    LLM: BaseChatModel

    def __init__(self, config: LLMsConfig):
        super().__init__(config)
        self.LLM = LLM.from_config(config.llm)

    def expand_query(self, query: str, collection_name: str, partition_name: str) -> str:
        """
        Expand the query with similar terms.
        We formulate the query expansion problem as follows: given a query
        q, we wish to generate an expanded query q' that contains additional
        query terms that may help in retrieving relevant documents. 
        We are using the CoT methodology from: https://arxiv.org/pdf/2305.03653

        Args:
            query (str): The original query.
        
        Returns:
            str: The expanded query.
        """
        # Generate the prompt
        prompt = (
            f"Answer the following query: {query}\n"
            "Provide the rationale before answering. Limit the response to less than 50 words."
        )
        
        # Generate the expanded query
        response = self.LLM.generate(prompt, max_length=100)
        
        # Repeat the original query 5 times before appending the response
        expanded_query = f"{query} " * 5 + response.strip()

        return expanded_query