from typing import Dict

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from ..rag.llm import LLM, LLMConfig


class WebsearchOnly:
    """Class dedicated to performing web searches and validating their usefulness."""

    def __init__(self, region: str = "wt-wt", max_results: int = 10):
        """Initialize the WebsearchOnly class with search parameters."""
        self.wrapper = DuckDuckGoSearchAPIWrapper(
            region=region, max_results=max_results
        )

    def websearch_pipeline(self, query: str) -> Dict[str, str]:
        """Perform a single web search."""
        search = DuckDuckGoSearchResults(api_wrapper=self.wrapper)
        web_output = search.run(query)
        return web_output

    def summarize_web_search(self, query: str, web_output: str) -> str:
        """Call LLM to summarize the current web output based on the original query, return a summary of the web search and the source."""
        llm = LLM.from_config(
            LLMConfig(llm_name="OpenMeditron/meditron3-8b", max_new_tokens=1200)
        )
        prompt = (
            f"Original Query: '{query}'\n"
            f"Web content: '{web_output}'\n"
            "Based on the original query and the web content, can you provide a response to the original query?"
        )
        response = llm.invoke(prompt).content
        assert isinstance(response, str)
        return response.strip()
