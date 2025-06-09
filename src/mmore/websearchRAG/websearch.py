from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

class WebsearchOnly:
    """Class dedicated to performing web searches and validating their usefulness."""

    def __init__(self, region: str = 'wt-wt', max_results: int = 10):
        """Initialize the WebsearchOnly class with search parameters."""
        self.wrapper = DuckDuckGoSearchAPIWrapper(region=region, max_results=max_results)

    def websearch_pipeline(self, query: str) -> Dict[str, str]:
        """Perform a single web search."""
        search = DuckDuckGoSearchResults(api_wrapper=self.wrapper)
        web_output = search.run(query)
        return web_output



    def multiple_queries(self, original_query: str, list_of_queries: List[str], keep_intact: bool) -> List[Dict[str, str]]:
        """Perform multiple web searches in parallel for a list of queries."""
        results = []
        with ThreadPoolExecutor() as executor:
            future_to_query = {executor.submit(self.websearch_pipeline, query): query for query in list_of_queries}
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result()
                    if keep_intact:
                        to_keep = self.check_source_of_web_search(original_query, result)
                        if to_keep:
                            results.append({'query': query, 'result': result, 'summary' : None})
                    else:
                        summary = self.resume_web_search(original_query, result)
                        results.append({'query': query, 'result': result, 'summary': summary})
                except Exception as e:
                    results.append({'query': query, 'error': str(e)})
        
        return results



    def check_source_of_web_search(self, query: str, web_output: str) -> bool:
        """Call LLM to determine if the current web output is useful based on the original query."""
        llm = LLM()  # TODO: Implement LLM
        prompt = (
            f"Original Query: '{query}'\n"
            f"Web Output: '{web_output}'\n"
            "Is the web output useful for the original query? Answer with 'True' or 'False'."
        )
        response = llm.invoke(prompt)
        return response.strip().lower() == 'true'

    def resume_web_search(self, query: str, web_output: str) -> str:
        """Call LLM to resume the current web output based on the original query, return a summary of the web search and the source."""
        llm = LLM()  # TODO: Implement LLM
        prompt = (
            f"Original Query: '{query}'\n"
            f"Web content: '{web_output}'\n"
            "Based on the original query and the web content, can you provide a response to the original query?"
        )
        response = llm.invoke(prompt)
        return response.strip() ### RAJOUTER LA SOURCE

# Example usage:
# websearch = WebsearchOnly()
# results = websearch.multiple_queries("original_query", ["query1", "query2", "query3"], keep_intact=True)
# for res in results:
#     print(res)
