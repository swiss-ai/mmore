import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, SystemMessage

from .llm import LLM
from .config import WebsearchConfig


class WebsearchPipeline:
    """Pipeline for enhancing RAG outputs with web search."""

    def __init__(self, config: WebsearchConfig):
        """Initialize pipeline."""
        self.config = config
        self.llm = LLM.from_config(config.get_llm_config())

    @staticmethod
    def clean_llm_output(text: str) -> str:
        """Remove internal model tokens."""
        return re.sub(r'<\|.*?\|>', '', text).strip()

    @staticmethod
    def extract_llm_answer(raw_response: str) -> str:
        """Extract the answer following the 'Answer:' prefix from the LLM response."""
        raw_response = WebsearchPipeline.clean_llm_output(raw_response)
        match = re.search(r'Answer:(.*)', raw_response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return raw_response

    def generate_summary(self, rag_answer: str, query: str) -> str:
        """Build the prompt, invoke the LLM, and extract the answer."""
        prompt = (
            "You have only the following context to answer the question—do not use any external knowledge.\n\n"
            f"Question: {query}\n\n"
            "Context:\n"
            f"{rag_answer}\n\n"
            "If the context contains the answer or useful information, respond with that information. "
            "Answer:"
        )

        messages = [
            SystemMessage(content="You are a helpful assistant that summarizes text relevant to the question."),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return self.extract_llm_answer(response.content)

    @staticmethod
    def extract_query_from_llm_output(text: str) -> str:
        """Extract everything between <question> tags, picking longest match."""
        matches = re.findall(r'<\s*question\s*>(.*?)<\s*/\s*question\s*>', text, re.DOTALL | re.IGNORECASE)
        return max(matches, key=len).strip() if matches else ""

    @staticmethod
    def validate_search_query(query: str, original_query: str) -> str:
        """Fallback to default if empty or >30 words."""
        return query if query and len(query.split()) <= 30 else original_query

    @staticmethod
    def duckduckgo_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Perform DuckDuckGo search."""
        try:
            with DDGS() as ddgs:
                return [
                    {'title': r.get('title'), 'url': r.get('href')}
                    for r in ddgs.text(query, max_results=max_results)
                ]
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def generate_search_query(self, original_query: str, rag_answer: str, previous_analysis: Optional[str] = None) -> str:
        """Generate and extract query enclosed in <question> tags."""
        base = f"Based on:\n- Original Query: '{original_query}'\n- RAG Answer: '{rag_answer}'"
        if previous_analysis:
            base += f"\n- Previous Findings: {previous_analysis}"
        prompt = (
            f"{base}\n"
            "Generate a concise search query (up to 30 words) that either directly answers the original question "
            "or complements previous findings by seeking missing or updated information. "
            "Enclose your answer within <question></question> tags."
        )
        messages = [SystemMessage(content="You are a search query generator."), HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)

        raw = self.clean_llm_output(response.content)
        extracted = self.extract_query_from_llm_output(raw)
        return self.validate_search_query(' '.join(extracted.split()), original_query)

    def analyze_search_results(self, original_query: str, rag_answer: str, results: List[Dict[str, str]]) -> str:
        """Analyze search results and create a combined summary with RAG and web information."""
        collated = '\n\n'.join([f"Source: {r['title']}\n{r.get('body', '')}" for r in results])
        prompt = (
            f"Original Query: {original_query}\n"
            f"Current Knowledge (RAG): {rag_answer}\n"
            "New Information from Web:\n" f"{collated}\n\n"
            "Provide a detailed analysis that combines the RAG knowledge with the new web information. "
            "Your response should:\n"
            "1. Integrate both RAG and web information comprehensively\n"
            "2. Include specific details, facts, and findings\n"
            "3. Highlight important updates or corrections from the web sources\n"
            "Structure your response as follows in the tags <enhanced_answer><enhanced_answer>:\n"
            "- Summary: A summary of all key points from the RAG and the web sources responding directly to the query\n"
            "- More detailed informations: Any additional useful information more detailed\n"
            "and outside of the tags:\n"
            "ADDITIONAL GAPS:\n"
            "- [List any remaining questions or areas needing more research in order to improve the answer of the original query]"
        )
        messages = [
            SystemMessage(content="You are a research analyst focused on providing detailed, comprehensive analysis."),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return self.clean_llm_output(response.content)

    @staticmethod
    def extract_enhanced_answer(text: str) -> Optional[Dict[str, str]]:
        """Extract the content between enhanced_answer tags, focusing on the last occurrence."""
        try:
            pattern = r'<enhanced_answer>\s*(.*?)\s*</enhanced_answer>'
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

            if not matches:
                # fallback extraction if tags missing
                pattern_sum = r'Summary:(.*?)(?:More detailed informations:|ADDITIONAL GAPS:|$)'
                summary_matches = re.findall(pattern_sum, text, re.DOTALL | re.IGNORECASE)

                pattern_det = r'More detailed informations:(.*?)(?:ADDITIONAL GAPS:|$)'
                details_matches = re.findall(pattern_det, text, re.DOTALL | re.IGNORECASE)

                if summary_matches or details_matches:
                    return {
                        'summary': summary_matches[-1].strip() if summary_matches else "",
                        'details': details_matches[-1].strip() if details_matches else ""
                    }
                return None

            extracted_content = matches[-1].strip()

            summary_lines = []
            details_lines = []
            current_section = None

            for line in extracted_content.splitlines():
                line = line.strip()
                if not line or line.startswith('Source:') or line.startswith('New Information'):
                    continue

                if line.lower().startswith('summary:'):
                    current_section = 'summary'
                    continue
                elif line.lower().startswith('more detailed informations'):
                    current_section = 'details'
                    continue

                if current_section == 'summary':
                    summary_lines.append(line)
                elif current_section == 'details':
                    details_lines.append(line)

            return {
                'summary': '\n'.join(summary_lines).strip(),
                'details': '\n'.join(details_lines).strip()
            }
        except Exception as e:
            print(f"Warning: Error during extraction: {e}")
            return None

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single JSON record with web search enhancement."""
        user_query = record.get('input', '')
        initial_rag_answer = record.get('answer', '')

        # Generate initial summary
        initial_summary = self.generate_summary(initial_rag_answer, user_query)

        previous_analysis = None
        all_sources: List[Dict[str, str]] = []
        seen_urls: Set[str] = set()
        current_knowledge = initial_summary
        final_answer = None

        for i in range(1, self.config.n_loops + 1):
            print(f"Loop {i} of {self.config.n_loops}")
            query = self.generate_search_query(user_query, current_knowledge, previous_analysis)
            print(f"Generated query: {query}, based on the original query: {user_query}")

            results = self.duckduckgo_search(query, self.config.max_searches)

            # Add only new sources (de-dup by url)
            for r in results:
                if r['url'] not in seen_urls:
                    all_sources.append({'title': r['title'], 'url': r['url']})
                    seen_urls.add(r['url'])

            analysis = self.analyze_search_results(user_query, current_knowledge, results)

            extracted_answer = self.extract_enhanced_answer(analysis)
            if extracted_answer:
                current_knowledge = extracted_answer.get('summary', '') + '\n' + extracted_answer.get('details', '')
                final_answer = extracted_answer
            else:
                # fallback to full analysis if no extraction
                current_knowledge = analysis

            previous_analysis = analysis

        return {
            'query': user_query,
            'RAG_summary': initial_summary,
            'WEB_RAG_summary': final_answer.get('summary', '') if final_answer else "",
            'WEBSEARCH_details': final_answer.get('details', '') if final_answer else "",
            'sources': all_sources,
        }

    def run(self):
        """Run the websearch pipeline."""
        with open(self.config.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = [self.process_record(record) for record in data]

        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {output_path}")
