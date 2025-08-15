import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..rag.llm import LLM, LLMConfig
from ..run_rag import rag
from .config import WebsearchConfig
from .logging_config import logger


@dataclass
class ProcessedResponse:
    query: str
    rag_informations: str
    rag_summary: str | None
    web_summary: str
    short_answer: str
    detailed_answer: str
    sources: Dict[str, Any]  # Maps URLs to lists of titles


def extract_response(content: str | list[str | dict]) -> str:
    response_content = content
    if isinstance(response_content, str):
        response = response_content
    else:
        response_tmp = response_content[-1]
        response_tmp: str | dict[str, str]

        if isinstance(response_tmp, str):
            response = response_tmp
        else:
            response = response_tmp.get("content", "")

    return response


class WebsearchPipeline:
    """
    Pipeline for running RAG and iterative websearch loops,
    integrating retrieved knowledge into enhanced answers.
    """

    def __init__(self, config: WebsearchConfig):
        self.config = config
        self.llm = self._initialize_llm()
        self.rag_results = None
        self.wrapper = DuckDuckGoSearchAPIWrapper(max_results=self.config.max_searches)
        self.search = DuckDuckGoSearchResults(
            api_wrapper=self.wrapper, output_format="list"
        )

    def _initialize_llm(self) -> BaseChatModel:
        if self.config.use_rag:
            rag_cfg = self.config.access_rag_config()
            llm_conf: Dict[str, Any] = rag_cfg.get("rag", {}).get("llm")
            if llm_conf is None:
                raise ValueError(
                    "Missing 'llm' config under 'rag' in RAG configuration."
                )
            return LLM.from_config(LLMConfig(**llm_conf))
        else:
            base_conf = self.config.get_llm_config()
            base_conf = base_conf.__dict__
            return LLM.from_config(LLMConfig(**base_conf))

    def generate_summary(self, rag_answer: str | None, query: str):
        """
        Summarize the RAG answer (used when rag_summary=True)
        """
        prompt = (
            "You have only the following context to answer the question, do not use any external knowledge.\n\n"
            f"Question: {query}\n\n"
            "Context:\n"
            f"{rag_answer or 'No context yet'}\n\n"
            "If the context contains the answer or any useful information, respond with that information. \n"
            "If no useful informations are, answer: no useful informations\n"
            "Answer: \n"
            "---------------------------"
        )

        messages = [
            SystemMessage(
                content="You are a helpful assistant that summarizes text relevant to the question."
            ),
            HumanMessage(content=prompt),
        ]

        response_llm = self.llm.invoke(messages)
        response = extract_response(response_llm.content)

        return self._clean_llm_output(response)

    def evaluate_subquery_relevance(
        self, query, current_subqueries, previous_subqueries
    ):
        prompt = (
            f"Original query:\n{query}\n\n"
            f"Previous subqueries that contribute to understanding:\n{previous_subqueries}\n\n"
            f"New subqueries:\n{current_subqueries}\n\n"
            "Are any of the new subqueries relevant in the context of the original query and previous subqueries? "
            "Respond strictly with 'yes' if at least one is relevant, or 'no' if none are."
        )
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content=prompt),
        ]
        response_llm = self.llm.invoke(messages)
        response_content = extract_response(response_llm.content)
        response = self._clean_llm_output(response_content)

        if "no" in response:
            return False
        else:
            return True

    def _clean_llm_output(self, content: str):
        delimiter = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        if delimiter not in content:
            return content

        # Extract the section after the delimiter
        cleaned_section = content.split(delimiter, 1)[-1].lower().strip()

        return cleaned_section

    def generate_subqueries(
        self, original_query: str, current_context: Optional[str] = None
    ) -> List[str]:
        """
        Generate concise search subqueries
        """
        n = self.config.n_subqueries
        instruction = f"You have the question and partial answer below:\nQuestion: {original_query}\n\n"
        if current_context is None:
            task = (
                f"Generate {n}-independant subqueries based on the original query, in order to generate the most complete research. Each subquery must be concise and ≤30 words.\n"
                f"The subqueries should print in this format: subquery 1: new question,  subquery 2: new question, etc. \n"
            )
        else:
            task = (
                f"Partial answer: {current_context}\n\n"
                f"Generate {n}-independant subqueries to refine the answer based on the original query. Each subquery must be concise and ≤30 words.\n"
                f"The subqueries should print in this format: subquery 1: new question,  subquery 2: new question, etc. \n"
                f"---ANSWER ---"
            )

        prompt = instruction + task
        messages = [
            SystemMessage(
                content="You are an assistant specializing in generating search queries."
            ),
            HumanMessage(content=prompt),
        ]

        response_llm = self.llm.invoke(messages)
        response = extract_response(response_llm.content)
        cleaned_answer = self._clean_llm_output(response)
        cleaned_answer = re.findall(r"subquery \d+: (.*)", cleaned_answer)
        return cleaned_answer

    def duckduckgo_search(self, query: str) -> List[Dict[str, str]]:
        """
        Perform a DuckDuckGo search using LangChain DuckDuckGo wrapper

        Returns a list of dicts with keys: 'title' and 'url'
        """
        try:
            results = self.search.invoke(query)

            formatted_results = []
            for r in results:
                snippet = r.get("snippet", "")
                url = r.get("link", "")  # note: it's "link" in LangChain results
                title = r.get("title", "")

                formatted_results.append(
                    {"snippet": snippet, "url": url, "title": title}
                )

            return formatted_results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    def integrate_with_llm(
        self, original: str, rag_doc: str | None, web_snippets: List[str]
    ) -> Dict[str, str]:
        # Build prompt for short & detailed answer
        sources = "\n".join(web_snippets)
        prompt = (
            f"Original Query: {original}\n"
            f"RAG Document Information:\n{rag_doc}\n\n"
            f"Web Information:\n{sources}\n\n"
            "Provide the response in the following format:\n"
            "short answer: <your concise answer>\n"
            "detailed answer: <your detailed answer>"
        )

        msgs = [
            SystemMessage(content="You are a research assistant."),
            HumanMessage(content=prompt),
        ]
        response_llm = self.llm.invoke(msgs)
        response = extract_response(response_llm.content)
        # parse
        clean_content = self._clean_llm_output(response)

        sa_matches = re.findall(
            r"short answer:\s*(.*?)(?=detailed answer:)",
            clean_content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        da_matches = re.findall(
            r"detailed answer:\s*(.*)", clean_content, flags=re.IGNORECASE | re.DOTALL
        )

        short = sa_matches[-1].strip().rstrip(",") if sa_matches else ""
        detailed = da_matches[-1].strip() if da_matches else ""
        return {"short": short, "detailed": detailed}

    def process_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        qr = rec.get("input", "").strip()
        rag_ans = rec.get("answer", "") if self.config.use_rag else ""
        self.rag_results = rag_ans
        rag_summary = (
            self.generate_summary(rag_ans, qr) if self.config.use_rag else None
        )

        source_map = {}
        current_context = rag_summary
        final_short, final_detailed = "", ""
        web_summary = ""
        web_summary_all = ""  # will be reassigned later

        web_summaries = []
        previous_sub = []

        for loop in range(self.config.n_loops):
            if self.config.use_rag:
                subs = self.generate_subqueries(qr, current_context)
            else:
                subs = self.generate_subqueries(qr)  # Based on original query only

            snippets = []
            subquery_summaries = []

            if loop > 0 and not self.evaluate_subquery_relevance(
                qr, subs, previous_sub
            ):
                break

            for sq in subs:
                time.sleep(10)
                res = self.duckduckgo_search(query=sq)

                subquery_snippets = []

                for r in res:
                    if r["url"] not in source_map:
                        source_map[r["url"]] = []

                    if r["title"] not in source_map[r["url"]]:
                        source_map[r["url"]].append(r["title"])

                    snippet = f"{r['snippet']})"
                    snippets.append(snippet)
                    subquery_snippets.append(snippet)

                    combined_snippets = "\n".join(subquery_snippets)

                    summary = self.generate_summary(combined_snippets, sq)
                    subquery_summaries.append(summary)

            previous_sub = subs

            combined_sub_summaries = "\n".join(
                [str(s) if s else "" for s in subquery_summaries]
            )
            web_summary = self.generate_summary(combined_sub_summaries, qr)
            web_summaries.append(web_summary)

            if self.config.use_summary:
                # Combine rag summary, web summary, and original query for final answer
                context_for_llm = f"RAG informations:\n{rag_summary or ''}\n\nWeb informations:\n{web_summary}"
            else:
                # If not summarizing subqueries, use rag summary or current context with snippets
                context_for_llm = current_context

            combined_web_summaries = "\n".join(
                [str(s) if s else "" for s in web_summaries]
            )
            web_summary_all = self.generate_summary(combined_web_summaries, qr)

            # Current context, web content  to generate the answer
            out = self.integrate_with_llm(qr, context_for_llm, snippets)
            final_short, final_detailed = out["short"], out["detailed"]

            # Prepare context for next search loop
            current_context = final_detailed

        solution = ProcessedResponse(
            query=qr,
            rag_informations=self.rag_results,
            rag_summary=rag_summary if self.config.use_rag else None,
            web_summary=web_summary_all,
            short_answer=final_short,
            detailed_answer=final_detailed,
            sources=source_map,
        )

        return asdict(solution)

    def run(self):
        # RAG pipeline
        if self.config.use_rag:
            if not self.config.rag_config_path:
                raise ValueError("rag_config_path required when use_rag=True")
            logger.info("Running RAG pipeline...")
            rag(self.config.rag_config_path)
            rc = self.config.access_rag_config()
            self.config.input_file = rc["mode_args"]["output_file"]

            assert self.config.input_file
            with open(self.config.input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            self.config.input_file = self.config.input_queries
            data = []

            assert self.config.input_file
            with open(self.config.input_file, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))  # JSONL format

        outputs = []
        outputs = [self.process_record(rec) for rec in data]

        # save
        outp = Path(self.config.output_file)
        outp.parent.mkdir(exist_ok=True, parents=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {outp}")

    def run_api(self, use_rag, use_summary, query):
        """
        Process queries and handle them with a temporary JSONL file.
        Parameters:
        - use_rag (bool): Indicates whether to use RAG.
        - use_summary (bool): Indicates whether to use summarization.
        - query (list): List of query dictionaries.
        Returns:
        - List of processed query results.
        """
        # Save query to a temporary JSONL file
        self.config.use_rag = use_rag
        self.config.use_summary = use_summary

        temp_file_path = self._save_query_as_json(query)

        try:
            outputs = []
            # Read from the temporary JSONL file
            with open(temp_file_path, "r", encoding="utf-8") as f:
                if self.config.use_rag:
                    for line in f:
                        record = json.loads(line)
                        outputs.append(self.process_record(record))
                else:
                    for line in f:
                        record = json.loads(line.strip())
                        outputs.append(self.process_record(record))

            return outputs

        finally:
            # Delete the temporary file
            logger.info(f"Deleting temporary file: {temp_file_path}")
            os.remove(temp_file_path)

    def _save_query_as_json(self, query):
        """Save query to a temporary JSONL file and return the file path."""
        suffix = ".json" if self.config.use_rag else ".jsonl"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False
        ) as temp_file:
            # Convert Pydantic models to dictionaries if needed
            if isinstance(query, list):
                temp_file.writelines(
                    json.dumps(q.dict() if hasattr(q, "dict") else q) + "\n"
                    for q in query
                )
            else:
                temp_file.write(
                    json.dumps(query.dict() if hasattr(query, "dict") else query) + "\n"
                )
            logger.info(f"Query saved to temporary file: {temp_file.name}")
            return temp_file.name
