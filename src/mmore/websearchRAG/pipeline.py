from .logging_config import logger

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import logging
import time

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


from langchain_core.messages import HumanMessage, SystemMessage

from ..run_rag import rag
from ..run_rag import read_queries
from ..rag.llm import LLM, LLMConfig
from .config import WebsearchConfig



# python -m mmore websearch --config-file examples/websearchRAG/config.yaml



class WebsearchPipeline:
    """
    Pipeline for running RAG and iterative websearch loops,
    integrating retrieved knowledge into enhanced answers.
    """

    def __init__(self, config: WebsearchConfig):
        self.config = config
        self.llm = self._initialize_llm()
        self.rag_results: Optional[List[Dict[str, Any]]] = None

    def _initialize_llm(self) -> LLM:
        if self.config.use_rag is True:
            rag_cfg = self.config.access_rag_config()
            llm_conf = rag_cfg.get("rag", {}).get("llm")
            if llm_conf is None:
                raise ValueError("Missing 'llm' config under 'rag' in RAG configuration.")
            return LLM.from_config(LLMConfig(**llm_conf))
        elif self.config.use_rag is False:
            base_conf = self.config.get_llm_config()
            if isinstance(base_conf, LLMConfig):  # Ensure it's a dictionary
                base_conf = base_conf.__dict__
            return LLM.from_config(LLMConfig(**base_conf))
        else:
            raise ValueError("Invalid value for 'use_rag'. Must be True or False.")


    def generate_summary(self, rag_answer: str, query: str) -> str:
        """
        Summarize the RAG answer (used when rag_summary=True).
        """
        prompt = (
            "You have only the following context to answer the question, do not use any external knowledge.\n\n"
            f"Question: {query}\n\n"
            "Context:\n"
            f"{rag_answer}\n\n"
            "If the context contains the answer or any useful information, respond with that information. \n"
            "If no useful informations are, answer: no useful informations"
            "Answer:"
            "---------------------------"
        )
        
        if not self.config.use_rag:
            return None

        messages = [
            SystemMessage(content="You are a helpful assistant that summarizes text relevant to the question."),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        print("##SUMMARY CLEAN##")
        print(self._clean_section(response.content))
        return self._clean_section(response.content)

    def _clean_section(self, content: str) -> str:
        delimiter = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        subquery_section = content.split(delimiter)[-1].strip()
        subquery_section = subquery_section.lower().strip()
        print("##Current Response##")
        print(subquery_section)
        print("##")
        return subquery_section

    @staticmethod
    def is_useful(text: str) -> bool:
        t = text.strip().lower()
        if not t or t.startswith("i don't know") or t.startswith("no"):
            return False
        return True

    def clean_llm_output(self, content):
        # Define the delimiter after which the subqueries are located
        delimiter = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        # Split the content based on the delimiter
        if delimiter in content:

            subquery_section = content.split(delimiter, 1)[-1]
            # Use regex to extract lines matching the subquery format
            subquery_section = subquery_section.lower().strip()
            subqueries = re.findall(r"subquery \d+: (.*)", subquery_section.strip())
            return subqueries
        else:
            return []

    def generate_subqueries(
        self,
        original_query: str,
        current_context: Optional[str] = None
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
            SystemMessage(content="You are an assistant specializing in generating search queries."),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        print("######")
        print("Clean response: ", self.clean_llm_output(response.content))
        print("--------------------")
        return self.clean_llm_output(response.content)

    # @staticmethod
    # def duckduckgo_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    #     time.sleep(2)
    #     try:
    #         with DDGS() as ddgs:
    #             print("query:", query)
    #             results = ddgs.text(query, max_results=max_results)
    #         return [{"title": r.get("title", ""), "url": r.get("href", "")} for r in results]
    #     except Exception as e:
    #         logger.error(f"DuckDuckGo error: {e}")
    #         return []

    @staticmethod
    def duckduckgo_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Perform a DuckDuckGo search using LangChain DuckDuckGo wrapper.

        Returns a list of dicts with keys: 'title' and 'url'.
        """
        time.sleep(2)  # polite delay
        try:
            wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
            search = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list")
            # Use run() method with output_format="list" to get list of dicts
            results = search.invoke(query)
            # Each item is expected to have keys like: 'title', 'link', 'snippet'
            # Map 'link' to 'url' for compatibility with existing code
            formatted_results = []
            for r in results:
                snippet = r.get("snippet", "")
                url = r.get("link", "")  # note: it's "link" in LangChain results
                if url:
                    formatted_results.append({"snippet": snippet, "url": url})
            return formatted_results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []





    def integrate_with_llm( self, original: str, rag_doc: str, web_snippets: List[str]) -> Dict[str, str]:
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


        msgs = [SystemMessage(content="You are a research assistant."), HumanMessage(content=prompt)]
        resp = self.llm.invoke(msgs)
        # parse
        clean_content = self._clean_section(resp.content)
       
        sa_matches = re.findall(
            r"short answer:\s*(.*?)(?=detailed answer:)",
            clean_content,
            flags=re.IGNORECASE|re.DOTALL
        )
        da_matches = re.findall(
            r"detailed answer:\s*(.*)",
            clean_content,
            flags=re.IGNORECASE|re.DOTALL
        )
        short = sa_matches[-1].strip().rstrip(",") if sa_matches else ""
        detailed = da_matches[-1].strip() if da_matches else ""
        return {"short": short, "detailed": detailed}






    def process_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        qr = rec.get("input", "").strip()
        rag_ans = rec.get("answer", "") if self.config.use_rag else ""
        rag_summary = self.generate_summary(rag_ans, qr) if self.config.use_rag else None

        all_sources: Set[str] = set()
        current_context = rag_summary or ""
        final_short, final_detailed = "", ""
        web_summary = ""

        web_summaries = []

        for loop in range(self.config.n_loops):
            if self.config.use_rag:
                subs = self.generate_subqueries(qr, current_context)
            else:
                subs = self.generate_subqueries(qr)  # Based on original query only

            snippets, urls = [], []
            subquery_summaries = []

            for sq in subs:
                #print("subquery:", sq)
                res = self.duckduckgo_search(sq, max_results=self.config.max_searches)

                subquery_snippets = []
                for r in res:
                    if r["url"] not in all_sources:
                        all_sources.add(r["url"])
                    snippet = f"{r['snippet']})"
                    snippets.append(snippet)
                    #print("Current sub-snippet", snippet)
                    subquery_snippets.append(snippet)

                # Summarize each subquery's snippets independently if rag_summary is True
                if self.config.rag_summary:
                    if subquery_snippets:
                        combined_snippets = "\n".join(subquery_snippets)
                        summary = self.generate_summary(combined_snippets, sq)
                        subquery_summaries.append(summary)
                    else:
                        subquery_summaries.append("")

            if self.config.rag_summary:
                combined_sub_summaries = "\n".join([str(s) if s else "" for s in subquery_summaries])
                web_summary = self.generate_summary(combined_sub_summaries, qr)
                web_summaries.append(web_summary)
                #print("Current websummary: ", web_summary)
            
                # Combine rag summary, web summary, and original query for final integration
                context_for_llm = f"RAG informations:\n{rag_summary or ''}\n\nWeb informations:\n{web_summary}"
            else:
                # If not summarizing subqueries, use rag summary or current context with snippets
                context_for_llm = current_context

            combined_web_summaries = "\n".join([str(s) if s else "" for s in web_summaries])
            web_summary_all = self.generate_summary(combined_web_summaries, qr)

            # Integrate all info with LLM
            out = self.integrate_with_llm(qr, context_for_llm, snippets)
            final_short, final_detailed = out["short"], out["detailed"]

            # Prepare context for next loop iteration
            current_context = final_detailed



        return {
            "query": qr,
            "rag_summary": rag_summary if self.config.use_rag else None,
            "web_summary": web_summary_all if self.config.rag_summary else None,
            "short_answer": final_short,
            "detailed_answer": final_detailed,
            "sources": list(all_sources),
        }






    def run(self):
        # RAG pipeline
        if self.config.use_rag:
            if not self.config.rag_config_path:
                raise ValueError("rag_config_path required when use_rag=True")
            logger.info("Running RAG pipeline...")
            rag(self.config.rag_config_path)
            rc = self.config.access_rag_config()
            self.config.input_file = rc["mode_args"]["output_file"]
            with open(self.config.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            self.config.input_file = self.config.input_queries
            data = []
            with open(self.config.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))  # JSONL format


        outputs = []
        for rec in data:
            outputs.append(self.process_record(rec))
        # save
        outp = Path(self.config.output_file)
        outp.parent.mkdir(exist_ok=True, parents=True)
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {outp}")
