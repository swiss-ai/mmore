from .logging_config import logger

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import logging
import time

from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, SystemMessage

from ..run_rag import rag
from ..run_rag import read_queries
from ..rag.llm import LLM, LLMConfig
from .config import WebsearchConfig

#python -m mmore websearch --config-file examples/websearchRAG/config.yaml

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
        if self.config.use_rag:
            rag_cfg = self.config.access_rag_config()
            llm_conf = rag_cfg.get("rag", {}).get("llm")
            if llm_conf is None:
                raise ValueError("Missing 'llm' config under 'rag' in RAG configuration.")
            return LLM.from_config(LLMConfig(**llm_conf))
        else:
            base_conf = self.config.get_llm_config()
            return LLM.from_config(base_conf)

    def generate_summary(self, rag_answer: str, query: str) -> str:
        """
        Summarize the RAG answer (used when rag_summary=True).
        """
        prompt = (
            "You have only the following context to answer the question, do not use any external knowledge.\n\n"
            f"Question: {query}\n\n"
            "Context:\n"
            f"{rag_answer}\n\n"
            "If the context contains the answer or useful information, respond with that information. \n"
            "If no useful informations are, answer: no useful informations"
            "Answer:"
            "---------------------------"

        )

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
        print("##Current Summary##")
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
            #print(f"subquery_section:", subquery_section)
            #print('subquery outputs', subqueries)
            return subqueries
        else:
            return []


    def generate_subqueries(
        self,
        original_query: str,
        rag_answer: Optional[str] = None
    ) -> List[str]:
        """
        Generate concise search subqueries using the fine-tuned multilingual model.
        """
        n = self.config.n_subqueries
        instruction = f"You have the question and partial answer below:\nQuestion: {original_query}\n\n"
        if rag_answer is None: 
            task = (
            f"Generate {n}-independant subqueries based the original query, in order to generate the most complete research. Each subquery must be concise and ≤30 words.\n"
            f"The subqueries should print in this format: subquery 1: new question,  subquery 2: new question, etc. \n"
        
            )
        else:
            task = (
            f"Partial RAG Answer: {rag_answer}\n\n"
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
        #print("######")
        #print("LLM response: ", response.content)
        print("######")
        print("Clean response: ", self.clean_llm_output(response.content))
        print("--------------------")
        return self.clean_llm_output(response.content)






    @staticmethod
    def duckduckgo_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
        time.sleep(1)
        try:
            with DDGS() as ddgs:
                print("query:", query)
                results = ddgs.text(query, max_results=max_results)
            return [{"title": r.get("title",""), "url": r.get("href","")} for r in results]
        except Exception as e:
            logger.error(f"DuckDuckGo error: {e}")
            return []



    def integrate_with_llm(
        self,
        original: str,
        rag_doc: str,
        web_snippets: List[str]
    ) -> Dict[str, str]:
        # Build prompt for short & detailed answer
        sources = "\n".join(web_snippets)
        prompt = (
            f"Original Query: {original}\n"
            f"RAG Document: {rag_doc}\n"
            f"Web Snippets:\n{sources}\n"
            "Provide output as: short answer: ..., detailed answer: ..."
        )
        msgs = [SystemMessage(content="You are a research assistant."), HumanMessage(content=prompt)]
        resp = self.llm.invoke(msgs)
        # parse
        sa = re.search(r"short answer:\s*(.*?)\s*(?=detailed answer:)", resp.content, flags=re.IGNORECASE|re.DOTALL)
        da = re.search(r"detailed answer:\s*(.*)", resp.content, flags=re.IGNORECASE|re.DOTALL)
        return {"short": sa.group(1).strip() if sa else "", "detailed": da.group(1).strip() if da else ""}

    def process_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        qr = rec.get("input","").strip()
        # RAG step
        rag_ans = rec.get("answer","") if self.config.use_rag else ""
        rag_summary = self.generate_summary(rag_ans, qr) if self.config.use_rag else None
        # iterative loops
        all_sources: Set[str] = set()
        current_context = rag_summary or ""
        final_short, final_detailed = "", ""
        for loop in range(self.config.n_loops):
            # generate subqueries
            subs = self.generate_subqueries(qr, current_context)
            snippets, urls = [], []
            for sq in subs:
                print("subquery: ", sq)
                res = self.duckduckgo_search(sq, max_results=self.config.max_searches)
                for r in res:
                    if r["url"] not in all_sources:
                        all_sources.add(r["url"])
                        # For simplicity assume snippet = title
                        snippets.append(f"{r['title']} ({r['url']})")
            # integrate with LLM
            out = self.integrate_with_llm(qr, current_context, snippets)
            final_short, final_detailed = out["short"], out["detailed"]
            # next context = detailed answer
            current_context = final_detailed
        return {
            "query": qr,
            "rag_summary": rag_summary,
            "short_answer": final_short,
            "detailed_answer": final_detailed,
            "sources": list(all_sources)
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

        # load input
        with open(self.config.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        outputs = []
        for rec in data:
            outputs.append(self.process_record(rec))
        # save
        outp = Path(self.config.output_file)
        outp.parent.mkdir(exist_ok=True, parents=True)
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {outp}")
