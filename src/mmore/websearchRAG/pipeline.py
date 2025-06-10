# mmore/websearch/pipeline.py
from .logging_config import logger


import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import logging
import json


from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, SystemMessage



from ..run_rag import rag
from ..run_rag import read_queries
from ..rag.llm import LLM, LLMConfig
from .config import WebsearchConfig



#python -m mmore websearch --config-file examples/websearchRAG/config.yaml



class WebsearchPipeline:
    """
    Pipeline for optionally running RAG first, then generating sub-queries,
    performing web searches, and finally integrating everything via the LLM.
    """

    def __init__(self, config: WebsearchConfig):
        self.config = config

        # Initialize the LLM
        self.llm = self._initialize_llm()

        # Will store RAG results if we run RAG
        self.rag_results: Optional[List[Dict[str, Any]]] = None



    def _initialize_llm(self) -> LLM:
        """
        Initialize the LLM using the updated configuration for the fine-tuned multilingual model.
        """
        if self.config.use_rag:
            rag_config = self.config.access_rag_config()
            llm_config_dict = rag_config.get("rag", {}).get("llm", None)
            if llm_config_dict is None:
                raise ValueError("Missing 'llm' config under 'rag' key in the RAG configuration.")
            llm_config = LLMConfig(**llm_config_dict)
            return LLM.from_config(llm_config)
        else:
            llm_config = self.config.get_llm_config()
            # Ensure the model is configured for multilingual use
            llm_config.language_support = "multilingual"
            llm_config.model_name = "fine_tuned_multilingual_model"
            return LLM.from_config(llm_config)


    def clean_llm_output(self, content):
        # Define the delimiter after which the subqueries are located
        delimiter = "---------------------------<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
        # Split the content based on the delimiter
        if delimiter in content:

            subquery_section = content.split(delimiter, 1)[-1]
            # Use regex to extract lines matching the subquery format
            subquery_section = subquery_section.lower().strip()
            subqueries = re.findall(r"subquery \d+: (.*)", subquery_section.strip())
            print(f"subquery_section:", subquery_section)
            print('subquery outputs', subqueries)
            return subqueries
        else:
            return []



        
    @staticmethod
    def extract_answer_after_prefix(raw: str, prefix: str) -> str:
        """
        Extract everything after a given prefix (case-insensitive).
        If prefix not found, return raw.
        """
        raw = WebsearchPipeline.clean_llm_output(raw)
        pattern = re.compile(re.escape(prefix) + r"(.*)", re.IGNORECASE | re.DOTALL)
        match = pattern.search(raw)
        return match.group(1).strip() if match else raw

    @staticmethod
    def is_useful_rag_answer(rag_answer: str) -> bool:
        """
        Decide whether the RAG answer is “useful.” We treat nonempty answers
        that do not just say “I don’t know” as useful.
        """
        ans = rag_answer.strip().lower()
        if not ans:
            return False
        # If it begins with a phrase like “i don’t know” or “no answer” etc.
        if ans.startswith("i don’t know") or ans.startswith("i dont know") or ans.startswith("no"):
            return False
        return True

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
        task = (
            f"Partial RAG Answer: {rag_answer}\n\n"
            f"Generate {n}-independant subqueries to refine the answer based on the original query. Each subquery must be concise and ≤30 words.\n"
            f"The subqueries should print in this format: subquery 1: new question,  subquery 2: new question, etc. \n"
            "---------------------------"
        )


        prompt = instruction + task
        messages = [
            SystemMessage(content="You are an assistant specializing in generating search queries."),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        print("######")
        print("LLM response: ", response.content)
        print("######")
        print("Clean response: ", self.clean_llm_output(response.content))
        print("--------------------")
        return self.clean_llm_output(response.content)

    def integrate_with_web(
        self,
        original_query: str,
        base_knowledge: str,
        all_results: List[Dict[str, str]]
    ) -> str:
        """
        Integrate web results and base knowledge into a comprehensive answer.
        """
        collated_sources = "\n".join([f"Source: {r['title']}, URL: {r['url']}" for r in all_results])
        prompt = (
            f"Original Query: {original_query}\n\n"
            f"Base Knowledge: {base_knowledge}\n\n"
            f"Web Results:\n{collated_sources}\n\n"
            "Combine the base knowledge and web results into a detailed answer. "
            "Enclose your response within <enhanced_answer>...</enhanced_answer> tags."
        )

        messages = [
            SystemMessage(content="You are a multilingual research assistant."),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        return self.clean_llm_output(response.content)


    @staticmethod
    def duckduckgo_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Perform a DuckDuckGo search and return a list of {'title', 'url'} dicts.
        """
        try:
            with DDGS() as ddgs:
                return [
                    {"title": r.get("title", ""), "url": r.get("href", "")}
                    for r in ddgs.text(query, max_results=max_results)
                ]
        except Exception as e:
            logger.info(f"Search error for \"{query}\": {e}")
            return []


    def integrate_with_web(
        self,
        original_query: str,
        base_knowledge: str,
        all_results: List[Dict[str, str]]
    ) -> str:
        """
        Integrate web results and base knowledge into a comprehensive answer.
        """
        collated_sources = "\n".join([f"Source: {r['title']}, URL: {r['url']}" for r in all_results])
        prompt = (
            f"Original Query: {original_query}\n\n"
            f"Base Knowledge: {base_knowledge}\n\n"
            f"Web Results:\n{collated_sources}\n\n"
            "Combine the base knowledge and web results into a detailed answer. "
            "Enclose your response within <enhanced_answer>...</enhanced_answer> tags."
        )

        messages = [
            SystemMessage(content="You are a multilingual research assistant."),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)

    @staticmethod
    def extract_enhanced_answer(text: Optional[str]) -> Dict[str, Any]:
        """
        Extract the content between <enhanced_answer>...</enhanced_answer>, and any “ADDITIONAL GAPS” afterwards.
        Handles NoneType input gracefully.
        """
        if not text or not isinstance(text, str):
            return {"answer": "", "additional_gaps": ""}

        # 1) Extract between tags
        match = re.search(r"<enhanced_answer>\s*(.*?)\s*</enhanced_answer>", text, re.DOTALL | re.IGNORECASE)
        answer = match.group(1).strip() if match else text.strip()

        # 2) Extract “ADDITIONAL GAPS:” block if present
        gaps = ""
        gap_match = re.search(r"ADDITIONAL GAPS:\s*(.*)$", text, re.DOTALL | re.IGNORECASE)
        if gap_match:
            gaps = gap_match.group(1).strip()

        return {"answer": answer, "additional_gaps": gaps}


    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single record (dict). The record must contain at least:
          - "input": the original query (str)
          - If use_rag=True, "answer": the RAG answer (str) for that query
            (The RAG pipeline’s output JSON is expected to have those keys.)

        Returns a dict with:
          - "query" (original input)
          - "rag_answer" (empty string if use_rag=False)
          - "generated_subqueries": [list of sub-queries]
          - "enhanced_answer": the final answer (text inside <enhanced_answer>…</enhanced_answer>)
          - "additional_gaps": any remaining gaps (empty str if none)
          - "sources": [ { "title": ..., "url": ... } ] (all unique sources gathered)
        """
        user_query = record.get("input", "").strip()
        rag_answer = record.get("answer", "").strip() if self.config.use_rag else ""

        # Decide if we have “useful” RAG info
        useful = False
        if self.config.use_rag and self.is_useful_rag_answer(rag_answer):
            useful = True

        # Step 1: Optionally run a summary on the RAG answer if requested
        if useful and self.config.rag_summary:
            rag_answer = self.generate_summary(rag_answer, user_query)

        # Step 2: Generate sub-queries
        subqueries = self.generate_subqueries(user_query, rag_answer if useful else None)

        # Step 3: For each sub-query, run DuckDuckGo search and collect sources
        all_sources: List[Dict[str, str]] = []
        seen_urls: Set[str] = set()
        for subq in subqueries:
            logger.debug(f"subquery: {subq}")
            results = self.duckduckgo_search(subq, max_results=self.config.max_searches)
            for r in results:
                if r["url"] not in seen_urls:
                    all_sources.append(r)
                    seen_urls.add(r["url"])


        # Step 4: Integrate RAG answer (if any) or just original query with all web results
        base_knowledge = rag_answer if useful else ""
        combined = self.integrate_with_web(user_query, base_knowledge, all_sources)

        extracted = self.extract_enhanced_answer(combined)
        final_answer = extracted["answer"]
        final_gaps = extracted["additional_gaps"]

        return {
            "query": user_query,
            "rag_answer": rag_answer,
            "generated_subqueries": subqueries,
            "enhanced_answer": final_answer,
            "additional_gaps": final_gaps,
            "sources": all_sources,
        }

    def run(self):
        """
        1) If use_rag=True: invoke run_rag(...) to produce RAG output JSON.
        2) Read self.config.input_file (either original queries or RAG output) as a JSON array.
        3) For each record, call process_record(...) and collect outputs.
        4) Write all outputs to self.config.output_file as JSON array.
        """
        # If use_rag=True, run the RAG pipeline first
        if self.config.use_rag:
            if not self.config.rag_config_path:
                raise ValueError("rag_config_path is required when use_rag=True.")

            logger.info(f"Running RAG pipeline with config: {self.config.rag_config_path}")
            # We expect run_rag(...) to return a List[Dict] if return_results=True.
            rag(self.config.rag_config_path)
            logger.info("RAG pipeline completed.")

            rag_cfg = self.config.access_rag_config()
            output_file = rag_cfg["mode_args"]["output_file"]
            self.config.input_file = output_file

            logger.info("Updated input file for the pipeline")

            #resume RAG also


            # The RAG pipeline also writes its own JSON to rag_config.mode_args.output_file.
            # We assume that output path equals self.config.input_file at this point.
            # (In practice, you should set self.config.input_file to that path in your YAML.)
        else:
            self.rag_results = None
            #no rag --> generate queries



        #Step 2: 



        # Step 2: Load the JSON to process. It should be a list of records.
        with open(self.config.input_file, "r", encoding="utf-8") as f:
            data: List[Dict[str, Any]] = json.load(f)

        # Step 3: Process each record
        all_outputs: List[Dict[str, Any]] = []  
        for record in data:
            out = self.process_record(record)
            all_outputs.append(out)

        
        
        # Step 4: Save results
        out_path = Path(self.config.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(all_outputs, out_f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {out_path}")



    def generate_summary(self, rag_answer: str, query: str) -> str:
        """
        Summarize the RAG answer (used when rag_summary=True).
        """
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
        return self.extract_answer_after_prefix(response.content, "Answer:")
