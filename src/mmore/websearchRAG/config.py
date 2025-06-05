# mmore/websearch/config.py

from dataclasses import dataclass
from typing import Any, Dict

from rag.llm import LLMConfig  # Reuse the same LLMConfig as RAG


@dataclass
class WebsearchConfig:
    """
    Configuration for WebsearchPipeline.

    Fields:
      rag_config_path:    (str or None) Path to the RAG config YAML. Required if use_rag=True.
      use_rag:            (bool) If True, run RAG first; otherwise skip directly to sub-query generation.
      rag_summary:        (bool) If True, run an initial LLM-based summary of the RAG answer.
      input_file:         (str) Path to the JSON file used as “queries” (or RAG output).
      output_file:        (str) Path where the enhanced JSON results will be written.
      n_subqueries:       (int) Number of sub-queries to generate via LLM.
      max_searches:       (int) Max results to fetch from DuckDuckGo per sub-query.
      llm_config:         (dict) Passed to rag.llm.LLMConfig (keys: llm_name, max_new_tokens, temperature, etc.)
    """

    rag_config_path: str  # e.g. "../rag/config.yaml"
    use_rag: bool
    rag_summary: bool
    input_file: str
    output_file: str
    n_subqueries: int
    max_searches: int
    llm_config: Dict[str, Any]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WebsearchConfig":
        # Validate required keys
        required = ["use_rag", "rag_summary", "input_file", "output_file", "n_subqueries", "max_searches", "llm_config"]
        for key in required:
            if key not in d:
                raise ValueError(f"Missing '{key}' in WebsearchConfig.")
        rag_config_path = d.get("rag_config_path", "")
        return WebsearchConfig(
            rag_config_path=rag_config_path,
            use_rag=d["use_rag"],
            rag_summary=d["rag_summary"],
            input_file=d["input_file"],
            output_file=d["output_file"],
            n_subqueries=int(d["n_subqueries"]),
            max_searches=int(d["max_searches"]),
            llm_config=d["llm_config"],
        )

    def get_llm_config(self) -> LLMConfig:
        """
        Convert the nested llm_config dict into an instance of rag.llm.LLMConfig.
        """
        return LLMConfig(**self.llm_config)
