# mmore/websearch/config.py

from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path
import yaml

from ..rag.llm import LLMConfig  # Reuse the same LLMConfig as RAG


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
    n_loops : int
    max_searches: int
    llm_config: Dict[str, Any]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WebsearchConfig":
        # Validate required keys
        required = ["use_rag", "rag_summary", "input_file", "output_file", "n_loops", "n_subqueries", "max_searches", "llm_config"]
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
            n_loops=d["n_loops"],
            n_subqueries=int(d["n_subqueries"]),
            max_searches=int(d["max_searches"]),
            llm_config=d["llm_config"],
        )

    def get_llm_config(self) -> LLMConfig:
        """
        Convert the nested llm_config dict into an instance of rag.llm.LLMConfig.
        """
        return LLMConfig(**self.llm_config)
    

    def access_rag_config(self) -> Dict[str, Any]:
        """
        Access and parse the RAG configuration file defined in `rag_config_path`.

        Returns:
            A dictionary representing the RAG configuration.
        """
        if not self.rag_config_path:
            raise ValueError("The 'rag_config_path' is not defined.")

        # Resolve the full path to the RAG config file
        rag_config_full_path = Path(self.rag_config_path)

        if not rag_config_full_path.exists():
            raise FileNotFoundError(f"RAG config file not found at {rag_config_full_path}")

        # Load the RAG configuration
        with open(rag_config_full_path, "r") as file:
            rag_config = yaml.safe_load(file)

        return rag_config