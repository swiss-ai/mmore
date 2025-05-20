"""Configuration for websearch pipeline."""
from dataclasses import dataclass
from typing import Dict, Any

from .llm import LLMConfig

@dataclass
class WebsearchConfig:
    """Configuration for websearch pipeline."""
    input_file: str
    output_file: str
    n_loops: int = 2
    llm_name: str = "OpenMeditron/meditron3-8b"
    max_searches: int = 10
    llm_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.llm_config is None:
            self.llm_config = {
                "max_new_tokens": 1000,
                "temperature": 0.7
            }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "WebsearchConfig":
        """Create config from dictionary."""
        return cls(
            input_file=config["input_file"],
            output_file=config["output_file"],
            n_loops=config.get("n_loops", 2),
            llm_name=config.get("llm_name", "OpenMeditron/meditron3-8b"),
            max_searches=config.get("max_searches", 10),
            llm_config=config.get("llm_config", None)
        )

    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return LLMConfig(
            llm_name=self.llm_name,
            max_new_tokens=self.llm_config["max_new_tokens"],
            temperature=self.llm_config["temperature"]
        ) 