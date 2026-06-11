"""Per-agent configuration dataclass."""

from dataclasses import dataclass, field
from typing import List, Optional

from ...rag.llm import LLMConfig


@dataclass
class AgentConfig:
    """Definition of a single agent in the privacy system."""

    llm: LLMConfig
    name: str = "agent"
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)
    checkpointer: Optional[str] = None
    checkpoint_path: Optional[str] = None
