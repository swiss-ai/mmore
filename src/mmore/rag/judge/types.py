"""Judge types: decisions, results, and configuration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..llm import LLMConfig


class JudgeDecision(str, Enum):
    PROCEED = "PROCEED"
    ADD_QUESTIONS = "ADD_QUESTIONS"
    ADD_CONTEXT = "ADD_CONTEXT"
    RE_RETRIEVE = "RE_RETRIEVE"


@dataclass
class JudgeResult:
    decision: JudgeDecision
    reason: str = ""
    exit_reason: str = ""
    context_relevance_score: Optional[float] = None
    extra_questions: List[str] = field(default_factory=list)
    web_query: Optional[str] = None
    retrieve_params: Optional[Dict[str, Any]] = None
    llm_invoked: bool = False
    raw_decision: Optional[str] = None
    coerced_decision: bool = False

    @classmethod
    def proceed(
        cls,
        reason: str,
        *,
        llm_invoked: bool = False,
        **kwargs: Any,
    ) -> "JudgeResult":
        return cls(
            decision=JudgeDecision.PROCEED,
            reason=reason,
            exit_reason=reason,
            llm_invoked=llm_invoked,
            **kwargs,
        )


@dataclass
class JudgeConfig:
    """LLM judge config. Only LLMJudge is implemented."""

    llm: LLMConfig
    system_prompt: str
    user_prompt: str
    metric_thresholds: Dict[str, float] = field(default_factory=dict)
    max_corrective_steps: int = 1
    allow_add_questions: bool = True
    allow_add_context: bool = True
    allow_re_retrieve: bool = True
    rerank_after_merge: bool = False
    max_web_results: int = 5
    max_chunk_chars: int = 400
    force_corrective_action: Optional[str] = None
