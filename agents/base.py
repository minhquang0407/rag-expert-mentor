from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from core.schemas import AgentMessage, AgentResult, AgentTask, BlackboardState
from runtime.protocols import clean_think_tags, normalize_latex_markdown


class BaseAgent(ABC):
    """Base interface for all multi-agent runtime agents."""

    name: str = "base"
    role: str = "base_agent"
    goal: str = "Provide a structured agent result."

    def __init__(self, llm_service: Any = None, tools: Optional[Dict[str, Any]] = None):
        self.llm_service = llm_service
        self.tools = tools or {}

    @abstractmethod
    def run(self, task: AgentTask, blackboard: BlackboardState) -> AgentResult:
        """Execute an agent task and return a structured result."""
        raise NotImplementedError

    def _build_result(
        self,
        task: AgentTask,
        content: str,
        *,
        confidence: float = 0.0,
        needs_revision: bool = False,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        message = AgentMessage(
            agent_name=self.name,
            role=self.role,
            content=normalize_latex_markdown(clean_think_tags(content)),
            confidence=confidence,
            needs_revision=needs_revision,
            metadata=metadata or {},
        )
        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            message=message,
            success=success,
            error=error,
        )

    def _llm_invoke_text(self, system_prompt: str, human_prompt: str) -> str:
        """Invoke the configured chat LLM and return cleaned text."""
        if not self.llm_service:
            raise RuntimeError(f"{self.name} has no llm_service configured.")

        chat_llm = getattr(self.llm_service, "chat_llm", None) or getattr(self.llm_service, "llm", None)
        if chat_llm is None:
            raise RuntimeError(f"{self.name} could not find a chat LLM on llm_service.")

        response = chat_llm.invoke([
            ("system", system_prompt),
            ("human", human_prompt),
        ])
        return normalize_latex_markdown(clean_think_tags(getattr(response, "content", str(response))))
