from core.schemas import AgentTask, BlackboardState
from agents.base import BaseAgent


class SupervisorAgent(BaseAgent):
    """Top-level controller for selecting and monitoring agentic workflows."""

    name = "supervisor"
    role = "Runtime Supervisor"
    goal = "Classify the user action and coordinate the high-level multi-agent workflow."

    def run(self, task: AgentTask, blackboard: BlackboardState):
        action_mode = task.metadata.get("action_mode", "LEARNING")
        allowed_modes = {"LEARNING", "LOCAL_QA", "GLOBAL_QA", "QUIZ", "REMEDIATION"}
        normalized_mode = action_mode if action_mode in allowed_modes else "LEARNING"
        blackboard.artifacts["action_mode"] = normalized_mode

        return self._build_result(
            task,
            f"Supervisor selected workflow mode: {normalized_mode}",
            confidence=0.8,
            metadata={"action_mode": normalized_mode},
        )
