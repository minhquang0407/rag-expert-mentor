from core.schemas import AgentTask, BlackboardState
from agents.base import BaseAgent


class ComposerAgent(BaseAgent):
    """Final synthesis agent that produces the polished user-facing output."""

    name = "composer"
    role = "Final Composer"
    goal = "Merge agent outputs into a coherent, polished lesson or answer."

    def run(self, task: AgentTask, blackboard: BlackboardState):
        source_messages = [
            msg for msg in blackboard.agent_messages
            if msg.agent_name not in {"critic", "composer", "memory", "graph"}
        ]

        if not source_messages:
            content = "No specialist agent content is available to compose."
            blackboard.final_output = content
            return self._build_result(task, content, success=False)

        composed = []
        for msg in source_messages:
            title = msg.role or msg.agent_name.title()
            composed.append(f"### {title}\n\n{msg.content.strip()}")

        final_output = "\n\n---\n\n".join(composed).strip()
        blackboard.final_output = final_output
        return self._build_result(task, final_output, confidence=0.8)
