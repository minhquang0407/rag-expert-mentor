from typing import Any, Dict, List, Optional

from core.schemas import AgentMessage, BlackboardState, CriticReport


class Blackboard:
    """Convenience wrapper around BlackboardState for agent collaboration."""

    def __init__(self, state: Optional[BlackboardState] = None, **kwargs):
        self.state = state or BlackboardState(**kwargs)

    def write_message(self, message: AgentMessage) -> None:
        self.state.agent_messages.append(message)

    def write_critic_report(self, report: CriticReport) -> None:
        self.state.critic_reports.append(report)

    def set_artifact(self, key: str, value: Any) -> None:
        self.state.artifacts[key] = value

    def get_artifact(self, key: str, default: Any = None) -> Any:
        return self.state.artifacts.get(key, default)

    def set_final_output(self, content: str) -> None:
        self.state.final_output = content

    def messages_by_agent(self, agent_name: str) -> List[AgentMessage]:
        return [msg for msg in self.state.agent_messages if msg.agent_name == agent_name]

    def latest_message(self, agent_name: Optional[str] = None) -> Optional[AgentMessage]:
        messages = self.messages_by_agent(agent_name) if agent_name else self.state.agent_messages
        return messages[-1] if messages else None

    def context_snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot useful for prompts and debugging."""
        return self.state.model_dump()
