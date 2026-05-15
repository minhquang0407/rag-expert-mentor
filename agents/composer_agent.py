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
        if blackboard.tool_results:
            artifact_lines = []
            artifact_index = 0
            for tool_result in blackboard.tool_results:
                for artifact in tool_result.artifacts:
                    title = artifact.title or tool_result.content or artifact.artifact_type
                    if artifact.artifact_type == "image":
                        artifact_lines.append(f"**{title}**\n\n<<tool:{artifact_index}>>")
                    elif artifact.path:
                        artifact_lines.append(f"- [{title}]({artifact.path})")
                    else:
                        artifact_lines.append(f"- {title}")
                    artifact_index += 1
            if artifact_lines:
                final_output += "\n\n---\n\n### Tool-Generated Visuals & Results\n\n" + "\n\n".join(artifact_lines)
        blackboard.final_output = final_output
        return self._build_result(task, final_output, confidence=0.8)
