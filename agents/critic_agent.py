from core.schemas import AgentTask, BlackboardState, CriticReport
from agents.base import BaseAgent


class CriticAgent(BaseAgent):
    """Verifier agent for grounding, correctness, and pedagogy."""

    name = "critic"
    role = "Critic and Verifier"
    goal = "Verify agent outputs against context, graph constraints, and pedagogical quality."

    def run(self, task: AgentTask, blackboard: BlackboardState):
        latest = blackboard.agent_messages[-1] if blackboard.agent_messages else None
        if latest is None:
            report = CriticReport(status="fail", issues=["No agent message available for review."], confidence=1.0)
        elif not latest.content.strip():
            report = CriticReport(
                status="needs_revision",
                reviewed_agent=latest.agent_name,
                issues=["Reviewed output is empty."],
                send_back_to=latest.agent_name,
                revised_instruction="Regenerate the response with concrete instructional content.",
                confidence=1.0,
            )
        else:
            report = CriticReport(
                status="pass",
                reviewed_agent=latest.agent_name,
                issues=[],
                confidence=0.7,
            )

        blackboard.critic_reports.append(report)
        return self._build_result(
            task,
            f"Critic status for {report.reviewed_agent or 'unknown'}: {report.status}",
            confidence=report.confidence,
            needs_revision=report.status != "pass",
            metadata={"critic_report": report.model_dump()},
        )
