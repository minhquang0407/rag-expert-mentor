import uuid

from core.schemas import AgentTask, BlackboardState, PlanStep, RuntimePlan
from agents.base import BaseAgent


class PlannerAgent(BaseAgent):
    """Dynamic planner that creates a multi-agent execution plan."""

    name = "planner"
    role = "Pedagogical Planner"
    goal = "Create an ordered multi-agent teaching plan based on context, graph state, and learner memory."

    def run(self, task: AgentTask, blackboard: BlackboardState):
        required_agents = task.metadata.get("required_agents") or ["concept", "example"]
        target_section = task.metadata.get("target_section") or blackboard.target_section or blackboard.lesson_goal
        content_focus = task.metadata.get("content_focus") or blackboard.micro_context or blackboard.lesson_goal
        unlearned = blackboard.graph_context.get("unlearned", []) if blackboard.graph_context else []

        planned_agents = list(required_agents)
        if unlearned and "concept" not in planned_agents:
            planned_agents.insert(0, "concept")

        role_instructions = {
            "concept": "Explain only the intuition, definitions, metaphors, and misconceptions. Do not include formulas, derivations, pseudocode, or worked examples.",
            "formula": "Explain only notation, symbols, matrices, and equations. Do not include broad intuition or algorithmic pseudocode.",
            "math": "Explain only derivations, proof logic, and mathematical assumptions.",
            "algorithm": "Explain only procedures, pseudocode, complexity, and implementation edge cases. Do not repeat conceptual introduction.",
            "example": "Provide only worked examples, counterexamples, and exercises.",
        }
        steps = [
            PlanStep(
                step_id=f"step_{idx}_{agent_name}",
                agent_name=agent_name,
                instruction=(
                    f"Current section: {target_section}. Current focus: {content_focus}. "
                    f"{role_instructions.get(agent_name, 'Teach only your assigned specialist perspective.')} "
                    "Treat recent_history as background only; do not switch topics to prior lessons."
                ),
                requires_critic=True,
                metadata={"target_section": target_section, "content_focus": content_focus},
            )
            for idx, agent_name in enumerate(planned_agents)
        ]
        steps.append(
            PlanStep(
                step_id="step_composer",
                agent_name="composer",
                instruction="Compose the final polished response from specialist outputs.",
                requires_critic=False,
            )
        )

        plan = RuntimePlan(
            plan_id=str(uuid.uuid4()),
            objective=task.instruction,
            steps=steps,
            rationale="Used ingestion-time required_agents as hints and added prerequisite support when needed.",
        )
        blackboard.artifacts["runtime_plan"] = plan.model_dump()
        return self._build_result(
            task,
            f"Planned agent path: {' -> '.join(step.agent_name for step in steps)}",
            confidence=0.85,
            metadata={"runtime_plan": plan.model_dump()},
        )
