from typing import Any, Dict, Iterable, Optional

from agents import (
    AlgorithmAgent,
    ComposerAgent,
    ConceptAgent,
    CriticAgent,
    ExampleAgent,
    FormulaAgent,
    GraphAgent,
    GraderAgent,
    MathAgent,
    MemoryAgent,
    PlannerAgent,
    QuizAgent,
    SupervisorAgent,
)
from core.schemas import AgentTask, BlackboardState, PlanStep, RuntimePlan
from runtime.protocols import agent_result_to_event, critic_report_to_event
from runtime.tracing import RuntimeTracer


class MultiAgentRuntime:
    """Executes GraphRAG-grounded pedagogical workflows with formal agents."""

    def __init__(
        self,
        llm_service: Any = None,
        vector_db: Any = None,
        graph_db: Any = None,
        agents: Optional[Dict[str, Any]] = None,
        critic_enabled: bool = True,
        max_revision_loops: int = 1,
        stream_agent_outputs: bool = True,
        tracer: Optional[RuntimeTracer] = None,
    ):
        self.llm_service = llm_service
        self.vector_db = vector_db
        self.graph_db = graph_db
        self.critic_enabled = critic_enabled
        self.max_revision_loops = max_revision_loops
        self.stream_agent_outputs = stream_agent_outputs
        self.tracer = tracer or RuntimeTracer()
        self.agents = agents or self._build_default_agents()

    def _build_default_agents(self) -> Dict[str, Any]:
        tools = {"vector_db": self.vector_db, "graph_db": self.graph_db}
        return {
            "supervisor": SupervisorAgent(self.llm_service, tools=tools),
            "planner": PlannerAgent(self.llm_service, tools=tools),
            "graph": GraphAgent(self.llm_service, tools=tools),
            "memory": MemoryAgent(self.llm_service, tools=tools),
            "concept": ConceptAgent(self.llm_service, tools=tools),
            "math": MathAgent(self.llm_service, tools=tools),
            "formula": FormulaAgent(self.llm_service, tools=tools),
            "algorithm": AlgorithmAgent(self.llm_service, tools=tools),
            "example": ExampleAgent(self.llm_service, tools=tools),
            "critic": CriticAgent(self.llm_service, tools=tools),
            "composer": ComposerAgent(self.llm_service, tools=tools),
            "quiz": QuizAgent(self.llm_service, tools=tools),
            "grader": GraderAgent(self.llm_service, tools=tools),
        }

    def build_blackboard(
        self,
        *,
        user_id: str = "guest_01",
        target_file: str = "",
        target_section: str = "",
        step_data: Optional[Dict[str, Any]] = None,
        macro_context: str = "",
        micro_context: str = "",
    ) -> BlackboardState:
        step_data = step_data or {}
        return BlackboardState(
            user_id=user_id,
            target_file=target_file,
            target_section=target_section,
            lesson_goal=step_data.get("step_title", "") or step_data.get("content_focus", ""),
            macro_context=macro_context,
            micro_context=micro_context or step_data.get("content_focus", ""),
        )

    def execute_learning(
        self,
        *,
        user_id: str,
        target_file: str,
        target_section: str,
        step_data: Dict[str, Any],
        macro_context: str,
        anchor_nodes: Optional[list[str]] = None,
    ) -> Iterable[Dict[str, Any]]:
        """Run a learning step and yield Streamlit-compatible events."""
        blackboard = self.build_blackboard(
            user_id=user_id,
            target_file=target_file,
            target_section=target_section,
            step_data=step_data,
            macro_context=macro_context,
        )
        required_agents = step_data.get("required_agents", ["concept", "example"])
        anchor_nodes = anchor_nodes or step_data.get("main_entities", []) or []

        self.tracer.start_trace(
            user_id=user_id,
            target_file=target_file,
            target_section=target_section,
            workflow_mode="LEARNING",
        )
        self.tracer.record("runtime_status", message="Initializing Multi-Agent Runtime...")
        yield {"type": "trace", "trace": self.tracer.as_dict()}
        yield {"type": "status", "message": "Initializing Multi-Agent Runtime..."}

        supervisor_task = AgentTask(
            task_id="supervisor_learning",
            agent_name="supervisor",
            instruction="Select the workflow mode for this learning step.",
            requires_critic=False,
            metadata={"action_mode": "LEARNING"},
        )
        supervisor_result = self.agents["supervisor"].run(supervisor_task, blackboard)
        blackboard.agent_messages.append(supervisor_result.message)
        yield agent_result_to_event(supervisor_result)

        if self.graph_db is not None and anchor_nodes:
            yield {"type": "status", "message": "GraphAgent is retrieving prerequisite context..."}
            graph_task = AgentTask(
                task_id="graph_context",
                agent_name="graph",
                instruction="Retrieve graph context for the current learning concepts.",
                requires_critic=False,
                metadata={"anchor_nodes": anchor_nodes},
            )
            graph_result = self.agents["graph"].run(graph_task, blackboard)
            blackboard.agent_messages.append(graph_result.message)
            yield agent_result_to_event(graph_result)

        if self.vector_db is not None or self.graph_db is not None:
            yield {"type": "status", "message": "MemoryAgent is retrieving learner memory..."}
            memory_task = AgentTask(
                task_id="memory_retrieval",
                agent_name="memory",
                instruction="Retrieve learner memory relevant to this learning step.",
                requires_critic=False,
                metadata={"query": blackboard.lesson_goal, "limit": 50},
            )
            memory_result = self.agents["memory"].run(memory_task, blackboard)
            blackboard.agent_messages.append(memory_result.message)
            yield agent_result_to_event(memory_result)

        yield {"type": "status", "message": "PlannerAgent is creating a dynamic teaching plan..."}
        planner_task = AgentTask(
            task_id="planner_learning",
            agent_name="planner",
            instruction=f"Plan a multi-agent lesson for: {blackboard.lesson_goal}",
            requires_critic=False,
            metadata={
                "required_agents": required_agents,
                "target_section": target_section,
                "content_focus": step_data.get("content_focus", ""),
            },
        )
        planner_result = self.agents["planner"].run(planner_task, blackboard)
        blackboard.agent_messages.append(planner_result.message)
        yield agent_result_to_event(planner_result)

        runtime_plan = self._with_support_steps(
            self._get_runtime_plan(blackboard, required_agents),
            include_graph=self.graph_db is not None and bool(anchor_nodes),
            include_memory=self.vector_db is not None or self.graph_db is not None,
        )
        blackboard.artifacts["runtime_plan"] = runtime_plan.model_dump()
        for step in runtime_plan.steps:
            if step.agent_name in {"composer", "graph", "memory"}:
                continue
            if step.agent_name not in self.agents:
                yield {"type": "status", "message": f"Skipping unknown agent: {step.agent_name}"}
                continue

            yield {"type": "status", "message": f"{step.agent_name.title()}Agent is working..."}
            yield {"type": "agent_start", "agent": step.agent_name}

            task = AgentTask(
                task_id=step.step_id,
                agent_name=step.agent_name,
                instruction=step.instruction,
                requires_critic=step.requires_critic,
                metadata=step.metadata,
            )
            result = self.agents[step.agent_name].run(task, blackboard)
            blackboard.agent_messages.append(result.message)

            if self.stream_agent_outputs:
                yield {"type": "chunk", "content": result.message.content}
            yield {"type": "agent_end", "agent": step.agent_name}
            yield agent_result_to_event(result)

            if self.critic_enabled and step.requires_critic:
                yield from self._run_critic(step.agent_name, blackboard)

        yield {"type": "status", "message": "ComposerAgent is synthesizing the final lesson..."}
        composer_task = AgentTask(
            task_id="composer_final",
            agent_name="composer",
            instruction="Compose the final polished lesson from specialist outputs.",
            requires_critic=False,
        )
        composer_result = self.agents["composer"].run(composer_task, blackboard)
        blackboard.agent_messages.append(composer_result.message)
        yield {"type": "final", "content": composer_result.message.content}
        yield agent_result_to_event(composer_result)

        if self.vector_db is not None or self.graph_db is not None:
            yield {"type": "status", "message": "MemoryAgent is persisting the final learning turn..."}
            persist_task = AgentTask(
                task_id="memory_persist_learning",
                agent_name="memory",
                instruction="Persist the final learning turn.",
                requires_critic=False,
                metadata={
                    "action": "persist",
                    "query": f"Learn {target_section}_{step_data.get('seq_id', 0)}",
                    "answer": blackboard.final_output,
                    "summary": f"[LECTURE] Multi-agent lesson completed: {target_section}",
                    "anchor_nodes": anchor_nodes,
                },
            )
            persist_result = self.agents["memory"].run(persist_task, blackboard)
            blackboard.agent_messages.append(persist_result.message)
            yield agent_result_to_event(persist_result)

        self.tracer.record("runtime_status", message="Multi-Agent learning step completed.")
        self.tracer.finish_trace(blackboard=blackboard, success=True)
        yield {"type": "trace", "trace": self.tracer.as_dict()}
        yield {"type": "status", "message": "Multi-Agent learning step completed."}

    def generate_assessment(
        self,
        *,
        user_id: str,
        target_file: str,
        target_section: str,
        step_data: Dict[str, Any],
        macro_context: str,
        question_count: int = 5,
    ) -> Dict[str, Any]:
        blackboard = self.build_blackboard(
            user_id=user_id,
            target_file=target_file,
            target_section=target_section,
            step_data=step_data,
            macro_context=macro_context,
        )
        concepts = step_data.get("main_entities", [])
        blackboard.artifacts["target_concepts"] = concepts
        task = AgentTask(
            task_id="quiz_generation",
            agent_name="quiz",
            instruction="Generate a diagnostic multiple-choice quiz for the current lesson.",
            requires_critic=False,
            metadata={"concepts": concepts, "question_count": question_count},
        )
        result = self.agents["quiz"].run(task, blackboard)
        return result.message.metadata.get("quiz", {}) if result.success else {"error": result.error or result.message.content}

    def grade_assessment(
        self,
        *,
        quiz: Dict[str, Any],
        answers: Dict[str, Any],
        user_id: str = "guest_01",
        pass_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        blackboard = BlackboardState(user_id=user_id)
        blackboard.artifacts["quiz"] = quiz
        task = AgentTask(
            task_id="quiz_grading",
            agent_name="grader",
            instruction="Grade the submitted quiz and produce remediation guidance.",
            requires_critic=False,
            metadata={"quiz": quiz, "answers": answers, "pass_threshold": pass_threshold},
        )
        result = self.agents["grader"].run(task, blackboard)
        return result.message.metadata

    def _run_critic(self, reviewed_agent: str, blackboard: BlackboardState):
        critic_task = AgentTask(
            task_id=f"critic_review_{reviewed_agent}",
            agent_name="critic",
            instruction=f"Review the output from {reviewed_agent}.",
            requires_critic=False,
        )
        critic_result = self.agents["critic"].run(critic_task, blackboard)
        blackboard.agent_messages.append(critic_result.message)
        yield critic_report_to_event(blackboard.critic_reports[-1])
        yield agent_result_to_event(critic_result)

    def _with_support_steps(
        self,
        runtime_plan: RuntimePlan,
        *,
        include_graph: bool,
        include_memory: bool,
    ) -> RuntimePlan:
        """Expose pre-planner support agents in the persisted runtime plan without rerunning them."""
        existing_agents = [step.agent_name for step in runtime_plan.steps]
        support_steps = []
        if include_graph and "graph" not in existing_agents:
            support_steps.append(PlanStep(
                step_id="support_graph_context",
                agent_name="graph",
                instruction="Retrieve prerequisite and learner graph context from Neo4j before teaching.",
                requires_critic=False,
                metadata={"phase": "context_retrieval", "executed_before_planner": True},
            ))
        if include_memory and "memory" not in existing_agents:
            support_steps.append(PlanStep(
                step_id="support_memory_retrieval",
                agent_name="memory",
                instruction="Retrieve semantic memory from Qdrant and recent learning history from Neo4j before teaching.",
                requires_critic=False,
                metadata={"phase": "context_retrieval", "executed_before_planner": True},
            ))

        if not support_steps:
            return runtime_plan

        runtime_plan.steps = support_steps + runtime_plan.steps
        runtime_plan.rationale = "GraphAgent/MemoryAgent provide context before the planned specialist path. " + runtime_plan.rationale
        return runtime_plan

    def _get_runtime_plan(self, blackboard: BlackboardState, required_agents: list[str]) -> RuntimePlan:
        raw_plan = blackboard.artifacts.get("runtime_plan")
        if raw_plan:
            return RuntimePlan(**raw_plan)

        # Fallback plan if PlannerAgent did not write an artifact.
        from core.schemas import PlanStep

        return RuntimePlan(
            plan_id="fallback_plan",
            objective=blackboard.lesson_goal,
            steps=[
                PlanStep(
                    step_id=f"fallback_{idx}_{agent_name}",
                    agent_name=agent_name,
                    instruction=f"Teach the current focus as the {agent_name} agent.",
                    requires_critic=True,
                )
                for idx, agent_name in enumerate(required_agents)
            ],
        )
