from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal


class TeachingStep(BaseModel):
    step_id: str = Field(description="Unique ID for the teaching step")
    step_title: str = Field(description="Title of the teaching step")
    content_focus: str = Field(description="Focus content (Raw text from the book)")

    # Array of Agent Queues decided by the Ingestion LLM
    required_agents: List[str] = Field(
        description="Ordered list of teaching experts. Choose only from: ['concept', 'formula', 'math', 'algorithm', 'example', 'dynamic:<role>']."
    )


class QueueState(BaseModel):
    current_step_id: str
    macro_context: str = Field(description="The full text of the current section to provide global understanding", default="")
    graph_context: str = "" # [NEW] Holds learned/unlearned prerequisites
    global_summary: str = ""
    concept_scratchpad: List[str] = []
    math_scratchpad: List[str] = []
    formula_scratchpad: List[str] = []
    algorithm_scratchpad: List[str] = []
    dynamic_scratchpad: List[str] = []

class QAResponse(BaseModel):
    """
    - Reason: To enforce a structured JSON output from the SupportAgent.
    - Function: Validates the LLM's answer.
    """
    answer: str = Field(description="The synthesized academic answer to the user's query.")


class AgentClaim(BaseModel):
    """A factual or pedagogical claim produced by an agent."""
    text: str = Field(description="The claim text")
    source: Optional[str] = Field(default=None, description="Optional source context or evidence")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Claim-level confidence score")


class AgentTask(BaseModel):
    """A typed instruction passed to an individual agent."""
    task_id: str = Field(description="Unique task identifier")
    agent_name: str = Field(description="Target agent name")
    instruction: str = Field(description="Natural language instruction for the agent")
    input_keys: List[str] = Field(default_factory=list, description="Blackboard keys the agent should read")
    output_key: Optional[str] = Field(default=None, description="Blackboard key where output should be written")
    requires_critic: bool = Field(default=True, description="Whether the output should be reviewed by CriticAgent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional task metadata")


class AgentMessage(BaseModel):
    """A structured message written by an agent to the shared blackboard."""
    agent_name: str
    role: str
    content: str
    claims: List[AgentClaim] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_revision: bool = False
    next_suggested_agents: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """The structured output of a completed agent task."""
    task_id: str
    agent_name: str
    message: AgentMessage
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class AgentTraceEvent(BaseModel):
    """One observable event emitted during a multi-agent runtime run."""
    event_id: str
    run_id: str
    event_type: str
    timestamp: float
    agent_name: Optional[str] = None
    task_id: Optional[str] = None
    message: str = ""
    success: Optional[bool] = None
    latency_ms: Optional[float] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class RuntimeTrace(BaseModel):
    """Persistent trace for one runtime execution."""
    run_id: str
    user_id: str = "guest_01"
    target_file: str = ""
    target_section: str = ""
    workflow_mode: str = ""
    started_at: float
    completed_at: Optional[float] = None
    success: bool = True
    events: List[AgentTraceEvent] = Field(default_factory=list)
    blackboard_snapshot: Dict[str, Any] = Field(default_factory=dict)


class QuizQuestion(BaseModel):
    """One multiple-choice assessment question."""
    question: str
    options: List[str] = Field(default_factory=list)
    answer_idx: int = 0
    explanation: str = ""
    concept_id: Optional[str] = None
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class QuizPayload(BaseModel):
    """A generated quiz for a lesson or concept group."""
    quiz_id: str
    title: str = "Assessment"
    questions: List[QuizQuestion] = Field(default_factory=list)
    target_concepts: List[str] = Field(default_factory=list)


class GradingResult(BaseModel):
    """Evaluation result for one submitted quiz."""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    correct_count: int = 0
    total_count: int = 0
    weak_concepts: List[str] = Field(default_factory=list)
    feedback: str = ""
    remediation_required: bool = False


class RemediationPlan(BaseModel):
    """Follow-up teaching plan for weak concepts detected by assessment."""
    weak_concepts: List[str] = Field(default_factory=list)
    recommended_agents: List[str] = Field(default_factory=list)
    instruction: str = ""


class ToolArtifact(BaseModel):
    """Artifact generated by a runtime tool."""
    artifact_id: str
    artifact_type: Literal["image", "table", "json", "text"] = "text"
    path: Optional[str] = None
    title: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A validated request for a runtime tool execution."""
    call_id: str
    agent_name: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result returned by a runtime tool execution."""
    call_id: str
    tool_name: str
    agent_name: str
    status: Literal["success", "error"] = "success"
    content: str = ""
    artifacts: List[ToolArtifact] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CriticReport(BaseModel):
    """Verification report produced by CriticAgent."""
    status: Literal["pass", "needs_revision", "fail"] = "pass"
    reviewed_agent: Optional[str] = None
    issues: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    send_back_to: Optional[str] = None
    revised_instruction: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class PlanStep(BaseModel):
    """One step in a runtime multi-agent execution plan."""
    step_id: str
    agent_name: str
    instruction: str
    depends_on: List[str] = Field(default_factory=list)
    requires_critic: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RuntimePlan(BaseModel):
    """A dynamic execution plan produced by PlannerAgent."""
    plan_id: str
    objective: str
    steps: List[PlanStep] = Field(default_factory=list)
    rationale: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BlackboardState(BaseModel):
    """Shared state for one multi-agent runtime execution."""
    user_id: str = "guest_01"
    target_file: str = ""
    target_section: str = ""
    lesson_goal: str = ""
    macro_context: str = ""
    micro_context: str = ""
    graph_context: Dict[str, Any] = Field(default_factory=dict)
    semantic_memory: List[Dict[str, Any]] = Field(default_factory=list)
    recent_history: List[Dict[str, Any]] = Field(default_factory=list)
    agent_messages: List[AgentMessage] = Field(default_factory=list)
    critic_reports: List[CriticReport] = Field(default_factory=list)
    final_output: str = ""
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)