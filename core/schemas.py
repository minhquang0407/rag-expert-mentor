from pydantic import BaseModel, Field
from typing import List, Optional


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