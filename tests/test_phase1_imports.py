from core.schemas import (
    AgentTask,
    AgentMessage,
    AgentResult,
    CriticReport,
    RuntimePlan,
    PlanStep,
    BlackboardState,
)
from runtime.blackboard import Blackboard
from runtime.protocols import clean_think_tags, parse_json_object

msg = AgentMessage(
    agent_name="concept",
    role="teacher",
    content="hello",
)

bb = Blackboard()
bb.write_message(msg)

plan = RuntimePlan(
    plan_id="p1",
    objective="test",
    steps=[
        PlanStep(
            step_id="s1",
            agent_name="concept",
            instruction="teach",
        )
    ],
)

cleaned = clean_think_tags('<think>internal reasoning</think>{"ok": true}')
parsed = parse_json_object('```json\n{"ok": true}\n```')

print("latest_message:", bb.latest_message().content)
print("plan_agent:", plan.steps[0].agent_name)
print("cleaned:", cleaned)
print("parsed_ok:", parsed["ok"])
