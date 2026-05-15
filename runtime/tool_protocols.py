import re
import uuid
from typing import List, Tuple

from core.schemas import ToolCall
from runtime.protocols import parse_json_object

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def extract_tool_calls(text: str, agent_name: str) -> Tuple[str, List[ToolCall]]:
    """Extract tool_call JSON blocks and return cleaned display text plus typed calls."""
    calls: List[ToolCall] = []

    for match in TOOL_CALL_PATTERN.finditer(text or ""):
        raw_json = match.group(1).strip()
        try:
            parsed = parse_json_object(raw_json)
            tool_name = str(parsed.get("tool_name", "")).strip()
            if not tool_name:
                continue
            calls.append(ToolCall(
                call_id=str(uuid.uuid4()),
                agent_name=agent_name,
                tool_name=tool_name,
                arguments=parsed.get("arguments", {}) or {},
            ))
        except Exception as exc:
            print(f"[ToolProtocol] Failed to parse tool call from {agent_name}: {exc}")

    cleaned = TOOL_CALL_PATTERN.sub("", text or "").strip()
    return cleaned, calls
