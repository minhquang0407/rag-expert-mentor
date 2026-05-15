import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from core.schemas import AgentTraceEvent, BlackboardState, RuntimeTrace


class RuntimeTracer:
    """Lightweight local JSONL tracer for multi-agent runtime observability."""

    def __init__(self, trace_dir: str = "runtime_traces"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.current_trace: Optional[RuntimeTrace] = None

    def start_trace(
        self,
        *,
        user_id: str,
        target_file: str = "",
        target_section: str = "",
        workflow_mode: str = "",
    ) -> RuntimeTrace:
        self.current_trace = RuntimeTrace(
            run_id=str(uuid.uuid4()),
            user_id=user_id,
            target_file=target_file,
            target_section=target_section,
            workflow_mode=workflow_mode,
            started_at=time.time(),
        )
        self.record("trace_start", message=f"Started {workflow_mode} runtime trace.")
        return self.current_trace

    def record(
        self,
        event_type: str,
        *,
        agent_name: Optional[str] = None,
        task_id: Optional[str] = None,
        message: str = "",
        success: Optional[bool] = None,
        latency_ms: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentTraceEvent]:
        if self.current_trace is None:
            return None

        event = AgentTraceEvent(
            event_id=str(uuid.uuid4()),
            run_id=self.current_trace.run_id,
            event_type=event_type,
            timestamp=time.time(),
            agent_name=agent_name,
            task_id=task_id,
            message=message,
            success=success,
            latency_ms=latency_ms,
            payload=payload or {},
        )
        self.current_trace.events.append(event)
        return event

    def finish_trace(
        self,
        *,
        blackboard: Optional[BlackboardState] = None,
        success: bool = True,
    ) -> Optional[RuntimeTrace]:
        if self.current_trace is None:
            return None

        self.current_trace.completed_at = time.time()
        self.current_trace.success = success
        if blackboard is not None:
            self.current_trace.blackboard_snapshot = blackboard.model_dump()
        self.record("trace_end", message="Runtime trace completed.", success=success)
        self.persist()
        return self.current_trace

    def persist(self) -> None:
        if self.current_trace is None:
            return

        path = self.trace_dir / f"{self.current_trace.run_id}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for event in self.current_trace.events:
                file.write(json.dumps(event.model_dump(), ensure_ascii=False) + "\n")
            file.write(json.dumps({"runtime_trace": self.current_trace.model_dump()}, ensure_ascii=False) + "\n")

    def as_dict(self) -> Dict[str, Any]:
        return self.current_trace.model_dump() if self.current_trace is not None else {}
