import uuid
from typing import List, Optional

from core.schemas import AgentTask, BlackboardState
from agents.base import BaseAgent


class MemoryAgent(BaseAgent):
    """Tool agent responsible for Qdrant and Neo4j learner memory."""

    name = "memory"
    role = "Learner Memory Agent"
    goal = "Retrieve, summarize, and persist learner memory."

    def run(self, task: AgentTask, blackboard: BlackboardState):
        action = task.metadata.get("action", "retrieve")
        if action == "persist":
            return self.persist_turn(task, blackboard)
        return self.retrieve_memory(task, blackboard)

    def retrieve_memory(self, task: AgentTask, blackboard: BlackboardState):
        vector_db = self.tools.get("vector_db")
        graph_db = self.tools.get("graph_db")
        query = task.metadata.get("query", blackboard.lesson_goal or task.instruction)
        limit = int(task.metadata.get("limit", 50))

        vector_error = None
        graph_error = None

        if vector_db is not None:
            try:
                raw_semantic_memory = vector_db.search_semantic_memory(blackboard.user_id, query, limit=5)
                focus_terms = [
                    str(blackboard.lesson_goal).lower(),
                    str(blackboard.target_section).lower(),
                    str(blackboard.micro_context).lower(),
                ]
                focus_terms = [term for term in focus_terms if term and term != "none"]
                blackboard.semantic_memory = [
                    item for item in raw_semantic_memory
                    if any(term in str(item).lower() or str(item).lower() in term for term in focus_terms)
                ][:3]
            except Exception as exc:
                vector_error = str(exc)
                blackboard.semantic_memory = []
                print(f"[MemoryAgent] ⚠️ Semantic memory retrieval skipped: {vector_error}")

        if graph_db is not None:
            try:
                blackboard.recent_history = graph_db.get_recent_history(blackboard.user_id, limit=min(limit, 3))
            except Exception as exc:
                graph_error = str(exc)
                blackboard.recent_history = []
                print(f"[MemoryAgent] ⚠️ Recent history retrieval skipped: {graph_error}")

        warnings = {}
        if vector_error:
            warnings["semantic_memory"] = vector_error
        if graph_error:
            warnings["recent_history"] = graph_error

        return self._build_result(
            task,
            f"Retrieved {len(blackboard.semantic_memory)} semantic memories and {len(blackboard.recent_history)} recent turns.",
            confidence=0.9 if not warnings else 0.6,
            metadata={"warnings": warnings},
        )

    def persist_turn(self, task: AgentTask, blackboard: BlackboardState):
        vector_db = self.tools.get("vector_db")
        graph_db = self.tools.get("graph_db")
        turn_id = task.metadata.get("turn_id", str(uuid.uuid4()))
        query = task.metadata.get("query", blackboard.lesson_goal or task.instruction)
        answer = task.metadata.get("answer", blackboard.final_output)
        summary = task.metadata.get("summary", answer[:500])
        anchor_nodes: Optional[List[str]] = task.metadata.get("anchor_nodes")

        try:
            if vector_db is not None:
                vector_db.upsert_user_memory(blackboard.user_id, turn_id, query, answer, summary)
            if graph_db is not None:
                graph_db.save_chat_turn(
                    user_id=blackboard.user_id,
                    turn_id=turn_id,
                    query=query,
                    raw_answer=answer,
                    summary=summary,
                    concept_ids=anchor_nodes or [],
                    target_file=blackboard.target_file,
                    target_section=blackboard.target_section,
                )

            return self._build_result(task, f"Persisted memory turn {turn_id}.", confidence=0.9)
        except Exception as exc:
            return self._build_result(task, f"MemoryAgent persistence failed: {exc}", success=False, error=str(exc))
