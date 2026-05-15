from core.schemas import AgentTask, BlackboardState
from agents.base import BaseAgent


class GraphAgent(BaseAgent):
    """Tool agent responsible for Neo4j graph context retrieval."""

    name = "graph"
    role = "GraphRAG Context Agent"
    goal = "Retrieve learned, unlearned, and related concepts from Neo4j."

    def run(self, task: AgentTask, blackboard: BlackboardState):
        graph_db = self.tools.get("graph_db")
        if graph_db is None:
            return self._build_result(task, "GraphAgent has no graph_db tool configured.", success=False)

        target_concepts = task.metadata.get("concepts") or task.metadata.get("anchor_nodes") or []
        learned, unlearned = [], []

        try:
            for concept in target_concepts:
                learned.extend(graph_db.get_learned_prerequisites(concept, user_id=blackboard.user_id))
                unlearned.extend(graph_db.get_unlearned_prerequisites(concept, user_id=blackboard.user_id))

            blackboard.graph_context.update({
                "target_concepts": target_concepts,
                "learned": sorted(set(learned)),
                "unlearned": sorted(set(unlearned)),
            })

            content = (
                f"Graph context retrieved. Learned prerequisites: {blackboard.graph_context['learned']}. "
                f"Unlearned prerequisites: {blackboard.graph_context['unlearned']}."
            )
            return self._build_result(task, content, confidence=0.9)
        except Exception as exc:
            return self._build_result(task, f"GraphAgent failed: {exc}", success=False, error=str(exc))
