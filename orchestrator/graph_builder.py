from langgraph.graph import StateGraph, START, END
from .state_machine import LessonState
from .langgraph_nodes import LessonRetrievalNode, QARetrievalNode, TeacherNode


def route_action(state: LessonState) -> str:
    """
    Hàm định tuyến logic (Router).
    Kiểm tra cờ 'action_mode' trong State để quyết định nhánh tiếp theo.
    """
    mode = state.get("action_mode", "QA")
    if mode == "LESSON_PROGRESS":
        return "lesson_retrieve"
    return "qa_retrieve"


class LessonOrchestrator:
    def __init__(self, db_store, dag_store, llm_service, checkpointer=None):
        self.db = db_store
        self.dag = dag_store
        self.llm = llm_service
        self.checkpointer = checkpointer
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(LessonState)

        # 1. Khai báo các Node
        workflow.add_node("lesson_retrieve", LessonRetrievalNode(self.db, self.dag))
        workflow.add_node("qa_retrieve", QARetrievalNode(self.db, self.dag, self.llm))
        workflow.add_node("teacher", TeacherNode(self.llm))

        # 2. Khai báo CẠNH ĐIỀU KIỆN (Ngã ba định tuyến từ START)
        workflow.add_conditional_edges(
            START,
            route_action,
            {
                "lesson_retrieve": "lesson_retrieve",
                "qa_retrieve": "qa_retrieve"
            }
        )

        # 3. Khai báo các Cạnh hội tụ (Cả 2 đường đều dẫn tới Teacher)
        workflow.add_edge("lesson_retrieve", "teacher")
        workflow.add_edge("qa_retrieve", "teacher")
        workflow.add_edge("teacher", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def run_lesson(self, query: str, thread_id: str, target_chapter: str = "", target_section: str = "",
                   checkpoint: int = 1, action_mode: str = "QA") -> str:
        """Hàm giao tiếp với Sinh viên (Endpoint)."""
        print(f"\n" + "=" * 50)
        print(f"🎓 HỆ THỐNG NHẬN LỆNH: {query} | MODE: {action_mode}")
        print("=" * 50)

        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "student_query": query,
            "target_file": target_chapter,
            "target_section": target_section,
            "action_mode": action_mode,
            "current_checkpoint": checkpoint,
            "structural_context": "",
            "dag_context": "",
            "chat_history": [],
            "assessment_result": {}
        }

        final_state = self.app.invoke(initial_state, config=config)
        ai_message = final_state["chat_history"][-1].content
        return ai_message