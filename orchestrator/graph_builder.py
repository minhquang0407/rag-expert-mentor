from langgraph.graph import StateGraph, START, END
from typing import Literal
from orchestrator.state_machine import LessonState

from orchestrator.langgraph_nodes import (
    PlannerNode,
    QARetrievalNode,
    RouterNode,
    ConceptNode,
    FormulaNode,
    MathNode,
    AlgorithmNode,
    ExampleNode
)


class LessonOrchestrator:
    """
    - Lí do tại sao dùng: Xây dựng Đồ thị LangGraph với cơ chế True Sequential Bypass.
    - Chức năng: Định tuyến động để nối trực tiếp các Node với nhau dựa trên quyết định của Router, bỏ qua vật lý các Node không cần thiết.
    """

    def __init__(self, db_store, neo4j_db, llm_service, checkpointer=None):
        self.db = db_store
        self.dag = neo4j_db
        self.llm = llm_service
        self.checkpointer = checkpointer
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(LessonState)

        # 1. Các Node nền tảng
        workflow.add_node("planner", PlannerNode(self.db, self.dag))
        workflow.add_node("qa", QARetrievalNode(self.db, self.dag, self.llm))
        workflow.add_node("router", RouterNode(self.llm))

        # 2. Hội đồng 5 Chuyên gia
        workflow.add_node("concept", ConceptNode(self.llm, self.dag))
        workflow.add_node("formula", FormulaNode(self.llm, self.dag))
        workflow.add_node("math", MathNode(self.llm, self.dag))
        workflow.add_node("algorithm", AlgorithmNode(self.llm, self.dag))
        workflow.add_node("example", ExampleNode(self.llm, self.dag))

        workflow.set_conditional_entry_point(
            self.route_start,
            {
                "to_planner": "planner",
                "to_pipeline": "router",  # Nút "Vào học" sẽ chạy Router đầu tiên
                "to_qa": "qa"
            }
        )

        workflow.add_edge("planner", END)
        workflow.add_edge("qa", END)
        workflow.add_edge("router", "concept")  # Concept luôn luôn mở bát

        # ==========================================
        # [CƠ CHẾ TRUE BYPASS]: CÁC CẠNH CÓ ĐIỀU KIỆN
        # ==========================================
        # Sau khi Concept xong, kiểm tra xem đi đâu tiếp?
        workflow.add_conditional_edges("concept", self.route_after_concept)

        # Nếu Formula chạy, sau đó đi đâu tiếp?
        workflow.add_conditional_edges("formula", self.route_after_formula)

        # Nếu Math chạy, sau đó đi đâu tiếp?
        workflow.add_conditional_edges("math", self.route_after_math)

        # Algorithm luôn nối thẳng vào Example (vì Example là Node chốt sổ bắt buộc)
        workflow.add_edge("algorithm", "example")
        workflow.add_edge("example", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # --- CÁC HÀM TÍNH TOÁN CẠNH (EDGE ROUTING FUNCTIONS) ---
    def route_after_concept(self, state: LessonState) -> str:
        experts = state.get("routed_experts", [])
        if "formula" in experts: return "formula"
        if "math" in experts: return "math"
        if "algorithm" in experts: return "algorithm"
        return "example"

    def route_after_formula(self, state: LessonState) -> str:
        experts = state.get("routed_experts", [])
        if "math" in experts: return "math"
        if "algorithm" in experts: return "algorithm"
        return "example"

    def route_after_math(self, state: LessonState) -> str:
        experts = state.get("routed_experts", [])
        if "algorithm" in experts: return "algorithm"
        return "example"

    def route_start(self, state: LessonState) -> str:
        action_mode = state.get("action_mode", "QA")
        if action_mode == "QA": return "to_qa"
        if action_mode == "START_LESSON": return "to_planner"
        return "to_pipeline"

    # ==========================================
    # CÁC HÀM THỰC THI (GIỮ NGUYÊN)
    # ==========================================
    def run_lesson(self, query: str, thread_id: str, target_chapter: str = "", target_section: str = "",
                   action_mode: str = "QA") -> str:
        print(f"\n" + "=" * 50)
        print(f"🎓 HỆ THỐNG NHẬN LỆNH: {query} | MODE: {action_mode}")
        print("=" * 50)

        config = {"configurable": {"thread_id": thread_id}}
        is_first_start = (action_mode == "START_LESSON")
        current_seq_idx = 0
        entity_groups = []

        try:
            current_state = self.app.get_state(config)
            if current_state and current_state.values:
                current_seq_idx = current_state.values.get("current_seq_index", 0)
                entity_groups = current_state.values.get("entity_groups", [])

                if action_mode == "NEXT_GROUP":
                    current_seq_idx += 1
                    print(f"🔄 Đang chuyển sang Bước Số: {current_seq_idx}")
        except Exception as e:
            pass

        if not entity_groups and target_chapter and target_section:
            try:
                entity_groups = self.db.get_curriculum_groups(target_file=target_chapter, target_section=target_section)
                if not entity_groups: return f"⚠️ Lỗi: Không tìm thấy giáo án."
            except Exception as e:
                return f"⚠️ Lỗi truy xuất CSDL Qdrant: {e}"

        if entity_groups and current_seq_idx >= len(entity_groups):
            return "🎉 Chúc mừng em! Chúng ta đã hoàn thành xuất sắc toàn bộ nội dung của Section này."

        initial_state = {
            "student_query": query, "target_file": target_chapter, "target_section": target_section,
            "action_mode": action_mode, "language": "Tiếng Việt", "is_planning_phase": is_first_start,
            "entity_groups": entity_groups, "current_seq_index": current_seq_idx, "lecture_parts": []
        }

        final_state = self.app.invoke(initial_state, config=config)
        return final_state.get("ai_response", "Error: Không nhận được bài giảng.")

    def stream_lesson(self, query: str, thread_id: str, target_chapter: str = "", target_section: str = "",
                      action_mode: str = "QA"):
        print(f"\n[STREAMING MODE] 🎓 NHẬN LỆNH: {query}")
        config = {"configurable": {"thread_id": thread_id}}
        is_first_start = (action_mode == "START_LESSON")
        current_seq_idx = 0
        entity_groups = []

        try:
            current_state = self.app.get_state(config)
            if current_state and current_state.values:
                current_seq_idx = current_state.values.get("current_seq_index", 0)
                entity_groups = current_state.values.get("entity_groups", [])

                if action_mode == "NEXT_GROUP":
                    current_seq_idx += 1
        except Exception as e:
            pass

        if not entity_groups and target_chapter and target_section:
            yield {"node": "system", "message": "📥 Đang kéo lộ trình từ Qdrant..."}
            try:
                entity_groups = self.db.get_curriculum_groups(target_file=target_chapter, target_section=target_section)
            except Exception as e:
                yield {"node": "error", "message": f"⚠️ Lỗi Qdrant: {e}"}
                return

        if entity_groups and current_seq_idx >= len(entity_groups):
            yield {"node": "finish", "message": "🎉 Đã hoàn thành Section!"}
            return

        initial_state = {
            "student_query": query, "target_file": target_chapter, "target_section": target_section,
            "action_mode": action_mode, "language": "Tiếng Việt", "is_planning_phase": is_first_start,
            "entity_groups": entity_groups, "current_seq_index": current_seq_idx, "lecture_parts": []
        }

        for event in self.app.stream(initial_state, config=config):
            for node_name, state_update in event.items():
                yield {"node": node_name, "state_update": state_update}