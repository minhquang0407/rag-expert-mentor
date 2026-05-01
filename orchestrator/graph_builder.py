from langgraph.graph import StateGraph, START, END
from typing import Literal
from orchestrator.state_machine import LessonState

from orchestrator.langgraph_nodes import (
    PlannerNode,
    QARetrievalNode,
    ConceptNode,
    FormulaNode,
    MathNode,
    AlgorithmNode,
    ExampleNode
)


class LessonOrchestrator:
    """
    - Lí do tại sao dùng: Xây dựng Đồ thị Luồng làm việc (Workflow) cho 5 Chuyên gia AI.
    - Chức năng: Khởi tạo Nodes, nối các cạnh (Edges) thành dây chuyền, và truyền Neo4j DB vào các Tác tử.
    """

    def __init__(self, db_store, neo4j_db, llm_service, checkpointer=None):
        self.db = db_store
        self.dag = neo4j_db
        self.llm = llm_service
        self.checkpointer = checkpointer
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(LessonState)

        # Cấp quyền truy cập Neo4j (self.dag) cho Planner và QA
        workflow.add_node("planner", PlannerNode(self.db, self.dag))
        workflow.add_node("qa", QARetrievalNode(self.db, self.dag, self.llm))

        # Cấp quyền truy cập Neo4j cho Hội đồng 5 Chuyên gia
        workflow.add_node("concept", ConceptNode(self.llm, self.dag))
        workflow.add_node("formula", FormulaNode(self.llm, self.dag))
        workflow.add_node("math", MathNode(self.llm, self.dag))
        workflow.add_node("algorithm", AlgorithmNode(self.llm, self.dag))
        workflow.add_node("example", ExampleNode(self.llm, self.dag))

        workflow.set_conditional_entry_point(
            self.route_start,
            {
                "to_planner": "planner",
                "to_pipeline": "concept",  # Bắt đầu dây chuyền từ Concept
                "to_qa": "qa"
            }
        )

        workflow.add_edge("planner", END)
        workflow.add_edge("qa", END)

        workflow.add_edge("concept", "formula")
        workflow.add_edge("formula", "math")
        workflow.add_edge("math", "algorithm")
        workflow.add_edge("algorithm", "example")
        workflow.add_edge("example", END)  # Xong 1 Cụm thực thể -> Dừng lại cho sinh viên đọc

        return workflow.compile(checkpointer=self.checkpointer)

    def route_start(self, state: LessonState) -> str:
        """Định tuyến tường minh dựa trên Action Mode thay vì dò chữ."""
        action_mode = state.get("action_mode", "QA")

        if action_mode == "QA":
            return "to_qa"

        if action_mode == "START_LESSON":
            return "to_planner"

        # Nếu là NEXT_GROUP thì chạy thẳng vào dây chuyền 5 Chuyên gia
        return "to_pipeline"

    def run_lesson(self, query: str, thread_id: str, target_chapter: str = "", target_section: str = "",
                   action_mode: str = "QA") -> str:
        """
        Endpoint giao tiếp với UI. Điều hướng bằng Action Mode tường minh.
        """
        print(f"\n" + "=" * 50)
        print(f"🎓 HỆ THỐNG NHẬN LỆNH: {query} | MODE: {action_mode}")
        print("=" * 50)

        config = {"configurable": {"thread_id": thread_id}}

        # Cờ này chỉ true khi mode là START_LESSON
        is_first_start = (action_mode == "START_LESSON")

        current_seq_idx = 0
        entity_groups = []

        # ==========================================
        # 1. KHÔI PHỤC TRẠNG THÁI VÀ TÍNH TOÁN INDEX
        # ==========================================
        try:
            current_state = self.app.get_state(config)
            if current_state and current_state.values:
                current_seq_idx = current_state.values.get("current_seq_index", 0)
                entity_groups = current_state.values.get("entity_groups", [])

                # Không cần dò chữ "hiểu, ok" nữa. Nếu UI gửi tín hiệu NEXT_GROUP -> auto tăng!
                if action_mode == "NEXT_GROUP":
                    current_seq_idx += 1
                    print(f"🔄 Đang chuyển sang Cụm Thực Thể số: {current_seq_idx}")
        except Exception as e:
            print(f"Không thể đọc state cũ (Có thể là phiên mới): {e}")

        # ==========================================
        # 2. KÉO DỮ LIỆU THẬT TỪ QDRANT
        # ==========================================
        if not entity_groups and target_chapter and target_section:
            print(f"📥 Đang kéo giáo án thật từ Qdrant cho: {target_chapter} -> {target_section}")
            try:
                entity_groups = self.db.get_curriculum_groups(target_file=target_chapter, target_section=target_section)
                if not entity_groups:
                    return f"⚠️ Lỗi: Không tìm thấy giáo án cho phần '{target_section}'."
            except Exception as e:
                return f"⚠️ Lỗi truy xuất CSDL Qdrant: {e}"

        if entity_groups and current_seq_idx >= len(entity_groups):
            return "🎉 Chúc mừng em! Chúng ta đã hoàn thành xuất sắc toàn bộ nội dung của Section này."

        # ==========================================
        # 3. KÍCH HOẠT ĐỒ THỊ
        # ==========================================
        initial_state = {
            "student_query": query,
            "target_file": target_chapter,
            "target_section": target_section,
            "action_mode": action_mode,
            "language": "Tiếng Việt",
            "is_planning_phase": True if is_first_start else False,
            "entity_groups": entity_groups,
            "current_seq_index": current_seq_idx,
            "lecture_parts": []
        }

        final_state = self.app.invoke(initial_state, config=config)
        ai_message = final_state.get("ai_response", "Error: Không nhận được bài giảng.")
        return ai_message

    def stream_lesson(self, query: str, thread_id: str, target_chapter: str = "", target_section: str = "",
                      action_mode: str = "QA"):
        """
        - Lí do tại sao dùng: Dành cho Developer Mode. Nhả (yield) từng bước thực thi của LangGraph ra UI thay vì bắt người dùng chờ.
        - Trả về: Một Generator (yield) chứa tên Node vừa chạy và State hiện tại.
        """
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
            pass  # Bỏ qua lỗi ở lần chạy đầu tiên

        # Kéo dữ liệu từ Qdrant
        if not entity_groups and target_chapter and target_section:
            yield {"node": "system", "message": "📥 Đang kéo giáo án từ Qdrant..."}
            try:
                entity_groups = self.db.get_curriculum_groups(target_file=target_chapter, target_section=target_section)
            except Exception as e:
                yield {"node": "error", "message": f"⚠️ Lỗi Qdrant: {e}"}
                return

        if entity_groups and current_seq_idx >= len(entity_groups):
            yield {"node": "finish", "message": "🎉 Đã hoàn thành Section!"}
            return

        initial_state = {
            "student_query": query,
            "target_file": target_chapter,
            "target_section": target_section,
            "action_mode": action_mode,
            "language": "Tiếng Việt",
            "is_planning_phase": True if is_first_start else False,
            "entity_groups": entity_groups,
            "current_seq_index": current_seq_idx,
            "lecture_parts": []
        }

        # SỬ DỤNG STREAMING
        # app.stream sẽ trả về từng bước (event) mỗi khi một Node thực thi xong
        for event in self.app.stream(initial_state, config=config):
            for node_name, state_update in event.items():
                yield {
                    "node": node_name,
                    "state_update": state_update
                }