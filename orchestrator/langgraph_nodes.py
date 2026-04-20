from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .state_machine import LessonState
from database.structural_db import QdrantVectorStore
from database.semantic_dag import SemanticDAG


class LessonRetrievalNode:
    """Node chuyên trách việc lấy toàn bộ bài giảng (Macro-Retrieval)."""

    def __init__(self, db: QdrantVectorStore, dag: SemanticDAG):
        self.db = db
        self.dag = dag

    def __call__(self, state: LessonState) -> Dict[str, Any]:
        print("\n[Node: Lesson_Retrieve] Lấy toàn bộ Section bằng Qdrant Scroll API...")

        target_file = state.get("target_file", "")
        target_section = state.get("target_section", "")

        # GỌI API SCROLL CỦA QDRANT ĐÃ ĐƯỢC ĐÓNG GÓI
        results = self.db.get_section_exact(target_file, target_section)

        if not results:
            return {"structural_context": "Không có thông tin trong sách.", "dag_context": ""}

        # Gộp toàn bộ văn bản của Section
        chunk_text = "\n\n".join([r["page_content"] for r in results])

        # Trích xuất và khử trùng lặp Anchor Nodes
        unique_anchors = set()
        for r in results:
            meta = r.get("metadata", {})
            if meta and meta.get("anchor_nodes"):
                anchors = [n.strip() for n in meta.get("anchor_nodes").split(",")]
                unique_anchors.update(anchors)

        dag_context = ""
        if unique_anchors:
            # Lội ngược dòng Đồ thị từ danh sách Siêu Đỉnh
            dag_context = self.dag.get_backward_context(list(unique_anchors))

        return {
            "structural_context": chunk_text,
            "dag_context": dag_context if dag_context else "Không có nền tảng tiên quyết."
        }


class QARetrievalNode:
    """Node chuyên trách việc tìm kiếm câu trả lời bằng Hybrid Search (Micro-Retrieval)."""

    def __init__(self, db: QdrantVectorStore, dag: SemanticDAG):
        self.db = db
        self.dag = dag

    def __call__(self, state: LessonState) -> Dict[str, Any]:
        print("\n[Node: QA_Retrieve] Kích hoạt Hybrid Search (BM25 + Dense Vector)...")

        query = state["student_query"]
        target_file = state.get("target_file", "")
        target_section = state.get("target_section", "")

        # GỌI API HYBRID SEARCH CỦA QDRANT (Lấy Top 3 kết quả)
        results = self.db.hybrid_search(query, target_file, target_section, limit=3)

        if not results:
            return {"structural_context": "Không có thông tin.", "dag_context": ""}

        # Gộp 3 chunks lại để tăng độ phủ cho LLM
        chunk_text = "\n\n---\n\n".join([r["page_content"] for r in results])

        # Trích xuất ngữ cảnh Đồ thị cho cả 3 chunks
        unique_anchors = set()
        for r in results:
            meta = r.get("metadata", {})
            if meta and meta.get("anchor_nodes"):
                anchors = [n.strip() for n in meta.get("anchor_nodes").split(",")]
                unique_anchors.update(anchors)

        dag_context = self.dag.get_backward_context(list(unique_anchors)) if unique_anchors else ""

        return {
            "structural_context": chunk_text,
            "dag_context": dag_context if dag_context else "Không có nền tảng tiên quyết."}


class TeacherNode:
    def __init__(self, llm_service):
        # Trích xuất đối tượng Chat model thuần túy từ service
        self.llm = llm_service.llm

    def __call__(self, state: LessonState) -> Dict[str, Any]:
        print("\n[Node: Teacher_Node] Giáo sư AI đang biên soạn bài giảng...")

        # 1. Thu thập nguyên liệu từ State (Cuốn sổ liên lạc)
        query = state["student_query"]
        core_context = state["structural_context"]
        pre_context = state["dag_context"]
        checkpoint = state.get("current_checkpoint", 1)
        history = state.get("chat_history", [])
        language = state["language"]
        # 2. Xây dựng System Prompt Sư phạm (Mix Engine Thực chiến)

        system_prompt = f"""
        You are a Professor of Mathematics and Computer Science (Expert Mentor).
        Your teaching philosophy is "Bottom-Up" and focused on the core essence.

        [CURRENT LESSON CONTEXT]:
        <TEXTBOOK_CORE>
        {core_context}
        </TEXTBOOK_CORE>

        [PREREQUISITE FOUNDATION FROM PREVIOUS CHAPTERS]:
        <PREREQUISITE_LOGIC>
        {pre_context}
        </PREREQUISITE_LOGIC>

        [EXECUTION PROTOCOL (CURRENTLY AT CHECKPOINT {checkpoint})]:
        You MUST ONLY teach according to the current Checkpoint:
        - IF Checkpoint = 1: Explain the Geometric Intuition and Concept Essence. End with a question that stimulates logical thinking.
        - IF Checkpoint = 2: Provide an in-depth Mathematical Proof (Strictly use LaTeX). End with a calculation exercise.
        - IF Checkpoint = 3: Provide Practical Examples and Application Exercises.

        [MANDATORY TEACHING PRINCIPLES]:
        1. BOTTOM-UP THINKING: ALWAYS use the information in <PREREQUISITE_LOGIC> as a stepping stone to explain <TEXTBOOK_CORE>. Do not skip the foundation.
        2. ANTI-HALLUCINATION (CRITICAL): ONLY rely on the provided Context. If information is missing or you are unsure, you MUST reply with this exact phrase: "Tôi không biết, tôi không có thông tin về nó." (I don't know, I do not have information about it). Do not fabricate, speculate, or create alternative data.
        3. MATHEMATICS: Present mathematical formulas clearly.
        4. ATTITUDE: Profound, academic, yet thought-provoking. Praise the student if they ask a good question.
        5. TEACHING LANGUAGE: {language}
        """
        # 3. Nạp bộ nhớ hội thoại (Memory injection)
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(history)  # Đưa toàn bộ lịch sử trò chuyện cũ vào
        messages.append(HumanMessage(content=query))  # Chèn câu hỏi hiện tại vào cuối

        # 4. Kích hoạt Suy luận LLM
        response = self.llm.invoke(messages)
        print(system_prompt)
        print("[Node: Teacher_Node] Bài giảng đã hoàn tất.")

        # 5. Cập nhật State
        # Trả về mảng chứa Câu hỏi của User và Câu trả lời của AI.
        # LangGraph sẽ tự động dùng operator.add để nối mảng này vào lịch sử tổng.
        return {
            "chat_history": [HumanMessage(content=query), AIMessage(content=response.content)]
        }