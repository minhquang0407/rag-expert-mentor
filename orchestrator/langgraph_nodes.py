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

    def __init__(self, db: QdrantVectorStore, dag: SemanticDAG, llm_service):
        self.db = db
        self.dag = dag
        self.llm_service = llm_service
    def __call__(self, state: LessonState) -> Dict[str, Any]:
        print("\n[Node: QA_Retrieve] Kích hoạt Hybrid Search (BM25 + Dense Vector)...")

        query = state["student_query"]
        target_file = state.get("target_file", "")
        target_section = state.get("target_section", "")

        # GỌI API HYBRID SEARCH CỦA QDRANT (Lấy Top 3 kết quả)
        results = self.db.search_candidates_and_fetch_parent(query, self.llm_service, target_file)

        if not results:
            print("[-] Không tìm thấy câu hỏi giả định nào khớp ý định.")
            return {"structural_context": "Không có thông tin trong giáo trình để trả lời câu hỏi này.",
                    "dag_context": ""}

        print("[+] Đã truy xuất thành công Section chứa câu trả lời!")

        # Gộp 3 chunks lại để tăng độ phủ cho LLM
        chunk_text = results[0]["page_content"]
        meta = results[0].get("metadata", {})

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
            "dag_context": dag_context if dag_context else "Không có nền tảng tiên quyết."
        }


class PlannerNode:
    """
    - Lí do tại sao dùng: Tách biệt giai đoạn "Lập kế hoạch" ra khỏi giai đoạn "Giảng bài".
    - Chức năng: Đọc mảng `entity_groups` từ State và xuất ra một thông báo Lộ trình học tập thân thiện.
    - Cách dùng: Được Router gọi khi `is_planning_phase == True`.
    - Tham số: Cần truyền các phụ thuộc db (Trong phiên bản này, ta giả định dữ liệu đã được Router nạp vào State).
    - Trả về, Kiểu trả về: Dict cập nhật state, bao gồm cờ is_planning_phase và lưu luôn vào chat_history.
    - Các hàm thay thế nếu có: Không có.
    """

    def __init__(self, db_service):
        self.db = db_service

    def __call__(self, state: dict) -> Dict[str, Any]:
        print("\n[Node: Planner] Đang chuẩn bị Lộ trình học tập...")
        groups = state.get("entity_groups", [])
        section = state.get("target_section", "Chưa rõ")

        if not groups:
            err_msg = "Xin lỗi, Giáo sư không tìm thấy giáo án cho Section này. Có thể dữ liệu chưa được nạp đầy đủ."
            return {
                "ai_response": err_msg,
                "chat_history": [AIMessage(content=err_msg)]
            }

        plan_text = f"Chào em! Hôm nay chúng ta sẽ chinh phục **{section}**. Để dễ hiểu nhất, Giáo sư đã chia bài này thành {len(groups)} phần trọng tâm:\n\n"

        for group in groups:
            entities = ", ".join(group.get("core_entities", []))
            plan_text += f"- **Phần {group.get('seq_id')}: {group.get('group_name')}** (Trọng tâm: *{entities}*)\n"

        plan_text += "\nEm đã sẵn sàng bắt đầu Phần 1 chưa? Hãy phản hồi để chúng ta vào Checkpoint đầu tiên nhé!"

        return {
            "ai_response": plan_text,
            "is_planning_phase": False,
            "current_seq_index": 0,
            "current_checkpoint": 1,
            # Ghi luôn kế hoạch này vào lịch sử hội thoại để AI nhớ lộ trình
            "chat_history": [AIMessage(content=plan_text)]
        }


from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


class BaseExpertNode:
    """Base class containing shared LLM logic for the 5 Expert Agents."""

    def __init__(self, llm_service, dag):
        self.llm = llm_service.llm
        self.dag = dag

    def generate_expert_content(self, state: dict, expert_role: str, specific_rule: str) -> str:
        groups = state.get("entity_groups", [])
        seq_idx = state.get("current_seq_index", 0)
        language = state.get("language", "Vietnamese")

        current_group = groups[seq_idx] if groups else {}
        group_name = current_group.get("group_name", "Core Concept")
        verbatim_context = current_group.get("verbatim_text", "")
        core_entities = current_group.get("core_entities", [])

        dag_context = self.dag.get_backward_context(
            core_entities) if core_entities else "No prerequisite logic required."

        previous_parts = "\n\n".join(state.get("lecture_parts", []))

        prompt = f"""
        You are a Professor of Mathematics and Computer Science (Expert Mentor).
        Your teaching philosophy is "Bottom-Up" and focused on the core essence.

        [CURRENT TOPIC]: Teaching Entity Group: **{group_name}**
        [YOUR EXPERT ROLE]: {expert_role}

        [CURRENT LESSON CONTEXT (STRICTLY ADHERE)]:
        <TEXTBOOK_CORE>
        {verbatim_context}
        </TEXTBOOK_CORE>

        [PREREQUISITE FOUNDATION FROM DAG]:
        <PREREQUISITE_LOGIC>
        {dag_context}
        </PREREQUISITE_LOGIC>

        [PREVIOUS LECTURE PARTS BY OTHER EXPERTS]:
        <CONTEXT_CHAIN>
        {previous_parts if previous_parts else "You are the first expert to speak. Introduce the topic."}
        </CONTEXT_CHAIN>

        [YOUR SPECIFIC PROTOCOL & RULE]:
        {specific_rule}

        [MANDATORY TEACHING PRINCIPLES]:
        1. BOTTOM-UP THINKING: ALWAYS use the information in <PREREQUISITE_LOGIC> to explain <TEXTBOOK_CORE>.
        2. ANTI-HALLUCINATION (CRITICAL): ONLY rely on the provided <TEXTBOOK_CORE>. If information is missing, reply: "I don't know, I don't have any information about that!". Do not fabricate data.
        3. MATHEMATICS: Present mathematical formulas clearly using LaTeX (enclose in $$).
        4. COHESION: Read the <CONTEXT_CHAIN> to ensure your explanation seamlessly continues the logical flow. Do NOT repeat what previous experts have said.
        5. TEACHING LANGUAGE: MUST output your final content in {language}.

        Constraint: Return ONLY your section of the lecture. Start exactly with a Level 3 Markdown Header (###).
        """
        response = self.llm.invoke([SystemMessage(content=prompt)])
        return response.content



class ConceptNode(BaseExpertNode):
    def __call__(self, state: dict) -> Dict[str, Any]:
        print(" -> [Concept Expert] is drafting the geometric intuition...")
        rule = """
        HEADER: '### 1. Geometric Intuition & Concept Essence'. 
        TASK: Explain the core concept using real-world metaphors, geometric visualization, or physical analogies. 
        CONSTRAINT: Do NOT use complex mathematical formulas here. Your tone should be inspiring, focusing on the "WHY" and the absolute essence of the idea.
        """
        content = self.generate_expert_content(state, "Concept Intuition Expert", rule)
        return {"lecture_parts": [content]}


class FormulaNode(BaseExpertNode):
    def __call__(self, state: dict) -> Dict[str, Any]:
        print(" -> [Formula Expert] is standardizing mathematical notations...")
        rule = """
        HEADER: '### 2. Formal Theory & Mathematical Notation'. 
        TASK: Present the formal theory and mathematical formulas. 
        CONSTRAINT: You MUST tightly link the mathematical symbols and variables (LaTeX) to the spatial/geometric metaphors that the Concept Expert just established in the <CONTEXT_CHAIN>. Explain what each variable represents in reality.
        """
        content = self.generate_expert_content(state, "Formal Theory Expert", rule)
        return {"lecture_parts": [content]}


class MathNode(BaseExpertNode):
    def __call__(self, state: dict) -> Dict[str, Any]:
        print(" -> [Math Expert] is deriving the proofs...")
        rule = """
        HEADER: '### 3. Mathematical Proof & Derivation'. 
        TASK: Provide an in-depth mathematical proof or step-by-step derivation for the formulas established by the Formula Expert. 
        CONSTRAINT: Strictly use rigorous step-by-step logic. Handle derivatives, integrals, or matrix transformations with absolute precision using LaTeX.
        """
        content = self.generate_expert_content(state, "Mathematical Proof Expert", rule)
        return {"lecture_parts": [content]}


class AlgorithmNode(BaseExpertNode):
    def __call__(self, state: dict) -> Dict[str, Any]:
        print(" -> [Algorithm Expert] is designing computational logic...")
        rule = """
        HEADER: '### 4. Algorithm & Computational Logic'. 
        TASK: Translate the mathematical theory into computational logic. Provide pseudocode or explain the core data structures needed to compute this concept. 
        CONSTRAINT: Bridge the gap between continuous mathematics and discrete computer science logic.
        """
        content = self.generate_expert_content(state, "Algorithm & Software Engineer Expert", rule)
        return {"lecture_parts": [content]}


class ExampleNode(BaseExpertNode):
    def __call__(self, state: dict) -> Dict[str, Any]:
        print(" -> [Application Expert] is finalizing the lecture...")

        groups = state.get("entity_groups", [])
        seq_idx = state.get("current_seq_index", 0)
        group_name = groups[seq_idx].get("group_name", "this concept") if groups else "this concept"

        rule = f"""
        HEADER: '### 5. Practical Examples & Application'. 
        TASK: Provide a concrete, solved practical example applying the theory. 
        CONSTRAINT: At the very end of your response, you MUST ask the student this exact question to close the lecture loop: 'Em đã hiểu hoàn toàn phần {group_name} chưa để Giáo sư chuyển sang phần tiếp theo?'
        """
        content = self.generate_expert_content(state, "Application Expert", rule)

        all_parts = state.get("lecture_parts", []) + [content]
        final_lecture = "\n\n---\n\n".join(all_parts)

        return {
            "lecture_parts": [content],
            "ai_response": final_lecture,
            "chat_history": [AIMessage(content=final_lecture)]
        }