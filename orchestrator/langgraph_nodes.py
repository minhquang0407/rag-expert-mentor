from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .state_machine import LessonState
from database.structural_db import QdrantVectorStore
from database.semantic_dag import Neo4jManager


class LessonRetrievalNode:
    """Node chuyên trách việc lấy toàn bộ bài giảng (Macro-Retrieval)."""

    def __init__(self, db: QdrantVectorStore, dag: Neo4jManager):
        self.db = db
        self.dag = dag

    def __call__(self, state: LessonState) -> Dict[str, Any]:
        """
        - Lí do tại sao dùng: Nạp toàn bộ dữ liệu văn bản từ Qdrant và cấu trúc đồ thị từ Neo4j cho Section hiện tại.
        - Chức năng: Kéo dữ liệu gốc từ DB.
        - Cách dùng: Gọi ở pha đầu của luồng START_LESSON.
        - Tham số: state (chứa target_file, target_section).
        - Trả về: Cập nhật structural_context và dag_context.
        """
        print("\n[Node: Lesson_Retrieve] Lấy toàn bộ Section bằng Qdrant Scroll API...")

        target_file = state.get("target_file", "")
        target_section = state.get("target_section", "")

        results = self.db.get_section_exact(target_file, target_section)

        if not results:
            return {"structural_context": "Không có thông tin trong sách.", "dag_context": ""}

        chunk_text = "\n\n".join([r["page_content"] for r in results])

        unique_anchors = set()
        for r in results:
            meta = r.get("metadata", {})
            if meta and meta.get("anchor_nodes"):
                anchors = [n.strip() for n in meta.get("anchor_nodes").split(",")]
                unique_anchors.update(anchors)

        dag_context = ""
        if unique_anchors:
            # Lấy bức tranh toàn cảnh (PDF Graph) của các siêu đỉnh
            subgraphs = []
            for anchor in unique_anchors:
                sg = self.dag.get_concept_subgraph(anchor)
                if sg["prerequisites"] or sg["leads_to"]:
                    subgraphs.append(
                        f"Khái niệm '{anchor}' cần học trước {sg['prerequisites']} và dẫn tới {sg['leads_to']}")
            dag_context = "\n".join(subgraphs)

        return {
            "structural_context": chunk_text,
            "dag_context": dag_context if dag_context else "Không có nền tảng tiên quyết."
        }


class QARetrievalNode:
    """Node chuyên trách việc tìm kiếm câu trả lời bằng Hybrid Search (Micro-Retrieval)."""

    def __init__(self, db: QdrantVectorStore, dag: Neo4jManager, llm_service):
        self.db = db
        self.dag = dag
        self.llm_service = llm_service

    def __call__(self, state: LessonState) -> Dict[str, Any]:
        print("\n[Node: QA_Retrieve] Kích hoạt Hybrid Search (BM25 + Dense Vector)...")

        query = state["student_query"]
        target_file = state.get("target_file", "")

        results = self.db.search_candidates_and_fetch_parent(query, self.llm_service, target_file)

        if not results:
            return {"structural_context": "Không có thông tin trong giáo trình để trả lời câu hỏi này.",
                    "dag_context": ""}

        chunk_text = results[0]["page_content"]
        meta = results[0].get("metadata", {})

        unique_anchors = set()
        for r in results:
            meta = r.get("metadata", {})
            if meta and meta.get("anchor_nodes"):
                anchors = [n.strip() for n in meta.get("anchor_nodes").split(",")]
                unique_anchors.update(anchors)

        dag_context = ""
        if unique_anchors:
            subgraphs = []
            for anchor in unique_anchors:
                sg = self.dag.get_concept_subgraph(anchor)
                subgraphs.append(f"'{anchor}' cần: {sg['prerequisites']} -> Dẫn tới: {sg['leads_to']}")
            dag_context = "\n".join(subgraphs)

        return {
            "structural_context": chunk_text,
            "dag_context": dag_context if dag_context else "Không có nền tảng tiên quyết."
        }


class PlannerNode:
    def __init__(self, db_service: QdrantVectorStore, dag_service: Neo4jManager):
        self.db = db_service
        self.dag = dag_service

    def __call__(self, state: dict) -> Dict[str, Any]:
        """
        - Lí do tại sao dùng: Lập lộ trình dạy dựa trên `teaching_roadmap` và chẩn đoán lỗ hổng cho TẤT CẢ các concepts sẽ xuất hiện.
        """
        print("\n[Node: Planner] Đang lập Lộ trình giảng dạy và chẩn đoán lỗ hổng...")
        steps = state.get("entity_groups", [])
        section = state.get("target_section", "Chưa rõ")

        if not steps:
            err_msg = "Xin lỗi, Giáo sư không tìm thấy lộ trình cho bài học này."
            return {"ai_response": err_msg, "chat_history": [AIMessage(content=err_msg)]}

        # Quét lỗ hổng qua toàn bộ các khái niệm trong lộ trình
        all_unlearned = set()
        for step in steps:
            for concept in step.get("associated_concepts", []):
                missing = self.dag.get_unlearned_prerequisites(concept, max_depth=2)
                all_unlearned.update(missing)

        plan_text = f"Chào em! Hôm nay chúng ta sẽ chinh phục **{section}**. "

        if all_unlearned:
            plan_text += f"\n\n⚠️ **CHẨN ĐOÁN LỖ HỔNG:** Giáo sư nhận thấy em chưa nắm vững các kiến thức nền tảng sau: *{', '.join(all_unlearned)}*. Đừng lo, Giáo sư sẽ lồng ghép ôn tập chúng trong quá trình giảng bài nhé!\n\n"
        else:
            plan_text += "Rất tuyệt, nền tảng của em đang rất vững. Chúng ta sẽ vào thẳng bài học.\n\n"

        plan_text += f"Lộ trình bài học của chúng ta gồm {len(steps)} bước:\n"

        for step in steps:
            concepts_str = ", ".join(step.get('associated_concepts', []))
            plan_text += f"- **Bước {step.get('seq_id')}: {step.get('step_title')}** (Liên quan: *{concepts_str}*)\n"

        plan_text += "\nEm đã sẵn sàng bắt đầu Bước 1 chưa? Hãy nhấn 'Vào học' nhé!"

        return {
            "ai_response": plan_text,
            "is_planning_phase": False,
            "current_seq_index": 0,
            "current_checkpoint": 1,
            "chat_history": [AIMessage(content=plan_text)]
        }


class BaseExpertNode:
    def __init__(self, llm_service, dag: Neo4jManager):
        self.llm = llm_service.llm
        self.dag = dag

    def generate_expert_content(self, state: dict, expert_role: str, specific_rule: str) -> str:
        """
        - Chức năng: Giảng dạy dựa trên 1 Bước (TeachingStep). Truyền verbatim quotes làm cốt lõi.
        """
        steps = state.get("entity_groups", [])
        seq_idx = state.get("current_seq_index", 0)
        language = state.get("language", "Vietnamese")

        current_step = steps[seq_idx] if steps else {}
        step_title = current_step.get("step_title", "Nội dung bài học")
        verbatim_context = current_step.get("verbatim_exact_quotes", "")
        associated_concepts = current_step.get("associated_concepts", [])

        # Lấy Điểm neo và Toàn cảnh Neo4j dựa trên Dàn concept của bước này
        learned_anchors = set()
        subgraphs = []
        for concept in associated_concepts:
            learned_anchors.update(self.dag.get_learned_prerequisites(concept, max_depth=3))
            sg = self.dag.get_concept_subgraph(concept)
            if sg["prerequisites"] or sg["leads_to"] or sg["related_concepts"]:
                subgraphs.append(
                    f"'{concept}' cần: {sg['prerequisites']} -> Dẫn tới: {sg['leads_to']}. Liên quan: {sg['related_concepts']}")

        anchors_text = ", ".join(learned_anchors) if learned_anchors else "Không có dữ liệu cũ để liên hệ."
        dag_context = "\n".join(subgraphs) if subgraphs else "Khái niệm độc lập."
        previous_parts = "\n\n".join(state.get("lecture_parts", []))

        prompt = f"""
        You are a Professor of Mathematics and Computer Science.
        Your teaching philosophy is "Bottom-Up" and Constructivist.

        [CURRENT TEACHING STEP]: **{step_title}**
        [CONCEPTS TO COVER]: {', '.join(associated_concepts)}
        [YOUR EXPERT ROLE]: {expert_role}

        [TEXTBOOK EXACT QUOTES (STRICTLY ADHERE - DO NOT HALLUCINATE OUTSIDE OF THIS)]:
        <TEXTBOOK_CORE>
        {verbatim_context}
        </TEXTBOOK_CORE>

        [BOOK LOGICAL FLOW (FROM NEO4J)]:
        <LOGIC_FLOW>
        {dag_context}
        </LOGIC_FLOW>

        [STUDENT'S COGNITIVE ANCHORS (WHAT THEY ALREADY KNOW)]:
        <COGNITIVE_ANCHORS>
        {anchors_text}
        </COGNITIVE_ANCHORS>

        [PREVIOUS LECTURE PARTS BY OTHER EXPERTS]:
        <CONTEXT_CHAIN>
        {previous_parts if previous_parts else "You are the first expert to speak. Introduce the topic."}
        </CONTEXT_CHAIN>

        [YOUR SPECIFIC PROTOCOL & RULE]:
        {specific_rule}

        [MANDATORY TEACHING PRINCIPLES]:
        1. CONSTRUCTIVISM: You MUST use items in <COGNITIVE_ANCHORS> to create analogies.
        2. BIG PICTURE: Use <LOGIC_FLOW> to explain *why* we are learning this.
        3. FOCUS: Explain the [CURRENT TEACHING STEP] using the exact facts from <TEXTBOOK_CORE>. 
        4. MATHEMATICS: Present formulas clearly using LaTeX (enclose in $$).
        5. COHESION: Read <CONTEXT_CHAIN> and seamlessly continue the logic.
        6. TEACHING LANGUAGE: {language}.

        Constraint: Return ONLY your section of the lecture. Start exactly with a Level 3 Markdown Header (###).
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


class ConceptNode(BaseExpertNode):
    def __call__(self, state: dict) -> Dict[str, Any]:
        print(" -> [Concept Expert] is drafting the geometric intuition...")
        if state.get("action_mode") == "NEXT_GROUP":
            prev_idx = state.get("current_seq_index", 0) - 1
            if prev_idx >= 0:
                steps = state.get("entity_groups", [])
                prev_concepts = steps[prev_idx].get("associated_concepts", [])
                for concept in prev_concepts:
                    self.dag.mark_concept_as_learned(concept)

        rule = "HEADER: '### 1. Trực giác hình học & Bản chất'. TASK: Explain the core concepts using real-world metaphors. CONSTRAINT: Do NOT use complex mathematical formulas."
        content = self.generate_expert_content(state, "Concept Intuition Expert", rule)

        existing_parts = state.get("lecture_parts", [])
        return {"lecture_parts": existing_parts + [content]}


class FormulaNode(BaseExpertNode):
    def __call__(self, state: dict) -> Dict[str, Any]:
        print(" -> [Formula Expert] is standardizing mathematical notations...")
        rule = "HEADER: '### 2. Ký hiệu Toán học & Công thức'. TASK: Present the formal theory and mathematical formulas."
        content = self.generate_expert_content(state, "Formal Theory Expert", rule)

        existing_parts = state.get("lecture_parts", [])
        return {"lecture_parts": existing_parts + [content]}


class MathNode(BaseExpertNode):
    def __call__(self, state: dict) -> Dict[str, Any]:
        print(" -> [Math Expert] is deriving the proofs...")
        rule = "HEADER: '### 3. Chứng minh Toán học & Suy luận'. TASK: Provide an in-depth mathematical proof or step-by-step derivation."
        content = self.generate_expert_content(state, "Mathematical Proof Expert", rule)

        existing_parts = state.get("lecture_parts", [])
        return {"lecture_parts": existing_parts + [content]}


class AlgorithmNode(BaseExpertNode):
    def __call__(self, state: dict) -> Dict[str, Any]:
        print(" -> [Algorithm Expert] is designing computational logic...")
        rule = "HEADER: '### 4. Thuật toán & Tư duy Máy tính'. TASK: Translate the mathematical theory into computational logic."
        content = self.generate_expert_content(state, "Algorithm & Software Engineer Expert", rule)

        existing_parts = state.get("lecture_parts", [])
        return {"lecture_parts": existing_parts + [content]}


class ExampleNode(BaseExpertNode):
    def __call__(self, state: dict) -> Dict[str, Any]:
        print(" -> [Application Expert] is finalizing the lecture...")
        steps = state.get("entity_groups", [])
        seq_idx = state.get("current_seq_index", 0)
        step_title = steps[seq_idx].get("step_title", "bước này") if steps else "bước này"

        rule = f"HEADER: '### 5. Ví dụ thực tế & Áp dụng'. TASK: Provide a practical example. CONSTRAINT: At the very end, ask: 'Em đã hiểu hoàn toàn nội dung **{step_title}** chưa để Giáo sư chuyển sang bước tiếp theo?'"
        content = self.generate_expert_content(state, "Application Expert", rule)

        # Tổng hợp toàn bộ bài giảng từ mảng đã được cộng dồn chính xác
        all_parts = state.get("lecture_parts", []) + [content]
        final_lecture = "\n\n---\n\n".join(all_parts)

        return {
            "lecture_parts": all_parts,
            "ai_response": final_lecture,
            "chat_history": [AIMessage(content=final_lecture)]
        }


class RouterNode:
    """Node Điều phối - Nhạc trưởng của hệ thống đa tác tử."""

    def __init__(self, llm_service):
        self.llm_service = llm_service

    def __call__(self, state: dict) -> Dict[str, Any]:
        print("\n[Node: Router] Đang điều phối chuyên gia cho bước này...")
        steps = state.get("entity_groups", [])
        seq_idx = state.get("current_seq_index", 0)

        if not steps or seq_idx >= len(steps):
            return {"routed_experts": ["concept", "formula", "math", "algorithm", "example"]}

        current_step = steps[seq_idx]
        verbatim_context = current_step.get("verbatim_exact_quotes", "")

        chosen_experts = ["concept", "example"]

        dynamic_choices = self.llm_service.decide_experts(verbatim_context)
        chosen_experts.extend(dynamic_choices)

        print(f"    + Danh sách chuyên gia được duyệt: {chosen_experts}")
        return {"routed_experts": chosen_experts, "lecture_parts": []}