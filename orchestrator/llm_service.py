import json
import os
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


class GeminiLLMService:
    def __init__(self, model_name: str = "gemini-1.5-pro", temperature: float = 0.1):
        """Khởi tạo kết nối API với nhiệt độ thấp để đảm bảo tính chính xác Toán học."""
        # Yêu cầu biến môi trường GOOGLE_API_KEY đã được thiết lập
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=2048
        )

    def extract_graph_entities(self, text_chunk: str) -> List[Dict[str, str]]:
        """
        Lí do tại sao dùng: Để chuyển đổi văn bản phi cấu trúc thành Đồ thị DAG.
        Chức năng: Trích xuất các thực thể Toán học và Mối quan hệ.
        Cách dùng: Gọi trong vòng lặp của khâu Data Ingestion.
        Tham số: text_chunk (str) - Đoạn văn bản cần trích xuất.
        Trả về: List[Dict] - Mảng các đối tượng chứa source, target, relation.
        """
        prompt = f"""
        Context: You are a Mathematical Knowledge Graph Extraction System.
        Task: Extract mathematical concepts and their relationships from the provided input text.
        Constraints: 
        1. Return ONLY a valid JSON array.
        2. JSON structure: [{"source": "Concept A", "target": "Concept B", "relation": "Relationship"}].
        3. Absolutely no explanations or Markdown formatting (no ```json blocks).
        Format: [{"source": "string", "target": "string", "relation": "string"}]
        [Text Input]:
        {text_chunk}
        """
        try:
            response = self.llm.invoke(prompt)
            clean_str = response.content.strip().replace('```json', '').replace('```', '')
            return json.loads(clean_str)
        except Exception as e:
            print(f"[!] Lỗi trích xuất LLM: {e}")
            return []

    def summarize_community(self, community_nodes: List[str]) -> str:
        """
        Description: Compressing Graph Information to reduce Context Window.
        Feature: Writing a Summary of First Principles for an Entity Community
        Use: Call after Leiden algorithm completely parsing community.
        parameter: community_nodes (List[str]) - List of Community Nodes.
        return: str - An academy summary text.
        """
        prompt = f"""
        As a Professor of Mathematics, write a concise summary (3-4 sentences) 
        explaining the geometric essence and logical interconnection of the following set of concepts
        {community_nodes}
        Requirement: Use profound academic language.
        """
        return self.llm.invoke(prompt).content

    def generate_teacher_response(self, context: str, user_query: str, current_checkpoint: int,
                                  project_name: str) -> str:
        """
        Lí do tại sao dùng: Đóng vai trò là Não bộ điều phối của Teacher Agent.
        Chức năng: Ép LLM giảng dạy theo đúng Checkpoint và không rò rỉ dữ liệu.
        Cách dùng: Gọi từ bên trong `teacher_node` của LangGraph.
        Tham số:
            - context: Lý thuyết từ DAG và ChromaDB.
            - user_query: Câu hỏi của sinh viên.
            - current_checkpoint: Trạng thái bài học hiện tại (1, 2, hoặc 3).
        Trả về: str - Bài giảng chuẩn định dạng.
        """
        system_prompt = f"""
        You are an Expert Mentor in Computer Science and Applied Mathematics.
        Your teaching philosophy is strictly "Bottom-Up & Project-Based".

        [KNOWLEDGE_BASE - STRICT RAG CONTEXT]:
        {context}

        [CURRENT PROJECT]: {project_name}

        [EXECUTION PROTOCOL]:
        You MUST execute ONLY Checkpoint {current_checkpoint}:
        - IF Checkpoint 1: Explain Geometric Intuition and Essence. End with a logical question.
        - IF Checkpoint 2: Provide Mathematical Derivation (use LaTeX). End with a calculation exercise.
        - IF Checkpoint 3: Guide Big-O analysis for the project.

        CRITICAL RULE: Rely ONLY on the [KNOWLEDGE_BASE]. If information is missing, explicitly state: "Tôi không biết, tôi không có thông tin về nó." Do not hallucinate.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]

        response = self.llm.invoke(messages)
        return response.content