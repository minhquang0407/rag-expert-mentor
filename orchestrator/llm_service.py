import os
import json
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


# ==========================================
# ĐỊNH NGHĨA SCHEMAS (ROADMAP-DRIVEN)
# ==========================================
class TeachingStep(BaseModel):
    seq_id: int = Field(description="Số thứ tự của bước dạy học")
    step_title: str = Field(description="Tiêu đề của bước (Ví dụ: Giới thiệu các loại Đồ thị)")
    associated_concepts: List[str] = Field(
        description="Danh sách các THỰC THỂ LÕI TRONG ĐỒ THỊ có liên quan đến bước này. Dùng để hệ thống truy vết Neo4j.")
    verbatim_exact_quotes: str = Field(
        description="KHÔNG SINH TEXT MỚI. Hãy copy-paste nguyên văn (chứa LaTeX) các đoạn lý thuyết trong sách gốc tương ứng với bước dạy này để AI làm tài liệu.")


class GraphTriplet(BaseModel):
    source: str
    relation: Literal["PREREQUISITE_OF", "RELATES_TO", "PART_OF", "DESCRIBES"] = Field(
        description="PREREQUISITE_OF: Phải học trước. PART_OF: Thành phần. DESCRIBES: Mô tả/Công thức. RELATES_TO: Liên quan."
    )
    target: str
    weight: int = Field(ge=1, le=10, description="Độ liên quan từ 1 đến 10")


class SectionExtraction(BaseModel):
    teaching_roadmap: List[TeachingStep] = Field(description="Lộ trình giảng dạy cho Section này.")
    graph_triplets: List[GraphTriplet] = Field(max_length=15,
                                               description="Đồ thị tri thức (DAG) trích xuất từ văn bản.")


class QuestionList(BaseModel):
    questions: List[str]


class RerankResult(BaseModel):
    best_parent_id: str


# ==========================================
# DỊCH VỤ LLM
# ==========================================
class GeminiLLMService:
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.1):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.environ.get("LLM_API_KEY"),
            max_output_tokens=8192
        )

    def extract_section_curriculum_and_dag(self, section_text: str) -> Dict[str, Any]:
        """
        - Lí do tại sao dùng: Trích xuất Lộ trình giảng dạy (Teaching Roadmap) và Đồ thị (DAG) trong cùng 1 lần gọi để tối ưu API.
        - Chức năng: Đọc section_text và ép LLM trả về JSON cấu trúc nghiêm ngặt (Structured Outputs).
        - Cách dùng: Gọi trong pipeline nạp dữ liệu (data_ingestion.py).
        - Tham số: section_text (str) - Nội dung toàn bộ của một Section/Bài học.
        - Trả về: Dictionary chứa mảng 'teaching_roadmap' và 'graph_triplets'.
        """
        prompt = f"""
        Nhiệm vụ: Phân tích sâu [SECTION VĂN BẢN] để xây dựng Lộ trình giảng dạy (Roadmap) và Đồ thị Tri thức.
        Vai trò: Bạn là một Chuyên gia Toán học và Kỹ sư Dữ liệu vô cảm.

        RÀNG BUỘC TUYỆT ĐỐI (ANTI-HALLUCINATION):
        1. Về Lộ trình giảng dạy (teaching_roadmap):
           - Hãy chia Section này thành các bước dạy học logic (Step 1, Step 2...). 
           - KHÔNG tự sinh giải thích. Ở trường "verbatim_exact_quotes", bạn PHẢI trích xuất COPY-PASTE NGUYÊN VĂN các câu/đoạn văn bản chứa công thức LaTeX từ sách gốc tương ứng với Bước đó.
        2. Về Đồ thị (graph_triplets):
           - Xây dựng mạng lưới các khái niệm liên kết với nhau.
           - CẤM tự bịa ra các khái niệm toán học không có trong văn bản (VD: "Phép cộng").
           - Tuân thủ 4 loại quan hệ: PREREQUISITE_OF, RELATES_TO, PART_OF, DESCRIBES.

        [SECTION VĂN BẢN]:
        {section_text}
        """
        try:
            structured_llm = self.llm.with_structured_output(SectionExtraction)
            result = structured_llm.invoke([HumanMessage(content=prompt)])
            return result.model_dump() if hasattr(result, 'model_dump') else result.dict()
        except Exception as e:
            print(f"[!] Lỗi Structured Extraction: {e}")
            return {"teaching_roadmap": [], "graph_triplets": []}

    def generate_hypothetical_questions(self, section_text: str, num_questions: int = 5) -> List[str]:
        """
        - Lí do tại sao dùng: Sinh câu hỏi giả định để phục vụ truy xuất Hybrid Search (BM25 + Dense).
        - Chức năng: Đóng vai học sinh tự đặt câu hỏi về đoạn text vừa đọc.
        - Trả về: Danh sách chuỗi (List of strings).
        """
        prompt = f"""
        Context: You are a Professor. Task: Generate EXACTLY {num_questions} FAQ questions for this text.
        Constraints: Return ONLY a valid JSON array of strings. Double-escape LaTeX backslashes.
        [TEXT]: {section_text}
        """
        try:
            # Bọc prompt trong HumanMessage
            result = self.llm.with_structured_output(QuestionList).invoke([HumanMessage(content=prompt)])
            return result.questions
        except Exception as e:
            print(f"[!] Lỗi sinh câu hỏi giả định: {e}")
            return []

    def rerank_candidate_questions(self, user_query: str, candidates: List[Dict[str, str]]) -> str:
        """
        - Lí do tại sao dùng: Sắp xếp lại (Reranking) các kết quả từ Vector DB dựa trên sức mạnh của LLM.
        - Chức năng: Tìm ra câu hỏi giả định nào sát nghĩa với câu hỏi của User nhất.
        - Trả về: ID (best_parent_id) của chunk chứa câu trả lời tốt nhất.
        """
        candidates_str = json.dumps(candidates, ensure_ascii=False, indent=2)
        prompt = f"""
        User Query: "{user_query}"\nCandidates:\n{candidates_str}\n
        Identify WHICH candidate question has the exact same semantic intent.
        """
        try:
            # Bọc prompt trong HumanMessage
            result = self.llm.with_structured_output(RerankResult).invoke([HumanMessage(content=prompt)])
            return result.best_parent_id
        except Exception as e:
            print(f"[!] Lỗi Rerank: {e}")
            if candidates: return candidates[0].get("parent_id", "")
            return ""