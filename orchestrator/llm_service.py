import json
import os
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
class GeminiLLMService:
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.1):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.environ.get("LLM_API_KEY"),
            max_output_tokens=8192 # Tăng limit token vì 1 Section xuất ra mảng JSON sẽ khá dài
        )

    def extract_graph_entities(self, section_text: str, glossary: List[str] = None, max_retries: int = 3) -> List[Dict[str, str]]:
        """
        Lí do tại sao dùng: Chuyển đổi từ Chunk-level sang Section-level Extraction.
        Chức năng: Trích xuất thực thể, mối quan hệ và ĐÁNH GIÁ TRỌNG SỐ (khoảng cách ngữ nghĩa).
        Cách dùng: Gọi 1 lần duy nhất cho mỗi Section.
        Tham số:
            - section_text: Toàn bộ văn bản của 1 Section (đã được gộp từ nhiều chunks).
            - glossary: Từ điển các thực thể đã biết từ các Section/Chapter trước.
        """
        glossary_text = ", ".join(glossary) if glossary else "Chưa có (Đây là phần mở đầu của sách)"

        prompt = f"""
        Context: You are a Mathematical Knowledge Graph Extraction System.
        Task: Extract a COMPREHENSIVE knowledge graph from the ENTIRE SECTION text provided below.

        [CORE GLOSSARY FROM PREVIOUS SECTIONS]:
        {glossary_text}

        Constraints: 
        1. Return ONLY a valid JSON array.
        2. JSON structure: [{{"source": "Concept A", "target": "Concept B", "relation": "Relationship", "weight": integer}}].
        3. CORE GLOSSARY RULE: If an extracted concept is identical or highly similar to an entity in the [CORE GLOSSARY], you MUST use the EXACT STRING from the glossary to maintain graph connectivity.
        4. WEIGHT RULE (DISTANCE): "weight" represents semantic distance. 
           - 1 to 3: Highly intimate/fundamental connection (e.g., Matrix is the core representation of Graph).
           - 4 to 6: Moderate connection.
           - 7 to 10: Loose or indirect connection.
        5. Absolutely no explanations or Markdown formatting.

        [Full Section Text Input]:
        {section_text}
        """

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                raw_content = response.content.strip()

                start_idx = raw_content.find('[')
                end_idx = raw_content.rfind(']')

                if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
                    clean_str = raw_content[start_idx:end_idx + 1]
                    return json.loads(clean_str)
                else:
                    print(f" -> [Bỏ qua] Section này không có cấu trúc đồ thị phù hợp.")
                    return []

            except json.JSONDecodeError as e:
                import time
                print(f"[!] Lỗi Parse JSON (Lần thử {attempt + 1}/{max_retries}). Thử lại...")
                time.sleep(2)
            except Exception as e:
                import time
                print(f"[!] API CONNECT ERROR: {e}")
                time.sleep(2)

        return []

    def extract_section_curriculum_and_dag(self, section_text: str) -> Dict[str, Any]:
        """
        - Lí do tại sao dùng: Thay thế cho hàm extract_graph_entities cũ. Tối ưu hóa API Call bằng cách bắt LLM làm 3 việc cùng lúc: Gom cụm thực thể (Clustering), Trích xuất nguyên văn (Verbatim Extraction), và Xây đồ thị (DAG extraction).
        - Chức năng: Phân tích một Section văn bản, tìm ra các khái niệm cốt lõi, gom chúng thành nhóm (group), sao chép chính xác nguyên văn lý thuyết, và thiết lập các cạnh (triplets) cho NetworkX.
        - Cách dùng: Gọi TỪ MỘT LUỒNG NẠP DỮ LIỆU (Ingestion Pipeline) offline trước khi cho sinh viên học.
        - Tham số:
            - `section_text` (str): Toàn bộ văn bản Markdown của một Section.
        - Trả về, Kiểu trả về: `Dict[str, Any]` - Một từ điển JSON chứa 2 khóa chính: 'curriculum_groups' và 'graph_triplets'.
        - Các hàm thay thế nếu có: Có thể dùng LangChain Extraction Chain kết hợp Pydantic, nhưng dùng JSON Prompt thô (Raw Prompt) với Gemini cho tốc độ và khả năng tuân thủ cấu trúc tốt nhất đối với mã LaTeX.
        """
        prompt = f"""
         Nhiệm vụ: Phân tích sâu [SECTION VĂN BẢN] và trích xuất cấu trúc dữ liệu JSON KÉP.
         Vai trò: Bạn là một Chuyên gia Cấu trúc Dữ liệu Toán học vô cảm.

         RÀNG BUỘC TUYỆT ĐỐI (NẾU VI PHẠM SẼ LÀM SẬP HỆ THỐNG):
         1. Phần "verbatim_text": TUYỆT ĐỐI KHÔNG diễn giải, tóm tắt, hay thay đổi bất kỳ ký tự nào. Phải COPY PASTE chính xác 100% nguyên văn các câu lý thuyết, công thức LaTeX từ văn bản gốc xung quanh thực thể đó.
         2. Đầu ra CHỈ LÀ MỘT OBJECT JSON HỢP LỆ. Không chứa dấu markdown ```json.

         CẤU TRÚC JSON YÊU CẦU:
         {{
           "curriculum_groups": [
             {{
               "seq_id": 1,
               "group_name": "Tên nhóm kiến thức (gom cụm các thực thể liên quan)",
               "core_entities": ["Thực thể A", "Thực thể B"],
               "verbatim_text": "Toàn bộ đoạn văn bản COPY NGUYÊN GỐC chứa định nghĩa và công thức của nhóm này."
             }}
           ],
           "graph_triplets": [
             {{"source": "Thực thể A", "relation": "quan hệ gì", "target": "Thực thể B", "weight": 2}}
           ]
         }}

         [SECTION VĂN BẢN]:
         {section_text}
         """

        try:
            # Gọi LLM (Cú pháp tùy thuộc vào thư viện trò đang dùng, đây là cú pháp giả định chuẩn)
            response = self.llm.invoke(prompt)
            raw_content = response.content.strip()

            # Xử lý an toàn: Tìm block JSON thực sự để tránh lỗi parse
            start_idx = raw_content.find('{')
            end_idx = raw_content.rfind('}')

            if start_idx != -1 and end_idx != -1:
                json_str = raw_content[start_idx:end_idx + 1]
                data = json.loads(json_str)
                return data
            else:
                raise ValueError("Không tìm thấy cấu trúc JSON trong phản hồi.")

        except Exception as e:
            print(f"[!] Lỗi Single-Pass Extraction: {e}")
            # Trả về cấu trúc rỗng chuẩn xác để hệ thống không bị crash (Graceful degradation)
            return {"curriculum_groups": [], "graph_triplets": []}
    def generate_hypothetical_questions(self, section_text: str, num_questions: int = 5) -> List[str]:
        prompt = f"""
        Context: You are a Professor formulating examination and FAQ questions.
        Task: Read the provided TEXTBOOK SECTION and generate EXACTLY {num_questions} questions that this section perfectly answers.

        Constraints:
        1. Return ONLY a valid JSON array of strings: ["question 1", "question 2", ...].
        2. The questions must cover the core definitions, theorems, and practical applications in the text.
        3. Do NOT include answers. No markdown outside the JSON array.

        [TEXTBOOK SECTION]:
        {section_text}
        """
        try:
            response = self.llm.invoke(prompt)
            raw_content = response.content.strip()
            start_idx = raw_content.find('[')
            end_idx = raw_content.rfind(']')
            if start_idx != -1 and end_idx != -1:
                return json.loads(raw_content[start_idx:end_idx + 1])
        except Exception as e:
            print(f"[!] Lỗi sinh câu hỏi: {e}")
        return []

    def rerank_candidate_questions(self, user_query: str, candidates: List[Dict[str, str]]) -> str:
        candidates_str = json.dumps(candidates, ensure_ascii=False, indent=2)

        prompt = f"""
        Task: Reranking Semantic Similarity.
        User Query: "{user_query}"

        Candidate Questions:
        {candidates_str}

        Identify WHICH candidate question has the exact same semantic intent as the User Query.
        Constraint: 
        1. Return ONLY a valid JSON object: {{"best_parent_id": "the_parent_id_here"}}.
        2. If NO candidate is remotely similar to the user query, return {{"best_parent_id": ""}}.
        3. Do not explain.
        """
        try:
            response = self.llm.invoke(prompt)
            raw_content = response.content.strip()
            start_idx = raw_content.find('{')
            end_idx = raw_content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                result = json.loads(raw_content[start_idx:end_idx + 1])
                return result.get("best_parent_id", "")
        except Exception as e:
            print(f"[!] Lỗi Rerank: {e}")
            # Fallback: Trả về ứng viên đầu tiên nếu LLM lỗi Parse
            if candidates: return candidates[0].get("parent_id", "")
        return ""