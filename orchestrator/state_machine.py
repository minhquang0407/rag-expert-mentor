from typing import TypedDict, List, Annotated, Dict, Any
import operator
from langchain_core.messages import BaseMessage
def add_messages(left: list, right: list):
    """Hàm hỗ trợ cộng dồn danh sách tin nhắn cho lịch sử chat."""
    return left + right
class LessonState(TypedDict):
    """
    - Lí do tại sao dùng: Định nghĩa cấu trúc Máy trạng thái toàn cục cho LangGraph.
    - Chức năng: Lưu trữ câu hỏi, lịch sử hội thoại có cộng dồn (reducer), và lộ trình giáo án thực thể.
    - Cách dùng: Được khởi tạo ở Router và truyền vào tham số của hàm `__call__` trong mọi Node.
    """
    # ==========================================
    # 1. ĐỊNH DANH & VỊ TRÍ BÀI HỌC
    # ==========================================
    student_query: str  # Câu hỏi hiện tại của sinh viên
    target_file: str  # Tên file PDF/Sách đang học
    target_section: str  # Mục lục đang học (VD: "Section 1.1")
    current_checkpoint: int  # Checkpoint hiện tại (1: Bản chất, 2: Toán học, 3: Q&A)
    action_mode: str  # Phân luồng từ UI (LESSON_PROGRESS hoặc QA)
    language: str  # Ngôn ngữ giảng dạy (English, Tiếng Việt)

    # ==========================================
    # 2. ĐIỀU HƯỚNG THỰC THỂ (ENTITY-BASED)
    # ==========================================
    is_planning_phase: bool  # Cờ hiệu: True = Chờ in Kế hoạch, False = Đang giảng bài
    entity_groups: List[Dict[str, Any]]  # Mảng JSON chứa các Cụm Thực thể và Verbatim Text
    current_seq_index: int  # Con trỏ duyệt qua entity_groups (bắt đầu từ 0)

    # ==========================================
    # 3. LỊCH SỬ & ĐÁNH GIÁ (STATEFUL MEMORY)
    # ==========================================
    # Sử dụng operator.add để LangGraph tự động nối (append) tin nhắn mới thay vì ghi đè
    chat_history: Annotated[List[BaseMessage], add_messages]
    lecture_parts: List[str]

    ai_response: str  # Câu trả lời của Giáo sư AI chuẩn bị in ra màn hình
    assessment_result: Dict[str, Any]  # Trạng thái đánh giá từ Evaluator Agent (Pass/Fail)

    # Thêm vào bên dưới cùng của class LessonState
    routed_experts: List[str]