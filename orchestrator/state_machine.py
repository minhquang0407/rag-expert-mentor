from typing import TypedDict, List, Annotated, Dict, Any
import operator


class LessonState(TypedDict):
    """
    Cấu trúc dữ liệu toàn cục chạy xuyên suốt phiên làm việc của LangGraph.
    Mọi Node (Teacher, Evaluator) đều chỉ đọc và ghi vào đây.
    """
    # 1. Định danh & Vị trí bài học
    student_query: str  # Câu hỏi hiện tại của sinh viên
    target_chapter: str  # Chương đang học (ví dụ: "Chương 1")
    target_file: str
    current_checkpoint: int  # 1: Trực giác hình học, 2: Toán học, 3: Code
    action_mode: str
    # 2. Ngữ cảnh RAG (Được nhồi vào từ Tầng 1)
    structural_context: str  # Văn bản kéo lên từ ChromaDB
    dag_context: str  # First Principles kéo lên từ NetworkX

    # 3. Lịch sử & Đánh giá
    # Sử dụng operator.add để LangGraph tự động nối (append) tin nhắn mới thay vì ghi đè
    chat_history: Annotated[List[Dict[str, str]], operator.add]

    # Trạng thái đánh giá từ Evaluator Agent (Pass/Fail)
    assessment_result: Dict[str, Any]

    # Language
    language: str