import streamlit as st
import sqlite3
import os
import json
import uuid
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Lấy đường dẫn tuyệt đối của thư mục chứa file app.py
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from langgraph.checkpoint.sqlite import SqliteSaver
from database.structural_db import QdrantVectorStore
from database.semantic_dag import SemanticDAG
from orchestrator.llm_service import GeminiLLMService
from orchestrator.graph_builder import LessonOrchestrator
from core.data_ingestion import run_ingestion_pipeline

# GIẢ ĐỊNH TRÒ ĐÃ CÓ BIẾN NÀY, NẾU KHÔNG HÃY ĐIỀN TRỰC TIẾP TÊN MODEL VÀO BÊN DƯỚI
from config.settings import LLM_MODEL_NAME


# ==========================================
# 1. KHỞI TẠO HỆ THỐNG
# ==========================================
@st.cache_resource
def init_system():
    db = QdrantVectorStore(collection_name="math_curriculum")
    llm = GeminiLLMService(model_name=LLM_MODEL_NAME, temperature=0.3)
    dag = SemanticDAG(llm_service=llm, vector_store=db)

    conn = sqlite3.connect("memory_checkpoint.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)

    orchestrator = LessonOrchestrator(db, dag, llm, checkpointer=memory)
    return orchestrator


orchestrator = init_system()

# ==========================================
# 2. QUẢN LÝ STATE (TRẠNG THÁI GIAO DIỆN)
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "language" not in st.session_state:
    st.session_state.language = "Tiếng Việt"
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "LOCKED"

# Đã XÓA biến st.session_state.current_checkpoint vì LangGraph giờ tự quản lý nội bộ

# ==========================================
# 3. SIDEBAR - ĐIỀU HƯỚNG & NẠP DỮ LIỆU
# ==========================================
with st.sidebar:
    st.header("⚙️ Cài đặt hệ thống")
    st.selectbox("🌐 Ngôn ngữ giảng dạy:", options=["English", "Tiếng Việt"], key="language")

    st.markdown("---")
    st.header("📂 Nạp Dữ Liệu Học Tập")
    uploaded_file = st.file_uploader("Tải lên tài liệu (.md)", type=["md"])

    if uploaded_file is not None:
        file_name = uploaded_file.name
        existing_docs = orchestrator.db.get_section_exact(target_file=file_name, target_section="")

        if existing_docs and len(existing_docs) > 0:
            st.success(f"✅ '{file_name}' đã có sẵn trong CSDL.")
        else:
            st.warning(f"⚠️ '{file_name}' chưa có trong CSDL.")
            if st.button(f"📥 Bắt đầu nạp '{file_name}'", use_container_width=True):
                with st.spinner("Đang trích xuất Khung xương và Đồ thị..."):
                    content = uploaded_file.read().decode("utf-8")
                    run_ingestion_pipeline(content, file_name, orchestrator.db, orchestrator.llm, orchestrator.dag)
                    st.success("🎉 Nạp dữ liệu thành công!")

                    import time
                    time.sleep(1)
                    st.rerun()
    st.markdown("---")
    st.markdown(f"**Phiên học:** {st.session_state.thread_id}")
    st.markdown("---")
    st.header("📖 Giáo Trình Học Tập")

    try:
        records, _ = orchestrator.db.client.scroll(
            collection_name=orchestrator.db.parent_coll,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        unique_sources = list(set([r.payload.get("source") for r in records if r.payload and "source" in r.payload]))
    except Exception as e:

        unique_sources = []

    if unique_sources:
        selected_file = st.selectbox("📚 Môn học:", options=unique_sources, key="target_file")
        toc_path = f"database/tocs/{selected_file}_toc.json"

        if os.path.exists(toc_path):
            with open(toc_path, "r", encoding="utf-8") as f:
                toc_tree = json.load(f)

            st.markdown("### 🗂️ Mục lục")
            if "target_section" not in st.session_state:
                st.session_state.target_section = ""

            for chapter, sections in toc_tree.items():
                with st.expander(f"📂 {chapter}"):
                    for sec in sections:
                        if st.button(f"📄 {sec}", key=f"btn_{selected_file}_{sec}"):
                            st.session_state.target_section = sec
                            st.session_state.messages = []
                            st.session_state.input_mode = "LOCKED"
                            st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"
                            st.rerun()

            if st.session_state.target_section:
                st.success(f"**🎯 Đang học:** {st.session_state.target_section}")
            else:
                st.info("👈 Hãy chọn một mục để bắt đầu!")
    else:
        st.warning("CSDL đang trống. Vui lòng nạp file.")
        st.session_state.target_file = ""

# ==========================================
# 4. KHU VỰC HIỂN THỊ CHÍNH (MAIN CHAT)
# ==========================================
st.title("Giáo sư AI - Chuyên gia Đa Hệ")
st.caption("Kiến trúc Multi-Agent Pipeline: Tự động tổng hợp trực giác, toán học và thuật toán.")

# Render lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query_to_send = None
action_mode_to_send = None

# Vùng Điều khiển Nút bấm
if st.session_state.get("target_section"):
    col1, col2 = st.columns(2)

    with col1:
        if len(st.session_state.messages) == 0:
            # Nút bắt đầu (Sẽ kích hoạt PlannerNode)
            if st.button("🚀 Bắt đầu lập kế hoạch", use_container_width=True):
                query_to_send = "Hãy bắt đầu bài học."
                action_mode_to_send = "START_LESSON"
                st.session_state.input_mode = "LOCKED"
        else:
            # Nút đi tiếp (Sẽ kích hoạt chuỗi 5 Chuyên gia)
            if st.button("✅ Đã hiểu, tiếp tục phần tiếp theo", use_container_width=True):
                query_to_send = "Em đã hiểu phần này, sẵn sàng tiếp tục!"
                action_mode_to_send = "NEXT_GROUP"
                st.session_state.input_mode = "LOCKED"

    with col2:
        if st.session_state.input_mode == "LOCKED":
            if st.button("❓ Mở Hỏi Đáp (Q&A)", use_container_width=True):
                st.session_state.input_mode = "UNLOCKED"
                st.rerun()
        else:
            if st.button("🔒 Đóng Hỏi Đáp", use_container_width=True):
                st.session_state.input_mode = "LOCKED"
                st.rerun()

    # Vùng nhập Text tự do
    if st.session_state.input_mode == "UNLOCKED":
        user_query = st.chat_input("Nhập câu hỏi tự do của bạn để hỏi Giáo sư...")
        if user_query:
            query_to_send = user_query
            action_mode_to_send = "QA"

# ... (Phần UI phía trên giữ nguyên) ...

# Thêm nút Toggle bật Developer Mode ở thanh Sidebar
with st.sidebar:
    st.markdown("---")
    st.header("🛠️ Công cụ Phát triển")
    dev_mode = st.toggle("🐞 Bật Developer Mode (Streaming Log)")

# ==========================================
# 5. ĐỘNG CƠ THỰC THI (XỬ LÝ QUERY BẤT KỲ)
# ==========================================
if query_to_send and action_mode_to_send:
    st.session_state.messages.append({"role": "user", "content": query_to_send})
    with st.chat_message("user"):
        st.markdown(query_to_send)

    with st.chat_message("assistant"):
        if dev_mode:
            # -----------------------------------------------------
            # CHẾ ĐỘ STREAMING (KIỂM SOÁT TỪNG BƯỚC)
            # -----------------------------------------------------
            status = st.status("🔍 Đang kích hoạt chuỗi tác tử...", expanded=True)
            final_response = ""

            # Gọi hàm stream_lesson thay vì run_lesson
            stream_generator = orchestrator.stream_lesson(
                query=query_to_send, thread_id=st.session_state.thread_id,
                target_chapter=st.session_state.target_file, target_section=st.session_state.target_section,
                action_mode=action_mode_to_send
            )

            # Lặp qua từng Event mà LangGraph nhả ra
            for step in stream_generator:
                node_name = step.get("node", "unknown")

                # In thông báo hệ thống (như việc kéo DB)
                if node_name == "system":
                    status.write(step["message"])

                # In trạng thái của từng Chuyên gia khi họ nộp bài
                elif node_name in ["concept", "formula", "math", "algorithm", "example"]:
                    status.write(f"✅ **[Chuyên gia {node_name.capitalize()}]** đã hoàn tất biên soạn.")

                    # Nếu là node cuối cùng, lấy bài giảng ra
                    if node_name == "example":
                        final_response = step["state_update"].get("ai_response", "")

                elif node_name == "planner":
                    status.write(f"✅ **[Kế hoạch sư]** đã trích xuất lộ trình.")
                    final_response = step["state_update"].get("ai_response", "")

            # Khi vòng lặp stream kết thúc
            status.update(label="Hoàn tất siêu bài giảng!", state="complete", expanded=False)

            # Hiển thị bài giảng ra màn hình chính
            if final_response:
                st.markdown(final_response)
            else:
                st.warning("Không lấy được kết quả cuối cùng từ Pipeline.")

        else:
            # -----------------------------------------------------
            # CHẾ ĐỘ NGƯỜI DÙNG BÌNH THƯỜNG (CHỜ ĐỢI HỘP ĐEN)
            # -----------------------------------------------------
            with st.spinner("⏳ Hội đồng 5 Chuyên gia đang cùng biên soạn bài giảng..."):
                final_response = orchestrator.run_lesson(
                    query=query_to_send, thread_id=st.session_state.thread_id,
                    target_chapter=st.session_state.target_file, target_section=st.session_state.target_section,
                    action_mode=action_mode_to_send
                )
                st.markdown(final_response)

    # Lưu lại lịch sử
    if final_response:
        st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.rerun()