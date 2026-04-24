import streamlit as st
import sqlite3
import os
import json
import uuid
import sys
from pathlib import Path

# Lấy đường dẫn tuyệt đối của thư mục chứa file main.py
current_dir = Path(__file__).parent.resolve()

# Thêm đường dẫn này vào sys.path nếu nó chưa có
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from langgraph.checkpoint.sqlite import SqliteSaver
from database.structural_db import QdrantVectorStore  # Đảm bảo file database/qdrant_db.py tồn tại
from database.semantic_dag import SemanticDAG
from orchestrator.llm_service import GeminiLLMService
from orchestrator.graph_builder import LessonOrchestrator
from dotenv import load_dotenv
from core.data_ingestion import run_ingestion_pipeline

load_dotenv()

from config.settings import LLM_MODEL_NAME
# ==========================================
# 1. KHỞI TẠO HỆ THỐNG
# ==========================================
@st.cache_resource
def init_system():
    db = QdrantVectorStore(collection_name="math_curriculum")
    llm = GeminiLLMService(model_name=LLM_MODEL_NAME,temperature=0.3)
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
    st.session_state.thread_id = "session_math_001"
if "current_checkpoint" not in st.session_state:
    st.session_state.current_checkpoint = 1
if "messages" not in st.session_state:
    st.session_state.messages = []
if "language" not in st.session_state:
    st.session_state.language = "English"
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "LOCKED"

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

        # SỬA LỖI 1: Tái sử dụng hàm get_section_exact để kiểm tra xem sách đã nạp chưa
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

    st.markdown("---")
    st.markdown(f"**Trạng thái:** Đang ở Checkpoint {st.session_state.current_checkpoint}")
    st.markdown(f"**Phiên học:** {st.session_state.thread_id}")
    st.markdown("---")
    st.header("📖 Giáo Trình Học Tập")

    # SỬA LỖI 2: Dùng lệnh Scroll của Qdrant thay cho .get() của ChromaDB
    try:
        records, _ = orchestrator.db.client.scroll(
            collection_name=orchestrator.db.collection_name,
            limit=10000,  # Quét tối đa 10k chunks
            with_payload=True,  # Lấy Metadata
            with_vectors=False  # Tắt load Vector để tiết kiệm RAM
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
                            st.session_state.current_checkpoint = 1
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
st.title("Giáo sư AI - Toán học & Giải thuật")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query_to_send = None
action_mode_to_send = None

if st.session_state.get("target_section"):
    col1, col2 = st.columns(2)

    with col1:
        if len(st.session_state.messages) == 0:
            if st.button("Start", use_container_width=True):
                query_to_send = "Hãy bắt đầu giảng bài mục này."
                action_mode_to_send = "LESSON_PROGRESS"
                st.session_state.input_mode = "LOCKED"
        else:
            if st.session_state.current_checkpoint < 3:
                if st.button(f"Continue (Chuyển sang Checkpoint {st.session_state.current_checkpoint + 1})",
                             use_container_width=True):
                    st.session_state.current_checkpoint += 1
                    query_to_send = f"Tôi đã hiểu. Hãy tiếp tục giảng Checkpoint {st.session_state.current_checkpoint}."
                    action_mode_to_send = "LESSON_PROGRESS"
                    st.session_state.input_mode = "LOCKED"
            else:
                st.success("✅ Đã hoàn tất 3 Checkpoint của mục này!")

    with col2:
        if st.session_state.input_mode == "LOCKED":
            if st.button("❓ Mở Hỏi Đáp (Q&A)", use_container_width=True):
                st.session_state.input_mode = "UNLOCKED"
                st.rerun()
        else:
            if st.button("🔒 Đóng Hỏi Đáp", use_container_width=True):
                st.session_state.input_mode = "LOCKED"
                st.rerun()

    if st.session_state.input_mode == "UNLOCKED":
        user_query = st.chat_input("Nhập câu hỏi tự do của bạn...")
        if user_query:
            query_to_send = user_query
            action_mode_to_send = "QA"

# ==========================================
# 5. ĐỘNG CƠ THỰC THI (XỬ LÝ QUERY BẤT KỲ)
# ==========================================
if query_to_send and action_mode_to_send:
    st.session_state.messages.append({"role": "user", "content": query_to_send})
    with st.chat_message("user"):
        st.markdown(query_to_send)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm cơ sở lý thuyết và biên soạn bài giảng..."):
            response = orchestrator.run_lesson(
                query=query_to_send,
                thread_id=st.session_state.thread_id,
                target_chapter=st.session_state.target_file,
                checkpoint=st.session_state.current_checkpoint,
                target_section=st.session_state.target_section,
                action_mode=action_mode_to_send
            )
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()