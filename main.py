import streamlit as st
import sqlite3
import os
import json
import uuid
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from langgraph.checkpoint.sqlite import SqliteSaver
from database.structural_db import QdrantVectorStore
from database.semantic_dag import Neo4jManager
from orchestrator.llm_service import GeminiLLMService
from orchestrator.graph_builder import LessonOrchestrator
from core.data_ingestion import run_ingestion_pipeline

from config.settings import LLM_MODEL_NAME


# ==========================================
# 1. KHỞI TẠO HỆ THỐNG
# ==========================================
@st.cache_resource
def init_system():
    db = QdrantVectorStore(collection_name="math_curriculum")
    llm = GeminiLLMService(model_name=LLM_MODEL_NAME, temperature=0.3)
    dag = Neo4jManager()
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

# Thêm 2 cờ State để điều khiển 3 bước nút bấm
if "plan_generated" not in st.session_state:
    st.session_state.plan_generated = False
if "lesson_started" not in st.session_state:
    st.session_state.lesson_started = False

# ==========================================
# 3. SIDEBAR - ĐIỀU HƯỚNG & NẠP DỮ LIỆU
# ==========================================
with st.sidebar:
    st.header("⚙️ Cài đặt hệ thống")
    st.selectbox("🌐 Ngôn ngữ giảng dạy:", options=["English", "Tiếng Việt"], key="language")
    st.markdown("---")
    st.header("📂 Nạp Dữ Liệu Học Tập")

    uploaded_file = st.file_uploader("Tải lên tài liệu (.md)", type=["md"])

    unique_sources = []
    try:
        records, _ = orchestrator.db.client.scroll(
            collection_name=orchestrator.db.parent_coll, limit=500, with_payload=True, with_vectors=False
        )
        unique_sources = list(set([r.payload.get("source") for r in records if r.payload and r.payload.get("source")]))
    except Exception as e:
        st.error(f"❌ Lỗi quét Qdrant: {e}")

    if uploaded_file is not None:
        file_name = uploaded_file.name
        if file_name in unique_sources:
            st.success(f"✅ '{file_name}' đã có sẵn trong CSDL.")
        else:
            st.warning(f"⚠️ '{file_name}' chưa có trong CSDL.")
            if st.button(f"📥 Bắt đầu nạp '{file_name}'", use_container_width=True):
                with st.spinner("Đang trích xuất Khung xương và Đồ thị..."):
                    content = uploaded_file.read().decode("utf-8")
                    run_ingestion_pipeline(content, file_name, orchestrator.db, orchestrator.llm, orchestrator.dag)
                    st.success("🎉 Nạp dữ liệu thành công! Đang tải lại giao diện...")
                    time.sleep(1)
                    st.rerun()

    st.markdown("---")
    st.markdown(f"**Phiên học:** {st.session_state.thread_id}")
    st.markdown("---")
    st.header("📖 Giáo Trình Học Tập")

    if unique_sources:
        selected_file = st.selectbox("📚 Môn học:", options=unique_sources, key="target_file")
        toc_path = os.path.join(current_dir, "database", "tocs", f"{selected_file}_toc.json")

        if os.path.exists(toc_path):
            with open(toc_path, "r", encoding="utf-8") as f:
                try:
                    toc_tree = json.load(f)
                except json.JSONDecodeError:
                    st.error("❌ File Mục lục bị lỗi định dạng JSON. Hãy nạp lại file.")
                    toc_tree = {}

            if toc_tree:
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
                                # ĐẶT LẠI CÁC CỜ TRẠNG THÁI KHI CHUYỂN BÀI MỚI
                                st.session_state.plan_generated = False
                                st.session_state.lesson_started = False
                                st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"
                                st.rerun()

                if st.session_state.target_section:
                    st.success(f"**🎯 Đang học:** {st.session_state.target_section}")
                else:
                    st.info("👈 Hãy chọn một mục để bắt đầu!")
            else:
                st.warning("⚠️ File Mục lục tồn tại nhưng nội dung trống rỗng.")
        else:
            st.error(f"⚠️ Dữ liệu Vector đã có, nhưng file JSON Mục lục bị mất tại: {toc_path}")
    else:
        st.warning("CSDL đang trống. Vui lòng tải file lên.")

    st.markdown("---")
    st.header("🛠️ Công cụ Phát triển")
    dev_mode = st.toggle("🐞 Bật Developer Mode (Streaming Log)")

# ==========================================
# 4. KHU VỰC HIỂN THỊ CHÍNH (MAIN CHAT)
# ==========================================
st.title("Giáo sư AI - Chuyên gia Đa Hệ")
st.caption("Kiến trúc Multi-Agent Pipeline: Tích hợp Vector Qdrant và GraphRAG Neo4j.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query_to_send = None
action_mode_to_send = None

if st.session_state.get("target_section"):
    col1, col2 = st.columns(2)

    with col1:
        # LUỒNG 3 BƯỚC MỚI ĐỂ TRÁNH LỖI NHẢY CÓC INDEX
        if not st.session_state.plan_generated:
            if st.button("🚀 Bắt đầu lập kế hoạch", use_container_width=True):
                query_to_send = "Hãy lập lộ trình bài học."
                action_mode_to_send = "START_LESSON"
                st.session_state.plan_generated = True
                st.session_state.input_mode = "LOCKED"
        elif not st.session_state.lesson_started:
            if st.button("▶️ Vào học Phần 1", use_container_width=True):
                query_to_send = "Em đã xem xong kế hoạch, sẵn sàng vào học Phần 1!"
                action_mode_to_send = "START_PIPELINE"  # Mã lệnh mới: Ép dạy index 0, không tăng index
                st.session_state.lesson_started = True
                st.session_state.input_mode = "LOCKED"
        else:
            if st.button("✅ Đã hiểu, tiếp tục", use_container_width=True):
                query_to_send = "Em đã hiểu phần này, sẵn sàng tiếp tục!"
                action_mode_to_send = "NEXT_GROUP"  # Cứ nhấn nút này là Index + 1
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

    if st.session_state.input_mode == "UNLOCKED":
        user_query = st.chat_input("Nhập câu hỏi tự do của bạn...")
        if user_query:
            query_to_send = user_query
            action_mode_to_send = "QA"

# ==========================================
# 5. ĐỘNG CƠ THỰC THI
# ==========================================
if query_to_send and action_mode_to_send:
    st.session_state.messages.append({"role": "user", "content": query_to_send})
    with st.chat_message("user"):
        st.markdown(query_to_send)

    with st.chat_message("assistant"):
        if dev_mode:
            status = st.status("🔍 Đang kích hoạt chuỗi tác tử...", expanded=True)
            final_response = ""

            stream_generator = orchestrator.stream_lesson(
                query=query_to_send, thread_id=st.session_state.thread_id,
                target_chapter=st.session_state.target_file, target_section=st.session_state.target_section,
                action_mode=action_mode_to_send
            )

            for step in stream_generator:
                node_name = step.get("node", "unknown")
                if node_name == "system":
                    status.write(step["message"])
                elif node_name in ["concept", "formula", "math", "algorithm", "example"]:
                    status.write(f"✅ **[Chuyên gia {node_name.capitalize()}]** đã hoàn tất biên soạn.")
                    if node_name == "example":
                        final_response = step["state_update"].get("ai_response", "")
                elif node_name == "planner":
                    status.write(f"✅ **[Kế hoạch sư]** đã trích xuất lộ trình.")
                    final_response = step["state_update"].get("ai_response", "")
                elif node_name == "finish":
                    final_response = step["message"]

            status.update(label="Hoàn tất!", state="complete", expanded=False)
            if final_response:
                st.markdown(final_response)
            else:
                st.warning("Không lấy được kết quả cuối cùng.")
        else:
            with st.spinner("⏳ Hệ thống đang xử lý..."):
                final_response = orchestrator.run_lesson(
                    query=query_to_send, thread_id=st.session_state.thread_id,
                    target_chapter=st.session_state.target_file, target_section=st.session_state.target_section,
                    action_mode=action_mode_to_send
                )
                st.markdown(final_response)

    if final_response:
        st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.rerun()