import streamlit as st
import os
import json
import uuid
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# Lấy đường dẫn tuyệt đối của thư mục chứa file
current_dir = Path(__file__).parent.resolve()
import sys

if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from core.container import Container
from config.settings import settings
from core.data_ingestion import run_ingestion_pipeline
from streamlit_agraph import agraph, Node, Edge, Config


# ==========================================
# 1. INITIALIZE CENTRAL SYSTEM
# ==========================================
@st.cache_resource
def init_system():
    """
    - Reason: Initialize all core services (Qdrant, Neo4j, LLM, Engine) exactly once and cache them in Streamlit to avoid reloading on every UI refresh.
    - Function: Establish DB connections and assemble the RuntimeEngine.
    - Usage: Called automatically when the Streamlit app starts.
    - Parameters: None.
    - Returns: RuntimeEngine - The central engine that coordinates all tasks.
    - Alternatives: None.
    """
    container = Container()
    container.config.from_pydantic(settings)
    engine = container.runtime_engine()
    return engine


engine = init_system()

# ==========================================
# 2. STATE MANAGEMENT (UI STATE)
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"
if "user_id" not in st.session_state:
    st.session_state.user_id = "guest_01"
if "messages" not in st.session_state:
    # Format mới: {"role": "user/assistant", "content": "...", "mode": "LEARNING/LOCAL_QA/GLOBAL_QA"}
    st.session_state.messages = []
if "target_file" not in st.session_state:
    st.session_state.target_file = ""
if "target_section" not in st.session_state:
    st.session_state.target_section = ""
if "entity_groups" not in st.session_state:
    st.session_state.entity_groups = []
if "current_seq_idx" not in st.session_state:
    st.session_state.current_seq_idx = 0
if "user_id" not in st.session_state:
    # Streamlit >= 1.30 uses st.query_params
    st.session_state.user_id = st.query_params.get("user", "guest_01")
    st.session_state.current_loaded_section = None

# ==========================================
# 3. SIDEBAR - NAVIGATION & DATA INGESTION
# ==========================================
with st.sidebar:
    st.header("🔐 Login")
    user_input = st.text_input("Username (User ID):", value=st.session_state.user_id)

    if user_input != st.session_state.user_id:
        st.session_state.user_id = user_input
        # Reload history from Neo4j when user changes
        if user_input:
            history = engine.graph_db.get_raw_chat_turns_by_user(user_input)
            st.session_state.messages = []
            for h in history:
                st.session_state.messages.append({"role": "user", "content": h["query"], "mode": "GLOBAL_QA"})
                st.session_state.messages.append({"role": "assistant", "content": h["answer"], "mode": "GLOBAL_QA"})
            st.session_state.current_loaded_section = None
            st.rerun()

    st.markdown("---")
    st.header("System Settings")

    st.markdown("---")
    st.header("Ingest Learning Data")
    uploaded_file = st.file_uploader("Upload document (.md)", type=["md"])

    unique_sources = []
    try:
        records, _ = engine.vector_db.client.scroll(
            collection_name=engine.vector_db.parent_coll,
            limit=500,
            with_payload=True,
            with_vectors=False
        )
        unique_sources = list(set([r.payload.get("source") for r in records if r.payload and r.payload.get("source")]))
    except Exception as e:
        st.error(f"Error scanning Qdrant: {e}")

    if uploaded_file is not None:
        file_name = uploaded_file.name
        if file_name in unique_sources:
            st.success(f"'{file_name}' is already in the database.")
        else:
            st.warning(f"'{file_name}' is not in the database yet.")
            if st.button(f"📥 Start ingesting '{file_name}'", use_container_width=True):
                # Using st.status for persistent logging in the UI
                with st.status(f"Ingesting '{file_name}'...", expanded=True) as status:
                    try:
                        st.write("🔍 Reading file content...")
                        content = uploaded_file.getvalue().decode("utf-8")
                        
                        if not content:
                            st.error("File content is empty!")
                            status.update(label="Ingestion Failed (Empty File)", state="error")
                        else:
                            st.write(f"📦 Processing {len(content)} characters...")
                            
                            # Run the pipeline
                            run_ingestion_pipeline(
                                content, file_name, 
                                engine.vector_db, 
                                engine.orchestrator.llm_service,
                                engine.graph_db
                            )
                            
                            st.write("✅ Ingestion completed successfully.")
                            status.update(label="Ingestion Complete!", state="complete")
                            
                            st.success("Data ingested successfully! Reloading interface...")
                            time.sleep(2)
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Ingestion Failed: {str(e)}")
                        st.exception(e)
                        status.update(label="Ingestion Failed", state="error")
                        # Stop here so user can read the error
                        st.stop()

    if unique_sources:
        st.markdown("---")
        st.header("🗑️ Manage Ingested Data")
        source_to_delete = st.selectbox("Select file to remove:", options=["-- Select --"] + unique_sources)
        if source_to_delete != "-- Select --":
            if st.button(f"🔥 Permanently Delete '{source_to_delete}'", use_container_width=True, type="primary"):
                with st.spinner(f"Deleting data for {source_to_delete}..."):
                    try:
                        # 1. Delete from Qdrant
                        engine.vector_db.delete_source(source_to_delete)
                        # 2. Delete from Neo4j
                        engine.graph_db.delete_source(source_to_delete)
                        toc_path = os.path.join(current_dir, "database", "tocs", f"{source_to_delete}_toc.json")
                        if os.path.exists(toc_path):
                            os.remove(toc_path)
                            print(f"[*] Deleted local TOC file: {toc_path}")
                        st.success(f"Successfully deleted all data for '{source_to_delete}'")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during deletion: {e}")

    st.markdown("---")
    st.header("📖 Learning Curriculum")

    if unique_sources:
        selected_file = st.selectbox("📚 Subject:", options=unique_sources, key="file_selector")

        if selected_file != st.session_state.target_file:
            st.session_state.target_file = selected_file
            st.session_state.target_section = ""

        toc_path = os.path.join(current_dir, "database", "tocs", f"{selected_file}_toc.json")

        if os.path.exists(toc_path):
            with open(toc_path, "r", encoding="utf-8") as f:
                toc_tree = json.load(f)

            if toc_tree:
                st.markdown("### 🗂️ Table of Contents")
                for chapter, sections in toc_tree.items():
                    with st.expander(f"📂 {chapter}"):
                        for sec in sections:
                            if st.button(f"📄 {sec}", key=f"btn_{selected_file}_{sec}"):
                                st.session_state.target_section = sec
                                
                                # [FIXED]: Don't clear messages directly. Set flag to trigger Lazy Loading from Neo4j
                                st.session_state.current_loaded_section = None 
                                
                                st.session_state.current_seq_idx = 0
                                st.session_state.entity_groups = engine.vector_db.get_curriculum_groups(selected_file,
                                                                                                        sec)
                                st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"
                                st.rerun()

                if st.session_state.target_section:
                    st.success(f"**Currently learning:** {st.session_state.target_section}")
                else:
                    st.info("Please select a topic to begin!")
    else:
        st.warning("Database is empty. Please upload a file.")

# ==========================================
# 3.5 LAZY LOADING FROM NEO4J (SSOT)
# ==========================================
def _sync_section_history():
    """
    - Reason: Solves Streamlit's state evaporation on tab switch or F5. Neo4j becomes the Single Source of Truth.
    - Function: Pulls chat history from Neo4j, classifies into LEARNING and LOCAL_QA for correct tab rendering.
      Restores the current lesson index (seq_idx).
    - Usage: Called automatically when st.session_state.current_loaded_section changes.
    - Parameters: None (reads directly from st.session_state).
    - Returns: None.
    """
    if st.session_state.target_file and st.session_state.target_section:
        current_section_key = f"{st.session_state.target_file}_{st.session_state.target_section}"
        
        # Only load if history for this section hasn't been fetched yet
        if st.session_state.get("current_loaded_section") != current_section_key:
            with st.spinner("Syncing learning history from Neo4j..."):
                history = engine.graph_db.get_history_by_section(
                    user_id=st.session_state.user_id,
                    target_file=st.session_state.target_file,
                    target_section=st.session_state.target_section,
                    limit=50
                )
                
                # Keep GLOBAL_QA messages (cross-section), clear old section-specific messages
                global_msgs = [m for m in st.session_state.messages if m.get("mode") == "GLOBAL_QA"]
                st.session_state.messages = global_msgs
                
                # Restore from DB
                restored_seq_idx = 0
                for h in history:
                    q = h.get("query", "")
                    a = h.get("answer", "")
                    
                    if q.startswith("Learn "):
                        # This is a lecture (LEARNING)
                        st.session_state.messages.append({"role": "assistant", "content": a, "mode": "LEARNING"})
                        restored_seq_idx += 1
                    else:
                        # This is a Q&A turn (LOCAL_QA)
                        st.session_state.messages.append({"role": "user", "content": q, "mode": "LOCAL_QA"})
                        st.session_state.messages.append({"role": "assistant", "content": a, "mode": "LOCAL_QA"})
                
                # Update lesson index based on how many lectures were restored
                if restored_seq_idx > 0:
                    st.session_state.current_seq_idx = restored_seq_idx
                    
                st.session_state.current_loaded_section = current_section_key

_sync_section_history()

# ==========================================
# 4. MAIN DISPLAY AREA & TABS
# ==========================================
st.title("AI Professor - Multi-System Expert")
st.caption("Multi-Agent Queue Architecture with Local/Global QA Routing.")

# Split into 2 main workspaces
tab_learning, tab_global_qa = st.tabs(["Learning Workspace", "Global Q&A"])

query_to_send = None
action_mode_to_send = None

# ==========================================
# WORKSPACE 1: LEARNING (LECTURE & LOCAL QA)
# ==========================================
with tab_learning:
    if st.session_state.target_section and st.session_state.entity_groups:
        st.subheader(f"{st.session_state.target_section}")

        # Split into 2 sub-tabs within Learning
        subtab_learn, subtab_local_qa, subtab_map = st.tabs(["Lecture Progress", "Lesson Q&A", "Knowledge Map"])

        # --- SUB-TAB: LECTURE PROGRESS ---
        with subtab_learn:
            # Only render messages belonging to LEARNING mode
            for msg in st.session_state.messages:
                if msg["mode"] == "LEARNING":
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            is_finished = st.session_state.current_seq_idx >= len(st.session_state.entity_groups)
            if not is_finished:
                btn_text = "Start Lesson" if st.session_state.current_seq_idx == 0 else "Understood, continue lesson"
                if st.button(btn_text, use_container_width=True):
                    query_to_send = "Please continue the lecture."
                    action_mode_to_send = "LEARNING"
            else:
                st.success("You have completed this lesson! Please switch to another topic in the Sidebar.")

        # --- SUB-TAB: LOCAL QA ---
        with subtab_local_qa:
            # Only render messages belonging to LOCAL_QA mode
            for msg in st.session_state.messages:
                if msg["mode"] == "LOCAL_QA":
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            local_q = st.chat_input("Ask details about the current lesson...", key="input_local")
            if local_q:
                query_to_send = local_q
                action_mode_to_send = "LOCAL_QA"
        # --- SUB-TAB: KNOWLEDGE MAP ---
        with subtab_map:
            st.markdown("### 🗺️ Global Knowledge Map")
            st.caption(f"Visualizing all concepts and relations within '{st.session_state.target_file}'. Green nodes indicate concepts you've already learned.")
            
            # Switch to Global Graph for the entire file
            graph_data = engine.graph_db.get_global_visual_graph(st.session_state.user_id, st.session_state.target_file)
            
            if graph_data["nodes"]:
                nodes = []
                for n in graph_data["nodes"]:
                    # Determine color based on status
                    if n["learned"]:
                        color = "#2ECC71" # Green
                    elif n["is_main"]:
                        color = "#E67E22" # Orange
                    else:
                        color = "#3498DB" # Blue
                    
                    nodes.append(Node(
                        id=n["id"], 
                        label=n["label"], 
                        size=25 if n["is_main"] else 15,
                        color=color,
                        title=n["title"]
                    ))
                
                edges = [Edge(source=e["source"], target=e["target"], label=e["label"]) for e in graph_data["edges"]]
                
                config = Config(
                    width=700, 
                    height=500, 
                    directed=True, 
                    physics=True, 
                    hierarchical=False,
                    # More customization
                )
                
                return_value = agraph(nodes=nodes, edges=edges, config=config)
                
                if return_value:
                    st.info(f"**Concept:** {return_value}")
            else:
                st.info("No concept graph data found for this section yet.")

    else:
        st.info("👈 Please select a lesson from the Sidebar to start the Learning Workspace.")

# ==========================================
# WORKSPACE 2: GLOBAL Q&A
# ==========================================
with tab_global_qa:
    # Only render messages belonging to GLOBAL_QA mode
    for msg in st.session_state.messages:
        if msg["mode"] == "GLOBAL_QA":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    global_q = st.chat_input("Ask any question across the entire system...", key="input_global")
    if global_q:
        query_to_send = global_q
        action_mode_to_send = "GLOBAL_QA"

# ==========================================
# 5. RUNTIME EXECUTION
# ==========================================
if query_to_send and action_mode_to_send:
    # 1. Save question to session state with Mode label
    if action_mode_to_send != "LEARNING":  # Avoid printing "Please continue" on screen
        st.session_state.messages.append({"role": "user", "content": query_to_send, "mode": action_mode_to_send})

    # Show temporary spinner on UI (will disappear after rerun)
    with st.spinner("⏳ The system is thinking and querying the Graph..."):

        # FLOW 1: LEARNING
        if action_mode_to_send == "LEARNING":
            step_data = st.session_state.entity_groups[st.session_state.current_seq_idx]
            
            status_box = st.status("Preparing lesson...", expanded=True)
            
            generator = engine.process_action(
                action_mode="LEARNING",
                target_file=st.session_state.target_file,
                target_section=st.session_state.target_section,
                step_data=step_data,
                user_id=st.session_state.user_id
            )
            
            full_lecture_text = ""
            current_text = ""
            stream_container = None
            
            for event in generator:
                if event["type"] == "status":
                    status_box.write(f"✔️ {event['message']}")
                    status_box.update(label=event["message"])
                elif event["type"] == "agent_start":
                    st.markdown(f"### [{event['agent'].upper()}]")
                    stream_container = st.empty()
                    current_text = ""
                elif event["type"] == "chunk":
                    current_text += event["content"]
                    stream_container.markdown(current_text + "▌")
                elif event["type"] == "agent_end":
                    stream_container.markdown(current_text)
                    full_lecture_text += f"### [{event['agent'].upper()}]\n{current_text}\n\n---\n\n"
            
            status_box.update(label="Lesson is completed!", state="complete", expanded=False)
            
            st.session_state.current_seq_idx += 1
            st.session_state.messages.append({"role": "assistant", "content": full_lecture_text, "mode": "LEARNING"})

        # FLOW 2: QA
        else:
            answer = engine.process_action(
                action_mode=action_mode_to_send,
                query=query_to_send,
                target_file=st.session_state.target_file,
                target_section=st.session_state.target_section,
                user_id=st.session_state.user_id
            )

            if answer:
                st.session_state.messages.append({"role": "assistant", "content": answer, "mode": action_mode_to_send})
            else:
                st.error("Error: Engine did not return an answer.")

    # Automatically reload interface to show messages in the correct Tab
    st.rerun()
