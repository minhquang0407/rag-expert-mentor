import streamlit as st
import os
import json
import uuid
import time
import hashlib
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
if "target_chapter" not in st.session_state:
    st.session_state.target_chapter = ""
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
    st.header("🔍 System Diagnostics")
    if st.button("Run Connectivity Check", use_container_width=True):
        with st.spinner("Checking connections..."):
            # 1. Check Qdrant
            try:
                engine.vector_db.client.get_collections()
                st.success("✅ Qdrant Cloud: Connected")
            except Exception as e:
                st.error(f"❌ Qdrant Cloud: {e}")
            
            # 2. Check Neo4j
            try:
                with engine.graph_db.driver.session() as session:
                    session.run("RETURN 1")
                st.success("✅ Neo4j AuraDB: Connected")
            except Exception as e:
                st.error(f"❌ Neo4j AuraDB: {e}")
            
            # 3. Check Learning/Chat LLM
            try:
                test_prompt = "Say 'OK'"
                res = engine.orchestrator.llm_service.chat_llm.invoke(test_prompt)
                st.success(f"✅ Chat LLM: Connected (Response: {res.content})")
            except Exception as e:
                st.error(f"❌ Chat LLM: {e}")
            
            # 4. Check Ingestion LLM (JSON Mode test)
            try:
                st.info("Testing Ingestion LLM (JSON Mode)...")
                test_json_prompt = "Output a JSON with a field 'status' set to 'ready'"
                # Ingestion LLM is stored in llm_service.llm
                res_json = engine.orchestrator.llm_service.llm.invoke(test_json_prompt)
                st.success(f"✅ Ingestion LLM: Connected (Response: {res_json.content})")
            except Exception as e:
                st.warning(f"⚠️ Ingestion LLM (JSON Mode) failed: {e}. I will try to auto-fix this if you proceed.")

    st.markdown("---")
    st.header("⚙️ User Settings")
    if st.button("Reset Learning Progress", use_container_width=True, type="secondary"):
        with st.spinner("Deleting history..."):
            engine.reset_all_user_data(st.session_state.user_id)
            # Clear relevant session states
            st.session_state.messages = []
            st.session_state.current_seq_idx = 0
            # Also clear any cached quizzes or briefings
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith("quiz_") or k.startswith("briefing_")]
            for k in keys_to_delete:
                del st.session_state[k]
            
            st.success("✅ Progress reset successfully!")
            time.sleep(1)
            st.rerun()

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
                try:
                    from config.settings import settings
                    st.info(f"🚀 Initializing Ingestion (Provider: {settings.llm_provider}, Model: {settings.llm_model_name})")
                    
                    content = uploaded_file.getvalue().decode("utf-8")
                    
                    if not content:
                        st.error("File content is empty!")
                    else:
                        st.write(f"📦 File read: {len(content)} characters. Calling Pipeline...")
                        
                        # Run the pipeline
                        run_ingestion_pipeline(
                            content, file_name, 
                            engine.vector_db, 
                            engine.orchestrator.llm_service,
                            engine.graph_db
                        )
                        
                        st.success("🎉 Data ingested successfully! Reloading interface...")
                        time.sleep(2)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ CRITICAL ERROR during Ingestion: {str(e)}")
                    st.exception(e)
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
                        # Check both local and /tmp paths for TOC
                        local_toc = os.path.join(current_dir, "database", "tocs", f"{source_to_delete}_toc.json")
                        tmp_toc = os.path.join("/tmp/database/tocs", f"{source_to_delete}_toc.json")
                        
                        for path in [local_toc, tmp_toc]:
                            if os.path.exists(path):
                                os.remove(path)
                                print(f"[*] Deleted TOC file: {path}")

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

        # Check both local and /tmp paths for TOC
        local_toc_path = os.path.join(current_dir, "database", "tocs", f"{selected_file}_toc.json")
        tmp_toc_path = os.path.join("/tmp/database/tocs", f"{selected_file}_toc.json")
        
        toc_path = tmp_toc_path if os.path.exists(tmp_toc_path) else local_toc_path

        if os.path.exists(toc_path):
            # --- NEW: LEARNING PROGRESS DASHBOARD ---
            progress_data = engine.graph_db.get_file_learning_progress(st.session_state.user_id, selected_file)
            st.markdown("### 📊 Mastery Progress")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(progress_data["percent"] / 100)
            with col2:
                st.write(f"**{progress_data['percent']:.0f}%**")
            st.caption(f"🏆 {progress_data['learned']} / {progress_data['total']} concepts mastered")
            st.markdown("---")

            with open(toc_path, "r", encoding="utf-8") as f:
                toc_tree = json.load(f)

            if toc_tree:
                st.markdown("### 🗂️ Table of Contents")
                for chapter, sections in toc_tree.items():
                    with st.expander(f"📂 {chapter}"):
                        for sec in sections:
                            if st.button(f"📄 {sec}", key=f"btn_{selected_file}_{sec}"):
                                st.session_state.target_section = sec
                                st.session_state.target_chapter = chapter
                                
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
# 3.8 GRAPH VISUALIZATION HELPER
# ==========================================
def render_knowledge_graph(engine, user_id, chapter_name=None, file_name=None):
    """Helper to render graph using agraph for either a specific chapter or an entire file."""
    if chapter_name and file_name:
        st.markdown(f"### 📍 Chapter Graph: {chapter_name}")
        st.caption(f"Visualizing concepts and relations within the chapter '{chapter_name}'.")
        graph_data = engine.graph_db.get_chapter_visual_graph(user_id, file_name, chapter_name)
    elif file_name:
        st.markdown(f"### 🗺️ Global Knowledge Map: {file_name}")
        st.caption(f"Visualizing all concepts within '{file_name}'. Green nodes are learned.")
        graph_data = engine.graph_db.get_global_visual_graph(user_id, file_name)
    else:
        st.info("No source provided for graph rendering.")
        return

    if graph_data["nodes"]:
        nodes = []
        for n in graph_data["nodes"]:
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
        config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False)
        
        return_value = agraph(nodes=nodes, edges=edges, config=config)
        
        # --- NEW: INTERACTIVE NODE EXPLORER ---
        if return_value:
            # Find the node details from our graph_data
            selected_node = next((n for n in graph_data["nodes"] if n["id"] == return_value), None)
            
            if selected_node:
                with st.container(border=True):
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.markdown(f"### 💡 {selected_node['id']}")
                        st.write(selected_node['title'] or "No description available.")
                        st.caption(f"Type: {selected_node['type'].capitalize()}")
                    
                    with col_b:
                        if not selected_node['learned']:
                            if st.button("✅ Mark Learned", key=f"learn_{selected_node['id']}", use_container_width=True):
                                engine.graph_db.mark_concept_as_learned(selected_node['id'], st.session_state.user_id)
                                st.success(f"Mastered: {selected_node['id']}!")
                                time.sleep(1)
                                st.rerun()
                        else:
                            st.success("🌟 Learned")
            else:
                st.info(f"Concept: {return_value}")
    else:
        st.info("No graph data found for this view.")

# ==========================================
# 4. MAIN DISPLAY AREA & TABS
# ==========================================
st.title("AI Professor - Multi-System Expert")
st.caption("Multi-Agent Queue Architecture with Local/Global QA Routing.")

# Split into 4 main workspaces
tab_learning, tab_briefing, tab_map, tab_global_qa = st.tabs([
    "Learning Workspace", "Source Briefing", "Knowledge Map", "Global Q&A"
])

query_to_send = None
action_mode_to_send = None

# ==========================================
# WORKSPACE 1: LEARNING (LECTURE & LOCAL QA)
# ==========================================
with tab_learning:
    if st.session_state.target_section and st.session_state.entity_groups:
        st.subheader(f"{st.session_state.target_section}")

        # Split into 4 sub-tabs within Learning
        subtab_learn, subtab_local_qa, subtab_assess, subtab_local_graph = st.tabs([
            "Lecture Progress", "Lesson Q&A", "Self-Assessment", "Local Graph"
        ])

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

        # --- SUB-TAB: SELF-ASSESSMENT ---
        with subtab_assess:
            st.markdown("### 🧠 AI Live Quiz")
            st.caption("Generate a real-time interactive quiz based on this lesson's content.")
            
            # Use session state to store quiz to avoid regeneration on every click
            quiz_key = f"quiz_{st.session_state.target_file}_{st.session_state.target_section}"
            
            if quiz_key not in st.session_state:
                if st.button("✨ Generate AI Live Quiz", use_container_width=True):
                    with st.spinner("AI is reading the lesson and preparing questions..."):
                        quiz, err = engine.get_lesson_quiz(st.session_state.target_file, st.session_state.target_section)
                        if quiz:
                            st.session_state[quiz_key] = quiz
                            st.rerun()
                        else:
                            st.error(err if err else "Could not generate quiz.")
            
            if quiz_key in st.session_state:
                quiz = st.session_state[quiz_key]
                if not quiz:
                    st.warning("AI failed to generate a quiz. Please try again.")
                    if st.button("Retry"):
                        del st.session_state[quiz_key]
                        st.rerun()
                else:
                    for i, q in enumerate(quiz):
                        with st.container(border=True):
                            st.markdown(f"**Question {i+1}:** {q['question']}")
                            user_choice = st.radio(f"Select your answer:", q["options"], key=f"q_{quiz_key}_{i}", index=None)
                            
                            if user_choice is not None:
                                # Check if correct
                                choice_idx = q["options"].index(user_choice)
                                if choice_idx == q["answer_idx"]:
                                    st.success(f"✅ Correct! {q['explanation']}")
                                else:
                                    st.error(f"❌ Incorrect. The correct answer is: {q['options'][q['answer_idx']]}")
                                    st.info(f"💡 **Explanation:** {q['explanation']}")
                    
                    if st.button("Reset Quiz", use_container_width=True):
                        del st.session_state[quiz_key]
                        st.rerun()
            else:
                # Show fallback to hypothetical questions if no quiz generated yet
                st.markdown("---")
                st.caption("Or view pre-generated study points:")
                p_id = hashlib.md5(f"{st.session_state.target_file}__{st.session_state.target_chapter}__{st.session_state.target_section}".encode('utf-8')).hexdigest()
                hy_questions = engine.vector_db.get_section_questions(p_id)
                if hy_questions:
                    for idx, hq in enumerate(hy_questions):
                        with st.expander(f"Study Point {idx+1}"):
                            st.write(hq)

        # --- SUB-TAB: LOCAL GRAPH ---
        with subtab_local_graph:
            if st.session_state.target_chapter and st.session_state.target_file:
                render_knowledge_graph(engine, st.session_state.user_id, 
                                        chapter_name=st.session_state.target_chapter, 
                                        file_name=st.session_state.target_file)
            else:
                st.warning("Please select a lesson to view its chapter graph.")

    else:
        st.info("👈 Please select a lesson from the Sidebar to start the Learning Workspace.")

# ==========================================
# WORKSPACE 2: SOURCE BRIEFING (NOTEBOOKLM STYLE)
# ==========================================
with tab_briefing:
    if st.session_state.target_file:
        st.markdown(f"## Source Briefing: {st.session_state.target_file}")
        st.caption("AI-generated overview, themes, and deep-dive discussion transcript.")
        
        briefing_key = f"briefing_{st.session_state.target_file}"
        
        if briefing_key not in st.session_state:
            if st.button("✨ Generate Source Briefing", use_container_width=True):
                with st.spinner("Reading every page of your document to ensure 100% accuracy..."):
                    briefing, err = engine.get_source_briefing(st.session_state.target_file)
                    if briefing:
                        st.session_state[briefing_key] = briefing
                        st.rerun()
                    else:
                        st.error(err if err else "Could not generate briefing.")
        
        if briefing_key in st.session_state:
            b = st.session_state[briefing_key]
            if b:
                # 1. Synopsis
                st.markdown("### 📝 Synopsis")
                st.write(b.get("synopsis", "No synopsis available."))
                
                # 2. Key Themes
                st.markdown("### 🔑 Key Themes")
                themes = b.get("key_themes", [])
                cols = st.columns(len(themes) if themes else 1)
                for i, theme in enumerate(themes):
                    cols[i % len(cols)].info(f"**{theme}**")
                
                # 3. Podcast Script
                st.markdown("### AI Deep Dive (Podcast Script)")
                with st.expander("Reveal Discussion Transcript"):
                    st.markdown(b.get("podcast_script", "No script available."))
                
                if st.button("Regenerate Briefing"):
                    del st.session_state[briefing_key]
                    st.rerun()
            else:
                st.error("Failed to generate briefing. Please try again.")
    else:
        st.info("👈 Please select a subject in the Sidebar to unlock the Source Briefing.")

# ==========================================
# WORKSPACE 3: KNOWLEDGE MAP (GLOBAL)
# ==========================================
with tab_map:
    if st.session_state.target_file:
        render_knowledge_graph(engine, st.session_state.user_id, file_name=st.session_state.target_file)
    else:
        st.info("👈 Please select or upload a subject in the Sidebar to view the Global Map.")

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
