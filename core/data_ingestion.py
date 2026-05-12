import json
import os
import hashlib
from database.document_processor import MathAwareDocumentProcessor


import streamlit as st

def run_ingestion_pipeline(markdown_content: str, file_name: str, db, llm, dag):
    st.write("🛠️ Starting Ingestion Pipeline...")
    processor = MathAwareDocumentProcessor()
    
    st.write("📂 Parsing structural hierarchy (Markdown Headers)...")
    final_document, toc_tree = processor.process_markdown(markdown_content)

    st.write(f"📁 Creating TOC directories and saving {file_name}_toc.json...")
    os.makedirs("./database/tocs", exist_ok=True)
    toc_path = f"./database/tocs/{file_name}_toc.json"
    with open(toc_path, "w", encoding="utf-8") as f:
        json.dump(toc_tree, f, ensure_ascii=False, indent=4)

    global_nodes_list = []
    st.write(f"🚀 Found {len(final_document)} sections. Starting LLM analysis loop...")

    for i, section in enumerate(final_document):
        sec_name = section["metadata"]["Section"]
        chapter_name = section["metadata"]["Chapter"]
        st.write(f"📝 **Processing Section {i+1}: {sec_name}** ({len(section.get('page_content',''))} chars)")
        
        full_section_text = section.get("page_content","")
        parent_id = hashlib.md5(f"{file_name}__{chapter_name}__{sec_name}".encode('utf-8')).hexdigest()

        st.write("🤖 Calling LLM for Extraction (Backbone & Graph)...")
        # =======================================================
        # 1. INVOKE SINGLE-PASS LLM EXTRACTION
        # =======================================================
        llm_data = llm.extract_section_curriculum_and_dag(full_section_text, existing_nodes=global_nodes_list)
        st.write("✅ LLM returned data.")

        # [NEW]: Extract main entities from the parsed LLM response
        main_entities = llm_data.get("main_entities", [])
        teaching_roadmap = llm_data.get("teaching_roadmap", [])
        
        kg_data = llm_data.get("knowledge_graph", {})
        edges = kg_data.get("edges", [])
        nodes = kg_data.get("nodes", [])

        # Add newly discovered nodes to the global list for the next sections to see
        for node in nodes:
            node_name = node.get("name", "").strip()
            if node_name and node_name not in global_nodes_list:
                global_nodes_list.append(node_name)

        # =======================================================
        # 2. BUILD NEO4J DAG
        # =======================================================
        section_anchors = set()
        for node in nodes:
            if "name" in node: section_anchors.add(node["name"])
        for e in edges:
            if "source" in e: section_anchors.add(e["source"])
            if "target" in e: section_anchors.add(e["target"])

        if nodes or edges:
            dag.save_knowledge_graph(
                nodes=nodes,
                edges=edges,
                file_name=file_name,
                chapter_name=chapter_name,
                section_title=sec_name,
                main_entities=main_entities
            )

        # =======================================================
        # 3. UPSERT CURRICULUM INTO QDRANT
        # =======================================================
        for idx, step_data in enumerate(teaching_roadmap):
            step_data["seq_id"] = idx
            db.upsert_curriculum_group(
                group_data=step_data,
                parent_id=parent_id,
                source_file=file_name,
                chapter=chapter_name,
                section=sec_name
            )

        print(f"    + Saved {len(teaching_roadmap)} Teaching Steps (Agent Queues).")

        # =======================================================
        # 4. UPSERT PARENT SECTION & HYPOTHETICAL QUESTIONS
        # =======================================================
        questions = llm.generate_hypothetical_questions(full_section_text, num_questions=5)
        print(f"    + Generated {len(questions)} hypothetical questions.")

        # [UPDATED]: Store main_entities directly into parent metadata for advanced retrieval
        parent_metadata = {
            "source": file_name,
            "section": sec_name,
            "seq_id":section["metadata"]["seq_id"],
            "anchor_nodes": ", ".join(list(section_anchors)),
            "main_entities": ", ".join(main_entities)  # Injecting the entities here
        }

        db.upsert_section(full_section_text, parent_metadata, parent_id)
        db.upsert_questions(questions, parent_id, file_name)

        print(f"    + Saved Section Anchor mapping {len(main_entities)} Main Entities.")

    print("\n=== DONE! ===")