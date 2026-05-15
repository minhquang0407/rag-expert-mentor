import hashlib
from database.document_processor import MathAwareDocumentProcessor

import streamlit as st

def run_ingestion_pipeline(markdown_content: str, file_name: str, db, llm, dag):
    processor = MathAwareDocumentProcessor()
    
    # 1. Initialize status framework
    with st.status("Processing learning data...", expanded=True) as status:
        status.update(label="Analyzing document structure (Markdown Headers)...")
        final_document, toc_tree = processor.process_markdown(markdown_content)

        status.update(label=f"Saving Table of Contents for: {file_name}")
        dag.save_table_of_contents(file_name, toc_tree)

        global_nodes_list = []
        status.update(label=f"Found {len(final_document)} sections. Starting AI processing loop...")
        
        loop_placeholder = st.empty()

        for i, section in enumerate(final_document):
            sec_name = section.get("metadata", {}).get("Section", "Unknown")
            chapter_name = section.get("metadata", {}).get("Chapter", "Unknown")
            
            with loop_placeholder.container():
                st.write(f"--- **Processing Section {i+1}/{len(final_document)}: {sec_name}** ---")
                
                try:
                    full_section_text = section.get("page_content","")
                    parent_id = hashlib.md5(f"{file_name}__{chapter_name}__{sec_name}".encode('utf-8')).hexdigest()

                    llm_data = llm.extract_section_curriculum_and_dag(full_section_text, existing_nodes=global_nodes_list)
                
                    main_entities = llm_data.get("main_entities", [])
                    teaching_roadmap = llm_data.get("teaching_roadmap", [])
                    
                    kg_data = llm_data.get("knowledge_graph", {})
                    edges = kg_data.get("edges", [])
                    nodes = kg_data.get("nodes", [])

                    for node in nodes:
                        node_name = node.get("name", "").strip()
                        if node_name and node_name not in global_nodes_list:
                            global_nodes_list.append(node_name)

                    # BUILD NEO4J DAG
                    section_anchors = set()
                    for node in nodes:
                        if "name" in node: section_anchors.add(node["name"])
                    for e in edges:
                        if "source" in e: section_anchors.add(e["source"])
                        if "target" in e: section_anchors.add(e["target"])

                    st.write(f"🧬 Saving {len(nodes)} nodes and relations to Neo4j...")
                    if nodes or edges:
                        dag.save_knowledge_graph(
                            nodes=nodes, edges=edges,
                            file_name=file_name, chapter_name=chapter_name,
                            section_title=sec_name, main_entities=main_entities
                        )

                    # UPSERT CURRICULUM INTO QDRANT
                    st.write(f"Saving {len(teaching_roadmap)} curriculum steps to Qdrant...")
                    for idx, step_data in enumerate(teaching_roadmap):
                        step_data["seq_id"] = idx
                        db.upsert_curriculum_group(
                            group_data=step_data, parent_id=parent_id,
                            source_file=file_name, chapter=chapter_name, section=sec_name
                        )

                    # UPSERT PARENT SECTION & HYPOTHETICAL QUESTIONS
                    st.write("Generating hypothetical questions...")
                    questions = llm.generate_hypothetical_questions(full_section_text, num_questions=5)

                    parent_metadata = {
                        "source": file_name,
                        "section": sec_name,
                        "seq_id": section["metadata"]["seq_id"],
                        "anchor_nodes": ", ".join(list(section_anchors)),
                        "main_entities": ", ".join(main_entities)
                    }

                    db.upsert_section(full_section_text, parent_metadata, parent_id)
                    db.upsert_questions(questions, parent_id, file_name)
                    
                    st.write(f"Storage complete for: {sec_name}")

                except Exception as inner_e:
                    st.error(f"Error at Section {i+1} ({sec_name}): {str(inner_e)}")
                    st.exception(inner_e)
                    st.stop()
        
        status.update(label="Data ingested successfully!", state="complete", expanded=False)
