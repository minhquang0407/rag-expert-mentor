import json
import os
import hashlib
from database.document_processor import MathAwareDocumentProcessor


def run_ingestion_pipeline(markdown_content: str, file_name: str, db, llm, neo4j_db):
    """
    - Lí do tại sao dùng: Xử lý file Markdown, giải quyết lỗi "văn bản mồ côi" (orphan text) và kích hoạt kiến trúc Roadmap-Driven.
    - Chức năng: Băm nhỏ, gom rác văn bản mồ côi vào Section đầu tiên, gửi LLM trích xuất Lộ trình và Đồ thị, sau đó lưu vào Qdrant & Neo4j.
    """
    processor = MathAwareDocumentProcessor(max_chunk_size=1000)
    raw_chunks, toc_tree = processor.process_markdown(markdown_content)

    os.makedirs("./database/tocs", exist_ok=True)
    toc_path = f"./database/tocs/{file_name}_toc.json"
    with open(toc_path, "w", encoding="utf-8") as f:
        json.dump(toc_tree, f, ensure_ascii=False, indent=4)
    print(f"[*] Đã lưu Mục lục tại {toc_path}")

    sections_dict = {}
    orphan_buffer = ""

    for chunk in raw_chunks:
        chap_name = chunk["metadata"].get("Header 1", "Default Chapter")
        sec_name = chunk["metadata"].get("Header 2", chunk["metadata"].get("Section"))

        if chap_name and not sec_name:
            orphan_buffer += chunk["page_content"] + "\n\n"
            continue

        if not sec_name: sec_name = "Default Section"
        key = (chap_name, sec_name)

        if key not in sections_dict:
            sections_dict[key] = []
            if orphan_buffer:
                chunk["page_content"] = orphan_buffer + chunk["page_content"]
                orphan_buffer = ""

        sections_dict[key].append(chunk)

    print("\n[START] Pipeline Ingestion (Roadmap-Driven GraphRAG)...")


    for (chap_name, sec_name), chunks_list in sections_dict.items():
        print(f"\n -> Processing: [{chap_name}] - {sec_name}")
        full_section_text = "\n\n".join([c["page_content"] for c in chunks_list])
        parent_id = hashlib.md5(f"{file_name}_{chap_name}_{sec_name}".encode('utf-8')).hexdigest()

        llm_data = llm.extract_section_curriculum_and_dag(full_section_text)

        teaching_roadmap = llm_data.get("teaching_roadmap", [])
        triplets = llm_data.get("graph_triplets", [])

        main_entities = set()
        for step in teaching_roadmap:
            main_entities.update(step.get("associated_concepts", []))

        if triplets or main_entities:
            neo4j_db.save_graph_triplets(triplets, file_name, chap_name, sec_name, list(main_entities))
            print(f"    + Đã đẩy {len(triplets)} cạnh và đánh dấu {len(main_entities)} Concept trọng tâm vào Neo4j.")

        for step in teaching_roadmap:
            db.upsert_curriculum_group(step, parent_id, file_name, chap_name, sec_name)

        print(f"    + Đã lập kế hoạch gồm {len(teaching_roadmap)} bước giảng dạy.")

        questions = llm.generate_hypothetical_questions(full_section_text, num_questions=5)
        print(f"    + Đã sinh {len(questions)} câu hỏi giả định.")

        section_anchors = set()
        for t in triplets:
            if "source" in t: section_anchors.add(t["source"])
            if "target" in t: section_anchors.add(t["target"])

        parent_metadata = {
            "source": file_name,
            "chapter": chap_name,
            "section": sec_name,
            "anchor_nodes": ", ".join(list(section_anchors))
        }

        db.upsert_section(full_section_text, parent_metadata, parent_id)
        db.upsert_questions(questions, parent_id, file_name)

    print("\n=== DONE! ===")