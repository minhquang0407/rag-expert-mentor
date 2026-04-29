import json
import os
import hashlib
from database.document_processor import MathAwareDocumentProcessor


def run_ingestion_pipeline(markdown_content: str, file_name: str, db, llm, dag):
    processor = MathAwareDocumentProcessor(max_chunk_size=1000)
    chunks, toc_tree = processor.process_markdown(markdown_content)

    os.makedirs("./database/tocs", exist_ok=True)
    toc_path = f"./database/tocs/{file_name}_toc.json"
    with open(toc_path, "w", encoding="utf-8") as f:
        json.dump(toc_tree, f, ensure_ascii=False, indent=4)
    print(f"[*] Đã lưu Mục lục tại {toc_path}")

    sections_dict = {}
    for i, chunk in enumerate(chunks):
        # Fallback an toàn nếu metadata không có Section
        sec_name = chunk["metadata"].get("Section", "General")
        if sec_name not in sections_dict: sections_dict[sec_name] = []
        sections_dict[sec_name].append(chunk)

    print("\n[START] Pipeline Ingestion: DAG, Curriculum & QA Generation...")

    for sec_name, chunks_list in sections_dict.items():
        print(f"\n -> Processing: {sec_name}")
        full_section_text = "\n\n".join([c["page_content"] for c in chunks_list])

        parent_id = hashlib.md5(f"{file_name}_{sec_name}".encode('utf-8')).hexdigest()

        # =======================================================
        # 1. GỌI HÀM LLM MỚI (SINGLE-PASS EXTRACTION)
        # =======================================================
        llm_data = llm.extract_section_curriculum_and_dag(full_section_text)

        curriculum_groups = llm_data.get("curriculum_groups", [])
        triplets = llm_data.get("graph_triplets", [])

        # =======================================================
        # 2. XÂY DỰNG ĐỒ THỊ DAG
        # =======================================================
        section_anchors = set()
        for t in triplets:
            if "source" in t: section_anchors.add(t["source"])
            if "target" in t: section_anchors.add(t["target"])

        if triplets:
            dag.build_graph_from_triplets(triplets)

        # =======================================================
        # 3. LƯU GIÁO ÁN VÀO QDRANT
        # =======================================================
        for group in curriculum_groups:
            db.upsert_curriculum_group(group, parent_id, file_name, sec_name)

        print(f"    + Đã lưu {len(curriculum_groups)} Cụm thực thể (Giáo án).")

        # =======================================================
        # 4. LƯU SECTION GỐC & CÂU HỎI (DÀNH CHO LUỒNG Q&A)
        # =======================================================
        questions = llm.generate_hypothetical_questions(full_section_text, num_questions=5)
        print(f"    + Đã sinh {len(questions)} câu hỏi giả định.")

        parent_metadata = {
            "source": file_name,
            "section": sec_name,
            "anchor_nodes": ", ".join(list(section_anchors))
        }

        db.upsert_section(full_section_text, parent_metadata, parent_id)
        db.upsert_questions(questions, parent_id, file_name)

    print("\n=== DONE! ===")