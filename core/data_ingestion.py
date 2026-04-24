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

    # Gom nhóm theo Section
    sections_dict = {}
    for i, chunk in enumerate(chunks):
        sec_name = chunk["metadata"].get("Section", "General")
        if sec_name not in sections_dict: sections_dict[sec_name] = []
        sections_dict[sec_name].append(chunk)

    global_glossary = set()

    print("\n[Bắt đầu] Pipeline Ingestion: DAG & LLM Question Generation...")
    for sec_name, chunks_list in sections_dict.items():
        print(f"\n -> Đang xử lý: {sec_name}")
        full_section_text = "\n\n".join([c["page_content"] for c in chunks_list])

        # Tạo UUID duy nhất (Parent ID) cho Section này dựa trên tên file và mục lục
        parent_id = hashlib.md5(f"{file_name}_{sec_name}".encode('utf-8')).hexdigest()

        # 1. TRÍCH XUẤT ĐỒ THỊ NGỮ NGHĨA (DAG)
        current_glossary = list(global_glossary)[-200:]
        triplets = llm.extract_graph_entities(full_section_text, glossary=current_glossary)

        section_anchors = set()
        for t in triplets:
            if "source" in t: section_anchors.add(t["source"]); global_glossary.add(t["source"])
            if "target" in t: section_anchors.add(t["target"]); global_glossary.add(t["target"])

        if triplets: dag.build_graph_from_triplets(triplets)

        # 2. SINH CÂU HỎI GIẢ ĐỊNH (REVERSE HyDE)
        questions = llm.generate_hypothetical_questions(full_section_text, num_questions=5)
        print(f"    + Đã sinh {len(questions)} câu hỏi giả định.")

        # 3. LƯU VÀO CSDL (TÁCH CHA CON)
        parent_metadata = {
            "source": file_name,
            "Section": sec_name,
            "anchor_nodes": ", ".join(list(section_anchors))
        }

        # Lưu vào bảng Cha
        db.upsert_section(full_section_text, parent_metadata, parent_id)
        # Lưu vào bảng Con
        db.upsert_questions(questions, parent_id, file_name)

    print("\n=== HOÀN TẤT NẠP DỮ LIỆU ===")