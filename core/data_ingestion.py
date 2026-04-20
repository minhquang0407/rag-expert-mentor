from database.document_processor import MathAwareDocumentProcessor
import json
import os


def run_ingestion_pipeline(markdown_content: str, file_name: str, db, llm, dag):
    processor = MathAwareDocumentProcessor(max_chunk_size=1000)

    # 1. Cắt văn bản & Trích xuất Mục lục
    chunks, toc_tree = processor.process_markdown(markdown_content)

    # SAVE FILE CONTENT
    os.makedirs("./database/tocs", exist_ok=True)
    toc_path = f"./database/tocs/{file_name}_toc.json"
    with open(toc_path, "w", encoding="utf-8") as f:
        json.dump(toc_tree, f, ensure_ascii=False, indent=4)
    print(f"[*] Đã lưu Mục lục tại {toc_path}")

    # SUPER-TEXT AGGREGATION)
    sections_dict = {}
    for i, chunk in enumerate(chunks):
        # Lấy tên Section từ metadata (mặc định là 'General' nếu không có)
        sec_name = chunk["metadata"].get("Section", "General")
        if sec_name not in sections_dict:
            sections_dict[sec_name] = []
        # Gắn kèm ID để sau này lưu Vector DB cho chuẩn
        sections_dict[sec_name].append({"id": i, "chunk_data": chunk})

    global_glossary = set()
    texts, metadatas, ids = [], [], []

    # ROLLING INGESTION)
    print("\n[Bắt đầu] Trích xuất Đồ thị Tri thức Cấp độ Section...")

    for sec_name, items in sections_dict.items():
        print(f"\n -> Đang xử lý Section: {sec_name} ({len(items)} chunks)")

        # Gộp tất cả text của các chunk trong Section này lại thành 1 văn bản lớn
        full_section_text = "\n\n".join([item["chunk_data"]["page_content"] for item in items])

        # Giới hạn Glossary truyền vào để tránh vượt quá context window
        current_glossary = list(global_glossary)[-200:]

        # GỌI LLM 1 LẦN DUY NHẤT CHO TOÀN BỘ SECTION
        section_triplets = llm.extract_graph_entities(full_section_text, glossary=current_glossary)

        # Thu thập các Điểm neo (Anchors) của Section này
        section_anchors = set()
        for t in section_triplets:
            if "source" in t:
                section_anchors.add(t["source"])
                global_glossary.add(t["source"])
            if "target" in t:
                section_anchors.add(t["target"])
                global_glossary.add(t["target"])

        anchor_str = ", ".join(list(section_anchors))

        # Cập nhật ngay vào Đồ thị Tổng
        if section_triplets:
            dag.build_graph_from_triplets(section_triplets)

        for item in items:
            chunk = item["chunk_data"]
            meta = chunk["metadata"]
            meta["source"] = file_name
            meta["anchor_nodes"] = anchor_str  # Gắn siêu-mỏ-neo vào từng mảnh nhỏ

            texts.append(chunk["page_content"])
            metadatas.append(meta)
            ids.append(f"{file_name}_chunk_{item['id']}")

    # 5. Lưu vào Não trái (ChromaDB)
    print("\n[*] Đang lưu Vector vào ChromaDB...")
    db.upsert_documents(texts, metadatas, ids)

    print("\n=== HOÀN TẤT NẠP DỮ LIỆU VÀ XÂY DỰNG ĐỒ THỊ ===")