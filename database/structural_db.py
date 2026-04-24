from qdrant_client import QdrantClient, models
from typing import List, Dict, Any


class QdrantVectorStore:
    def __init__(self, collection_name: str = "math_curriculum"):
        """
        Khởi tạo 2 Collection:
        1. {collection_name}_sections: Bảng Cha (Chỉ lưu văn bản, dùng để truy xuất Exact Match).
        2. {collection_name}_questions: Bảng Con (Được mã hóa Vector để Hybrid Search).
        """
        self.client = QdrantClient(url="http://localhost:6333")
        self.parent_coll = f"{collection_name}_sections"
        self.child_coll = f"{collection_name}_questions"

        self.client.set_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.client.set_sparse_model("Qdrant/bm25")

        # Tạo Bảng Cha (Chỉ cần Dense Vector cơ bản, hoặc lưu payload)
        if not self.client.collection_exists(self.parent_coll):
            self.client.create_collection(
                collection_name=self.parent_coll,
                vectors_config=self.client.get_fastembed_vector_params()
            )

        # Tạo Bảng Con (Bắt buộc Hybrid FastEmbed)
        if not self.client.collection_exists(self.child_coll):
            self.client.create_collection(
                collection_name=self.child_coll,
                vectors_config=self.client.get_fastembed_vector_params(),
                sparse_vectors_config=self.client.get_fastembed_sparse_vector_params()
            )

    def upsert_section(self, section_text: str, metadata: Dict[str, Any], section_id: str):
        """Lưu một Section vào Bảng Cha."""
        self.client.add(
            collection_name=self.parent_coll,
            documents=[section_text],
            metadata=[metadata],
            ids=[section_id]
        )

    def upsert_questions(self, questions: List[str], parent_id: str, source_file: str):
        """Lưu các câu hỏi giả định vào Bảng Con, trỏ ngược về Bảng Cha."""
        if not questions: return

        ids = [f"{parent_id}_q_{i}" for i in range(len(questions))]
        metadatas = [{"parent_id": parent_id, "source": source_file, "type": "hypothetical_question"} for _ in
                     questions]

        self.client.add(
            collection_name=self.child_coll,
            documents=questions,
            metadata=metadatas,
            ids=ids
        )

    def get_section_exact(self, target_file: str, target_section: str) -> List[Dict[str, Any]]:
        """Dùng cho luồng LESSON_PROGRESS. Tìm trực tiếp trên Bảng Cha."""
        conditions = []
        if target_file: conditions.append(
            models.FieldCondition(key="source", match=models.MatchValue(value=target_file)))
        if target_section: conditions.append(
            models.FieldCondition(key="Section", match=models.MatchValue(value=target_section)))

        filter_query = models.Filter(must=conditions) if conditions else None
        records, _ = self.client.scroll(
            collection_name=self.parent_coll, scroll_filter=filter_query, limit=1000, with_payload=True
        )
        return [{"page_content": r.document, "metadata": r.payload} for r in records]

    def search_candidates_and_fetch_parent(self, query: str, llm_service, target_file: str = "") -> List[
        Dict[str, Any]]:
        """
        Tích hợp toàn bộ Luồng Option 3 + 4:
        1. Tìm 5 câu hỏi giả định gần nhất.
        2. Dùng LLM Rerank tìm ra parent_id xịn nhất.
        3. Truy xuất Section gốc từ parent_id đó.
        """
        conditions = []
        if target_file: conditions.append(
            models.FieldCondition(key="source", match=models.MatchValue(value=target_file)))
        filter_query = models.Filter(must=conditions) if conditions else None

        # 1. Lọc thô (Stage 1) trên Bảng Con
        results = self.client.query(
            collection_name=self.child_coll,
            query_text=query,
            query_filter=filter_query,
            limit=5
        )

        if not results: return []

        # Chuẩn bị danh sách ứng viên cho LLM
        candidates = [{"question": r.document, "parent_id": r.payload.get("parent_id")} for r in results]

        # 2. Lọc tinh bằng LLM (Stage 2)
        best_parent_id = llm_service.rerank_candidate_questions(query, candidates)
        if not best_parent_id: return []

        # 3. Kéo dữ liệu từ Bảng Cha (Parent Fetch)
        parent_records, _ = self.client.scroll(
            collection_name=self.parent_coll,
            scroll_filter=models.Filter(must=[models.HasIdCondition(has_id=[best_parent_id])]),
            limit=1,
            with_payload=True
        )

        return [{"page_content": r.document, "metadata": r.payload} for r in parent_records]