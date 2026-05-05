import os
import uuid
from typing import Dict, Any, List
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


class QdrantVectorStore:
    def __init__(self, collection_name="math_curriculum"):
        self.parent_coll = collection_name
        self.child_coll = f"{collection_name}_questions"

        self.client = QdrantClient(host="localhost", port=6333)

        self.embed_model = FastEmbedEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Chặn lỗi Qdrant không nhận dạng model (cập nhật mới nhất của thư viện)
        try:
            self.client.set_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        except Exception:
            pass

        for coll in [self.parent_coll, self.child_coll]:
            if not self.client.collection_exists(coll):
                self.client.create_collection(
                    collection_name=coll,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )

    def upsert_section(self, text: str, metadata: dict, parent_id: str):
        """Lưu toàn bộ nội dung Section làm mỏ neo gốc."""
        vector = self.embed_model.embed_query(text)

        payload = metadata.copy()
        payload["parent_id"] = parent_id
        payload["type"] = "section_anchor"
        payload["page_content"] = text  # Đã chuẩn hóa dùng page_content

        self.client.upsert(
            collection_name=self.parent_coll,
            points=[PointStruct(id=parent_id, vector=vector, payload=payload)]
        )

    def upsert_questions(self, questions: List[str], parent_id: str, source_file: str):
        """Lưu danh sách câu hỏi giả định (Child vectors)."""
        if not questions:
            return

        vectors = list(self.embed_model.embed_documents(questions))

        points = []
        for idx, (q_text, vec) in enumerate(zip(questions, vectors)):
            valid_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_q_{idx}"))

            payload = {
                "page_content": q_text,  # Đổi từ document sang page_content cho đồng bộ
                "parent_id": parent_id,
                "source": source_file,
                "type": "question"
            }
            points.append(PointStruct(id=valid_id, vector=vec, payload=payload))

        self.client.upsert(
            collection_name=self.child_coll,
            points=points
        )

    def get_curriculum_groups(self, target_file: str, target_section: str) -> list:
        """Kéo mảng JSON curriculum_groups từ Qdrant dựa vào Tên sách và Mục lục."""
        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="source", match=models.MatchValue(value=target_file)),
                models.FieldCondition(key="section", match=models.MatchValue(value=target_section)),
                models.FieldCondition(key="type", match=models.MatchValue(value="curriculum_group"))
            ]
        )

        records, _ = self.client.scroll(
            collection_name=self.parent_coll,
            scroll_filter=query_filter,
            limit=100,
            with_payload=True
        )

        groups = []
        for r in records:
            if "curriculum_data" in r.payload:
                groups.append(r.payload["curriculum_data"])

        groups.sort(key=lambda x: x.get("seq_id", 0))
        return groups

    def upsert_curriculum_group(self, group_data: dict, parent_id: str, source_file: str, chapter: str, section: str):
        """Lưu nhóm giáo án (Teaching Step) vào Qdrant."""
        # VÁ LỖI 1: Lấy đúng key của Schema mới (verbatim_exact_quotes)
        vector_text = group_data.get("verbatim_exact_quotes", "")
        # Cứu hộ trong trường hợp LLM sinh tên cũ
        if not vector_text: vector_text = group_data.get("verbatim_text", "")

        vector = self.embed_model.embed_query(vector_text)
        point_id = str(uuid.uuid4())

        payload = {
            "source": source_file,
            "chapter": chapter,
            "section": section,
            "parent_id": parent_id,
            "type": "curriculum_group",
            "curriculum_data": group_data
        }

        # VÁ LỖI 2: Đưa vào self.parent_coll thay vì biến không tồn tại
        self.client.upsert(
            collection_name=self.parent_coll,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)]
        )

    def get_section_exact(self, target_file: str, target_section: str) -> List[Dict[str, Any]]:
        """Dùng cho luồng LESSON_PROGRESS. Tìm trực tiếp trên Bảng Cha."""
        conditions = []
        if target_file: conditions.append(
            models.FieldCondition(key="source", match=models.MatchValue(value=target_file)))
        if target_section: conditions.append(
            models.FieldCondition(key="section", match=models.MatchValue(value=target_section)))

        filter_query = models.Filter(must=conditions) if conditions else None
        records, _ = self.client.scroll(
            collection_name=self.parent_coll, scroll_filter=filter_query, limit=1000, with_payload=True
        )
        # VÁ LỖI 3: Lấy đúng key page_content
        return [{"page_content": r.payload.get("page_content", ""), "metadata": r.payload} for r in records]

    def search_candidates_and_fetch_parent(self, query: str, llm_service, target_file: str = "") -> List[
        Dict[str, Any]]:
        """Tích hợp toàn bộ Luồng Option 3 + 4 cho Q&A."""
        conditions = []
        if target_file: conditions.append(
            models.FieldCondition(key="source", match=models.MatchValue(value=target_file)))
        filter_query = models.Filter(must=conditions) if conditions else None

        # 1. Lọc thô trên Bảng Con
        results = self.client.query(
            collection_name=self.child_coll,
            query_text=query,
            query_filter=filter_query,
            limit=5
        )

        if not results: return []

        # VÁ LỖI 3: Lấy đúng key page_content
        candidates = [{"question": r.payload.get("page_content", ""), "parent_id": r.payload.get("parent_id")} for r in
                      results]

        best_parent_id = llm_service.rerank_candidate_questions(query, candidates)
        if not best_parent_id: return []

        # 3. Kéo dữ liệu từ Bảng Cha
        parent_records, _ = self.client.scroll(
            collection_name=self.parent_coll,
            scroll_filter=models.Filter(must=[models.HasIdCondition(has_id=[best_parent_id])]),
            limit=1,
            with_payload=True
        )

        return [{"page_content": r.payload.get("page_content", ""), "metadata": r.payload} for r in parent_records]