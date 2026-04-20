from qdrant_client import QdrantClient, models
from typing import List, Dict, Any


class QdrantVectorStore:
    def __init__(self, collection_name: str = "math_curriculum"):
        """
        - Lí do tại sao dùng: Thiết lập kết nối đến Qdrant Server và cấu hình Dual-Embedding (Dense + Sparse).
        - Chức năng: Khởi tạo collection nếu chưa tồn tại, tải các mô hình nhúng cục bộ vào bộ nhớ RAM.
        - Cách dùng: Gọi khi khởi tạo ứng dụng `db = QdrantVectorStore()`.
        - Tham số: `collection_name` (Tên không gian lưu trữ).
        - Trả về, Kiểu trả về: Đối tượng `QdrantVectorStore`.
        - Các hàm thay thế nếu có: Có thể dùng `QdrantClient(":memory:")` để test trên RAM mà không cần Docker.
        """
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection_name = collection_name

        # 1. Cấu hình FastEmbed Models
        # Dense Model: Hỗ trợ đa ngôn ngữ (Toán học Anh/Việt)
        self.client.set_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        # Sparse Model: Trích xuất từ khóa dựa trên tần suất (BM25)
        self.client.set_sparse_model("Qdrant/bm25")

        # 2. Khởi tạo Collection với cấu hình Hybrid
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.client.get_fastembed_vector_params(),
                sparse_vectors_config=self.client.get_fastembed_sparse_vector_params()
            )
            print(f"[+] Đã khởi tạo Qdrant Collection: {collection_name}")

    def upsert_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """
        - Lí do tại sao dùng: Nạp dữ liệu vào CSDL.
        - Chức năng: Tự động chạy văn bản qua 2 mô hình (Dense và Sparse) để sinh Vector, sau đó lưu kèm siêu dữ liệu.
        - Cách dùng: `db.upsert_documents(["Định lý Bayes..."], [{"Section": "1.1"}], ["chunk_1"])`
        - Tham số: `texts` (Danh sách văn bản), `metadatas` (Danh sách thẻ thông tin), `ids` (Định danh độc nhất).
        - Trả về, Kiểu trả về: `None`.
        - Các hàm thay thế nếu có: `self.client.upsert()` nếu tự tính toán vector ở bên ngoài thay vì dùng FastEmbed.
        """
        # Qdrant client.add() tự động xử lý FastEmbed song song
        self.client.add(
            collection_name=self.collection_name,
            documents=texts,
            metadata=metadatas,
            ids=ids
        )
        print(f"[*] Đã nạp thành công {len(texts)} chunks vào Qdrant.")

    def get_section_exact(self, target_file: str, target_section: str) -> List[Dict[str, Any]]:
        """
        - Lí do tại sao dùng: Thay thế cho lệnh `.get()` của ChromaDB trong luồng [LESSON_PROGRESS].
        - Chức năng: Cuộn (Scroll) và lấy toàn bộ văn bản thuộc một Mục lục cụ thể, triệt tiêu ảo giác LLM.
        - Cách dùng: Lấy toàn bộ văn bản để xây dựng Siêu Đỉnh (Super-Node).
        - Tham số: `target_file` (Tên file), `target_section` (Tên mục lục).
        - Trả về, Kiểu trả về: `List[Dict]` chứa nội dung và metadata.
        - Các hàm thay thế nếu có: Không có. Thao tác Scroll là bắt buộc để lấy dữ liệu Exact Match số lượng lớn.
        """
        conditions = []
        if target_file:
            conditions.append(models.FieldCondition(key="source", match=models.MatchValue(value=target_file)))
        if target_section:
            conditions.append(models.FieldCondition(key="Section", match=models.MatchValue(value=target_section)))

        filter_query = models.Filter(must=conditions) if conditions else None

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_query,
            limit=1000,  # Đảm bảo lấy sạch toàn bộ chunk của Section
            with_payload=True  # Payload tương đương với Metadata trong Chroma
        )

        # Tiền xử lý dữ liệu trả về cho giống chuẩn cũ
        return [{"page_content": r.document, "metadata": r.payload} for r in records]

    def hybrid_search(self, query: str, target_file: str = "", target_section: str = "", limit: int = 3) -> List[
        Dict[str, Any]]:
        """
        - Lí do tại sao dùng: Tìm kiếm thông minh cho luồng [QA]. Khắc phục điểm yếu bắt từ khóa của Dense Vector.
        - Chức năng: Kích hoạt thuật toán RRF kết hợp BM25 và Multilingual-E5, ép phạm vi tìm kiếm theo bộ lọc.
        - Cách dùng: `db.hybrid_search("Ma trận kề là gì?", limit=3)`
        - Tham số: `query` (Câu hỏi), `target_file`, `target_section` (Bộ lọc mỏ neo), `limit` (Số lượng kết quả).
        - Trả về, Kiểu trả về: `List[Dict]` chứa nội dung tài liệu.
        - Các hàm thay thế nếu có: `self.client.search()` nếu chỉ muốn tìm bằng Dense Vector truyền thống.
        """
        conditions = []
        if target_file:
            conditions.append(models.FieldCondition(key="source", match=models.MatchValue(value=target_file)))
        if target_section:
            conditions.append(models.FieldCondition(key="Section", match=models.MatchValue(value=target_section)))

        filter_query = models.Filter(must=conditions) if conditions else None

        # client.query() của Qdrant tự động thực thi Native RRF khi có cả Dense và Sparse model
        results = self.client.query(
            collection_name=self.collection_name,
            query_text=query,
            query_filter=filter_query,
            limit=limit
        )

        return [{"page_content": r.document, "metadata": r.payload} for r in results]