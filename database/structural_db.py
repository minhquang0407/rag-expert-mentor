from core.interfaces import IVectorStore
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
class ChromaVectorStore(IVectorStore):
    def __init__(self, db_path: str = "./database/chroma_storage"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name="textbook_knowledge",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        if not chunks:
            return

        self.collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        print(f"[*] [ChromaDB] Đã đồng bộ {len(chunks)} khối tri thức xuống ổ cứng.")

    def retrieve_chapter_in_order(self, target_chapter: str) -> str:
        """
        [CHẾ ĐỘ 1: KHÔI PHỤC CẤU TRÚC]
        Lấy toàn bộ một chương và NẮN THẲNG lại thứ tự dựa trên seq_id.
        """
        results = self.collection.get(
            where={"chapter": target_chapter}
        )

        if not results['documents']:
            return "Không tìm thấy dữ liệu cho chương này."

        retrieved_chunks = []
        for i in range(len(results['ids'])):
            retrieved_chunks.append({
                "content": results['documents'][i],
                "seq_id": results['metadatas'][i].get("seq_id", 0)
            })

        # THUẬT TOÁN SẮP XẾP LẠI THEO THỨ TỰ THỜI GIAN (O(N log N))
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["seq_id"])

        # Nối ghép hoàn chỉnh. LLM sẽ nhận được văn bản mượt mà từ trên xuống dưới.
        full_text = "\n\n".join([chunk["content"] for chunk in sorted_chunks])
        return full_text

    def retrieve_semantic(self, query: str, target_chapter: str = None, top_k: int = 3) -> List[str]:
        """
        [CHẾ ĐỘ 2: TRUY XUẤT NGỮ NGHĨA]
        Tìm kiếm các đoạn văn cụ thể trả lời cho một câu hỏi.
        """
        filter_dict = {"chapter": target_chapter} if target_chapter else None

        # Dùng .query() để kích hoạt tính toán khoảng cách Vector
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_dict
        )

        if not results['documents'] or not results['documents'][0]:
            return []

        return results['documents'][0]