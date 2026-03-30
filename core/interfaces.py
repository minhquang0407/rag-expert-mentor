from abc import ABC, abstractmethod
from typing import List, Dict, Any


class ILLMService(ABC):
    """Interface tiêu chuẩn cho mọi Mô hình Ngôn ngữ."""

    @abstractmethod
    def generate_markdown_from_pdf(self, pdf_bytes: bytes) -> str:
        """Nhận byte của file PDF và trả về Markdown nguyên thủy (bảo toàn LaTeX)."""
        pass

    @abstractmethod
    def extract_graph_entities(self, text_chunk: str) -> List[Dict[str, str]]:
        """Nhận một đoạn văn bản và trích xuất thành mảng JSON chứa các triplet (Source, Target, Relation)."""
        pass


class IVectorStore(ABC):
    """Interface cho Cơ sở dữ liệu Vector (ChromaDB, FAISS, Qdrant...)."""

    @abstractmethod
    def upsert_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        pass

    @abstractmethod
    def retrieve(self, query: str, filters: Dict[str, Any] = None, top_k: int = 3) -> List[str]:
        pass


class IGraphStore(ABC):
    """Interface cho Cơ sở dữ liệu Đồ thị (NetworkX, Neo4j...)."""

    @abstractmethod
    def add_triplets(self, triplets: List[Dict[str, str]]) -> None:
        pass

    @abstractmethod
    def get_backward_context(self, anchor_nodes: List[str], max_depth: int) -> str:
        """Duyệt ngược đồ thị để lấy First Principles."""
        pass