from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class ILLMService(ABC):
    """Interface tiêu chuẩn cho dịch vụ LLM (Trích xuất & Hỏi đáp)."""

    @abstractmethod
    def extract_section_curriculum_and_dag(self, section_text: str, existing_nodes: List[str] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_hypothetical_questions(self, section_text: str, num_questions: int = 5) -> List[str]:
        pass

    @abstractmethod
    def rerank_candidate_questions(self, user_query: str, candidates: List[Dict[str, str]]) -> str:
        pass


class IVectorStore(ABC):
    """Interface cho Cơ sở dữ liệu Vector."""

    @abstractmethod
    def upsert_section(self, text: str, metadata: dict, parent_id: str) -> None:
        pass

    @abstractmethod
    def upsert_questions(self, questions: List[str], parent_id: str, source_file: str) -> None:
        pass

    @abstractmethod
    def upsert_curriculum_group(self, group_data: dict, parent_id: str, source_file: str, chapter: str, section: str) -> None:
        pass

    @abstractmethod
    def get_curriculum_groups(self, target_file: str, target_section: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_section_exact(self, target_file: str, target_section: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def search_candidates_and_fetch_parent(self, query: str, llm_service: ILLMService, target_file: str = "") -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def upsert_user_memory(self, user_id: str, turn_id: str, query: str, answer: str, summary: str):
        pass

    @abstractmethod
    def search_semantic_memory(self, user_id: str, query: str, limit: int = 5) -> List[Dict]:
        pass


class IGraphStore(ABC):
    """Interface cho Cơ sở dữ liệu Đồ thị."""

    @abstractmethod
    def save_knowledge_graph(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], file_name: str, chapter_name: str, section_title: str, main_entities: List[str]):
        pass

    @abstractmethod
    def mark_concept_as_learned(self, concept_id: str, user_id: str = "guest_01"):
        pass

    @abstractmethod
    def get_unlearned_prerequisites(self, target_concept: str, max_depth: int = 2, user_id: str = "guest_01") -> List[str]:
        pass

    @abstractmethod
    def get_learned_prerequisites(self, target_concept: str, max_depth: int = 3, user_id: str = "guest_01") -> List[str]:
        pass

    @abstractmethod
    def get_concept_subgraph(self, target_concept: str, max_depth: int = 1) -> Dict[str, List[str]]:
        pass

    @abstractmethod
    def get_graph_context(self, node_names: List[str], search_mode: str = "search") -> List[Dict[str, str]]:
        pass

    @abstractmethod
    def save_chat_turn(self, user_id: str, turn_id: str, query: str, raw_answer: str, summary: str, concept_ids: list = None, target_file: str = "", target_section: str = ""):
        pass

    @abstractmethod
    def get_recent_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        pass

    @abstractmethod
    def get_raw_chat_turns(self, turn_ids: List[str]) -> List[Dict]:
        pass

    @abstractmethod
    def get_raw_chat_turns_by_user(self, user_id: str) -> List[Dict]:
        pass