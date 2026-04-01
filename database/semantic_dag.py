import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from typing import List, Dict, Set


# Giả định lớp này tuân thủ IGraphStore đã định nghĩa trong interfaces.py
class SemanticDAG:
    def __init__(self, llm_service, vector_store):
        """Khởi tạo Đồ thị có hướng và liên kết với các dịch vụ khác."""
        self.graph = nx.DiGraph()
        self.llm = llm_service  # Dùng để tóm tắt cộng đồng
        self.db = vector_store  # Dùng để lưu bản tóm tắt cộng đồng

    def build_graph_from_triplets(self, triplets: List[Dict[str, str]]) -> None:
        """Xây dựng khung xương đồ thị từ mảng JSON bóc tách."""
        for t in triplets:
            source = t.get("source")
            target = t.get("target")
            relation = t.get("relation", "")

            if source and target:
                # Quy ước: Cạnh hướng từ Nền tảng -> Khái niệm bậc cao
                self.graph.add_edge(source, target, relation=relation)

        print(
            f"[*] [SemanticDAG] Đã cập nhật. Tổng số đỉnh: {self.graph.number_of_nodes()}, Cạnh: {self.graph.number_of_edges()}")

    def detect_and_summarize_communities(self) -> None:
        """Thực thi thuật toán gom cụm và gọi LLM tóm tắt."""
        if self.graph.number_of_nodes() == 0:
            return

        print("[*] [SemanticDAG] Đang chạy thuật toán phân rã Modularity...")
        # Modularity yêu cầu đồ thị vô hướng
        undirected_g = self.graph.to_undirected()
        communities = list(greedy_modularity_communities(undirected_g))

        print(f"[*] [SemanticDAG] Phát hiện {len(communities)} cộng đồng tri thức.")

        for i, comm in enumerate(communities):
            comm_id = f"community_{i}"
            nodes_list = list(comm)

            # 1. Đánh dấu ID cộng đồng vào từng đỉnh
            for node in nodes_list:
                self.graph.nodes[node]['community_id'] = comm_id

            # 2. Gọi LLM viết First Principles
            summary = self.llm.summarize_community(nodes_list)

            # 3. Lưu vào ChromaDB (Ký hiệu metadata đặc biệt để phân biệt với text thường)
            self.db.upsert_documents(
                chunks=[summary],
                metadatas=[{"type": "community_summary", "community_id": comm_id}],
                ids=[comm_id]
            )

    def get_backward_context(self, anchor_nodes: List[str], max_depth: int = 2) -> str:
        """Thuật toán duyệt ngược dòng thời gian (Reverse BFS)."""
        visited_nodes: Set[str] = set()
        queue = [(node, 0) for node in anchor_nodes if node in self.graph]
        collected_comm_ids: Set[str] = set()

        # Thu thập cộng đồng của chính các điểm neo
        for node in anchor_nodes:
            if node in self.graph and 'community_id' in self.graph.nodes[node]:
                collected_comm_ids.add(self.graph.nodes[node]['community_id'])

        # Duyệt Reverse BFS
        while queue:
            current_node, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            # Lệnh predecessors() cực kỳ quan trọng: Chỉ lấy các đỉnh đi VÀO đỉnh hiện tại
            for predecessor in self.graph.predecessors(current_node):
                if predecessor not in visited_nodes:
                    visited_nodes.add(predecessor)
                    queue.append((predecessor, depth + 1))

                    if 'community_id' in self.graph.nodes[predecessor]:
                        collected_comm_ids.add(self.graph.nodes[predecessor]['community_id'])

        # Lấy văn bản tóm tắt từ ChromaDB
        context_texts = []
        for cid in collected_comm_ids:
            # Query chính xác bằng ID
            summary_result = self.db.collection.get(ids=[cid])
            if summary_result and summary_result['documents']:
                context_texts.append(f"[NỀN TẢNG - {cid}]: {summary_result['documents'][0]}")

        return "\n\n".join(context_texts)