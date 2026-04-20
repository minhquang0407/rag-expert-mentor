import heapq

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from typing import List, Dict, Set, Tuple


# Giả định lớp này tuân thủ IGraphStore đã định nghĩa trong interfaces.py
class SemanticDAG:
    def __init__(self, llm_service, vector_store):
        """Khởi tạo Đồ thị có hướng và liên kết với các dịch vụ khác."""
        self.graph = nx.DiGraph()
        self.llm = llm_service  # Dùng để tóm tắt cộng đồng
        self.db = vector_store  # Dùng để lưu bản tóm tắt cộng đồng

    def build_graph_from_triplets(self, triplets: List[Dict[str, str]]):
        """
        Lí do tại sao dùng: Nạp dữ liệu JSON vào cấu trúc Toán học NetworkX.
        Chức năng: Cập nhật các Node và Edge, bổ sung Trọng số Khoảng cách (Weight).
        """
        for triplet in triplets:
            source = triplet.get("source")
            target = triplet.get("target")
            relation = triplet.get("relation", "")
            # Lấy trọng số, mặc định là 5 (mức trung bình) nếu LLM quên xuất ra
            weight = triplet.get("weight", 5)

            if source and target:
                # Đảm bảo Node tồn tại
                if not self.graph.has_node(source):
                    self.graph.add_node(source)
                if not self.graph.has_node(target):
                    self.graph.add_node(target)

                # Thêm hoặc cập nhật Cạnh với Trọng số (weight)
                self.graph.add_edge(source, target, relation=relation, weight=int(weight))

        print(
            f"[*] Cập nhật DAG: Hiện có {self.graph.number_of_nodes()} đỉnh và {self.graph.number_of_edges()} cạnh có trọng số.")

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

    def get_backward_context(self, anchor_nodes: List[str], max_nodes: int = 15, max_weight: int = 6) -> str:
        """
        - Lí do tại sao dùng: Trích xuất "Nguyên lý thứ nhất" (First Principles) làm nền tảng cho bài học hiện tại, đảm bảo triết lý Sư phạm Bottom-Up.
        - Chức năng: Thuật toán Weighted Reverse BFS (Dijkstra lội ngược). Tìm các đường đi có trọng số thấp (liên kết chặt chẽ) dẫn tới các Điểm neo.
        - Cách dùng: Gọi bên trong RetrievalNode sau khi đã lấy được danh sách anchor_nodes từ Qdrant.
        - Tham số:
            - `anchor_nodes`: Danh sách tên các thực thể hiện tại.
            - `max_nodes`: Giới hạn số lượng đỉnh lội ngược (Chống tràn RAM/Token).
            - `max_weight`: Ngưỡng lọc trọng số. Bỏ qua các liên kết lỏng lẻo (> 6).
        - Trả về, Kiểu trả về: `str` - Chuỗi văn bản đã định dạng XML chứa các Mối quan hệ logic.
        - Các hàm thay thế nếu có: Reverse DFS (Duyệt theo chiều sâu), nhưng không tối ưu bằng BFS Priority Queue.
        """
        if not anchor_nodes:
            return ""

        # Khởi tạo Hàng đợi ưu tiên (Priority Queue) cho thuật toán Dijkstra
        # Cấu trúc phần tử: (Tổng trọng số tích lũy, Tên đỉnh hiện tại)
        pq: List[Tuple[int, str]] = []
        visited_nodes: Set[str] = set()
        collected_edges: Set[Tuple[str, str, str]] = set()

        # Nạp các đỉnh gốc vào hàng đợi
        for anchor in anchor_nodes:
            if self.graph.has_node(anchor):
                heapq.heappush(pq, (0, anchor))
                visited_nodes.add(anchor)

        # Thực thi Thuật toán Lan truyền ngược (Reverse Propagation)
        nodes_processed = 0
        while pq and nodes_processed < max_nodes:
            current_cost, current_node = heapq.heappop(pq)
            nodes_processed += 1

            # Lệnh predecessors(): Cực kỳ quan trọng. Chỉ lấy các đỉnh đi VÀO đỉnh hiện tại (Tiền quyết định)
            for predecessor in self.graph.predecessors(current_node):
                edge_data = self.graph.get_edge_data(predecessor, current_node)
                weight = edge_data.get('weight', 5)

                # Bộ lọc Trọng số: Bỏ qua các cạnh có quan hệ quá lỏng lẻo
                if weight <= max_weight:
                    relation = edge_data.get('relation', 'liên kết với')
                    # Thu thập Cạnh (Edge) này vào bộ nhớ
                    collected_edges.add((predecessor, relation, current_node))

                    if predecessor not in visited_nodes:
                        visited_nodes.add(predecessor)
                        # Đẩy đỉnh cha vào hàng đợi để tiếp tục lội ngược với chi phí cộng dồn
                        heapq.heappush(pq, (current_cost + weight, predecessor))

        # Định dạng đầu ra siêu nén (XML Logic Paths) cho LLM
        if not collected_edges:
            return ""

        context_lines = ["<LOGICAL_CHAINS>"]
        for src, rel, dst in collected_edges:
            context_lines.append(f"  [{src}] --({rel})--> [{dst}]")
        context_lines.append("</LOGICAL_CHAINS>")

        return "\n".join(context_lines)