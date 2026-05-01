import os
from neo4j import GraphDatabase
from typing import List, Dict, Any


class Neo4jManager:
    """
    - Lí do tại sao dùng: Quản lý Đồ thị Tri thức toàn cục, giải quyết triệt để lỗi ghi đè dữ liệu khi nhiều sách có cùng tên Section.
    """

    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "ExpertMentor2026")
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            self._initialize_user()
        except Exception as e:
            print(f"[Neo4j] ❌ Lỗi kết nối: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def _initialize_user(self, user_id: str = "guest_01"):
        query = "MERGE (u:User {id: $user_id}) RETURN u"
        with self.driver.session() as session:
            session.run(query, user_id=user_id)

    def save_graph_triplets(self, triplets: List[Dict[str, Any]], file_name: str, chapter_name: str, section_title: str,
                            main_entities: List[str]):
        """
        - Chức năng: Lưu Graph. Tạo chuỗi định vị tuyệt đối (Absolute Locator) để tránh trùng lặp Section giữa các File.
        - Tham số mới: Nhận thêm file_name và chapter_name.
        """
        allowed_relations = {"PREREQUISITE_OF", "RELATES_TO", "PART_OF", "DESCRIBES"}

        # Tạo chuỗi định vị tuyệt đối (Ví dụ: "a_math.md :: Chương 2 :: 2.2 Chuẩn hóa")
        locator = f"{file_name}::{chapter_name}::{section_title}"

        with self.driver.session() as session:
            # 1. Cập nhật các main_entity
            for main_ent in main_entities:
                if not main_ent: continue
                session.run("""
                MERGE (c:Concept {id: $id})
                SET c.is_main = true
                SET c.source_locators = CASE WHEN $loc IN coalesce(c.source_locators, []) THEN c.source_locators ELSE coalesce(c.source_locators, []) + $loc END
                """, id=main_ent.strip(), loc=locator)

            # 2. Xây dựng các cạnh và cập nhật micro_entities
            for t in triplets:
                source = str(t.get("source", "")).strip()
                target = str(t.get("target", "")).strip()
                raw_rel = str(t.get("relation", "RELATES_TO")).strip().upper().replace(" ", "_")

                if not source or not target: continue
                rel = raw_rel if raw_rel in allowed_relations else "RELATES_TO"

                cypher_query = f"""
                MERGE (c1:Concept {{id: $source}})
                SET c1.source_locators = CASE WHEN $loc IN coalesce(c1.source_locators, []) THEN c1.source_locators ELSE coalesce(c1.source_locators, []) + $loc END

                MERGE (c2:Concept {{id: $target}})
                SET c2.source_locators = CASE WHEN $loc IN coalesce(c2.source_locators, []) THEN c2.source_locators ELSE coalesce(c2.source_locators, []) + $loc END

                WITH c1, c2
                CALL apoc.create.relationship(c1, $rel, {{}}, c2) YIELD rel
                RETURN rel
                """
                session.run(cypher_query, source=source, target=target, rel=rel, loc=locator)

    # ... (Các hàm mark_concept_as_learned, get_unlearned_prerequisites, get_learned_prerequisites, get_concept_subgraph giữ nguyên y hệt bản cũ) ...
    def mark_concept_as_learned(self, concept_id: str, user_id: str = "guest_01"):
        query = """
        MATCH (u:User {id: $user_id})
        MATCH (c:Concept {id: $concept_id})
        MERGE (u)-[r:HAS_LEARNED]->(c)
        SET r.timestamp = timestamp()
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id, concept_id=concept_id)

    def get_unlearned_prerequisites(self, target_concept: str, max_depth: int = 2, user_id: str = "guest_01") -> List[
        str]:
        query = f"""
        MATCH (target:Concept {{id: $target_concept}})
        MATCH (prereq:Concept)-[:PREREQUISITE_OF*1..{max_depth}]->(target)
        OPTIONAL MATCH (u:User {{id: $user_id}})-[:HAS_LEARNED]->(prereq)
        WITH prereq, u
        WHERE u IS NULL
        RETURN DISTINCT prereq.id AS missing_concept
        """
        with self.driver.session() as session:
            result = session.run(query, target_concept=target_concept, user_id=user_id)
            return [record["missing_concept"] for record in result]

    def get_learned_prerequisites(self, target_concept: str, max_depth: int = 3, user_id: str = "guest_01") -> List[
        str]:
        query = f"""
        MATCH (target:Concept {{id: $target_concept}})
        MATCH (prereq:Concept)-[:PREREQUISITE_OF*1..{max_depth}]->(target)
        MATCH (u:User {{id: $user_id}})-[:HAS_LEARNED]->(prereq)
        RETURN DISTINCT prereq.id AS learned_concept
        """
        with self.driver.session() as session:
            result = session.run(query, target_concept=target_concept, user_id=user_id)
            return [record["learned_concept"] for record in result]

    def get_concept_subgraph(self, target_concept: str, max_depth: int = 1) -> Dict[str, List[str]]:
        query = f"""
        MATCH (target:Concept {{id: $target_concept}})
        OPTIONAL MATCH (prereq:Concept)-[:PREREQUISITE_OF*1..{max_depth}]->(target)
        OPTIONAL MATCH (target)-[:PREREQUISITE_OF*1..{max_depth}]->(leads_to:Concept)
        OPTIONAL MATCH (target)-[:RELATES_TO]-(related:Concept)
        RETURN 
            collect(DISTINCT prereq.id) AS prerequisites,
            collect(DISTINCT leads_to.id) AS leads_to,
            collect(DISTINCT related.id) AS related_concepts
        """
        with self.driver.session() as session:
            result = session.run(query, target_concept=target_concept).single()
            prereqs = [p for p in result["prerequisites"] if p is not None] if result and result[
                "prerequisites"] else []
            leads = [l for l in result["leads_to"] if l is not None] if result and result["leads_to"] else []
            related = [r for r in result["related_concepts"] if r is not None] if result and result[
                "related_concepts"] else []
            return {
                "prerequisites": prereqs,
                "leads_to": leads,
                "related_concepts": related
            }