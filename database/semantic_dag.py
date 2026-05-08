import os
from neo4j import GraphDatabase
from typing import List, Dict, Any
from core.interfaces import IGraphStore


class Neo4jManager(IGraphStore):
    """
    - Lí do tại sao dùng: Quản lý Đồ thị Tri thức toàn cục, giải quyết triệt để lỗi ghi đè dữ liệu khi nhiều sách có cùng tên Section.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
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

    def save_knowledge_graph(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], file_name: str, chapter_name: str, section_title: str,
                            main_entities: List[str]):
        """
        - Chức năng: Lưu Graph (bao gồm Node có type và Edge có relation). Tạo chuỗi định vị tuyệt đối.
        """
        allowed_relations = {"PREREQUISITE_OF", "RELATES_TO", "PART_OF", "DESCRIBES", "VERSUS"}

        locator = f"{file_name}::{chapter_name}::{section_title}"

        with self.driver.session() as session:
            # 1. Lưu các Node và Type
            for node in nodes:
                node_name = str(node.get("name", "")).strip()
                node_type = str(node.get("type", "concept")).strip()
                if not node_name: continue
                
                session.run("""
                MERGE (n:Concept {id: $id})
                SET n.type = $type
                SET n.source_locators = CASE WHEN $loc IN coalesce(n.source_locators, []) THEN n.source_locators ELSE coalesce(n.source_locators, []) + $loc END
                """, id=node_name, type=node_type, loc=locator)

            # 2. Đánh dấu Main Entities
            for main_ent in main_entities:
                if not main_ent: continue
                session.run("""
                MERGE (c:Concept {id: $id})
                SET c.is_main = true
                SET c.source_locators = CASE WHEN $loc IN coalesce(c.source_locators, []) THEN c.source_locators ELSE coalesce(c.source_locators, []) + $loc END
                """, id=main_ent.strip(), loc=locator)

            # 3. Xây dựng các cạnh (Edges)
            for e in edges:
                source = str(e.get("source", "")).strip()
                target = str(e.get("target", "")).strip()
                raw_rel = str(e.get("relation", "RELATES_TO")).strip().upper().replace(" ", "_")

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

    def get_graph_context(self, node_names: List[str], search_mode: str = "search") -> List[Dict[str, str]]:
        """
        - Reason: Dual-mode graph traversal for different cognitive tasks (Learning vs. Q&A).
        - Function: Executes targeted Cypher queries based on the search_mode.
        - Parameters:
            - node_names (List[str]): The target entities to anchor the search.
            - search_mode (str): "semi_search" for backwards 1-hop, "search" for undirected 2-hops.
        """
        if not node_names:
            return []

        if search_mode == "semi_search":
            # Semi-Search: Look backwards 1 hop (incoming edges). Pattern: (m)-[r]->(n)
            query = """
            MATCH (m)-[r]->(n)
            WHERE n.id IN $node_names
            RETURN DISTINCT m.id AS source, type(r) AS relation, n.id AS target
            """
        else:
            # Search: Look all directions up to 2 hops. Pattern: (n)-[*1..2]-(m)
            # Unwind relationships to return individual edges instead of full paths
            query = """
            MATCH p=(n)-[*1..2]-(m)
            WHERE n.id IN $node_names
            UNWIND relationships(p) AS r
            WITH DISTINCT r
            RETURN startNode(r).id AS source, type(r) AS relation, endNode(r).id AS target
            """

        try:
            with self.driver.session() as session:
                results = session.run(query, node_names=node_names)
                return [{"source": record["source"], "relation": record["relation"], "target": record["target"]} for record in results]
        except Exception as e:
            print(f"[!] Neo4j Query Error: {e}")
            return []