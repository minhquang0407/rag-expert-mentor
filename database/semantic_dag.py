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
                node_desc = str(node.get("description", "")).strip()
                if not node_name: continue
                
                session.run("""
                MERGE (n:Concept {id: $id})
                SET n.type = $type
                SET n.description = $desc
                SET n.source_locators = CASE WHEN $loc IN coalesce(n.source_locators, []) THEN n.source_locators ELSE coalesce(n.source_locators, []) + $loc END
                """, id=node_name, type=node_type, desc=node_desc, loc=locator)

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
                CREATE (c1)-[rel:$(rel)]->(c2)
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
            # Semi-Search: Look backwards 1 hop. Pattern: (m)-[r]->(n)
            query = """
            MATCH (m)-[r]->(n)
            WHERE n.id IN $node_names
            RETURN DISTINCT 
                m.id AS source, m.description AS source_desc,
                type(r) AS relation, 
                n.id AS target, n.description AS target_desc
            """
        else:
            # Search: Look all directions up to 2 hops.
            query = """
            MATCH p=(n)-[*1..2]-(m)
            WHERE n.id IN $node_names
            UNWIND relationships(p) AS r
            WITH DISTINCT r
            RETURN 
                startNode(r).id AS source, startNode(r).description AS source_desc,
                type(r) AS relation, 
                endNode(r).id AS target, endNode(r).description AS target_desc
            """

        try:
            with self.driver.session() as session:
                results = session.run(query, node_names=node_names)
                return [
                    {
                        "source": record["source"], 
                        "source_desc": record["source_desc"],
                        "relation": record["relation"], 
                        "target": record["target"],
                        "target_desc": record["target_desc"]
                    } for record in results
                ]
        except Exception as e:
            print(f"[!] Neo4j Query Error: {e}")
            return []


    def get_recent_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Fetches the N most recent chat summaries for context window."""
        cypher = """
        MATCH (u:User {id: $user_id})-[:HAS_TURN]->(t:ChatTurn)
        RETURN t.id AS id, t.raw_query AS query, t.summary AS summary
        ORDER BY t.timestamp DESC LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, user_id=user_id, limit=limit)
            # Reverse to chronological order (oldest to newest in the context window)
            return [{"id": r["id"], "query": r["query"], "summary": r["summary"]} for r in result][::-1]

    def get_raw_chat_turns(self, turn_ids: List[str]) -> List[Dict]:
        """Fetches raw data when LLM Router requests it."""
        if not turn_ids: return []
        cypher = """
        MATCH (t:ChatTurn) WHERE t.id IN $turn_ids
        RETURN t.id AS id, t.raw_query AS query, t.raw_answer AS answer
        """
        with self.driver.session() as session:
            result = session.run(cypher, turn_ids=turn_ids)
            return [{"id": r["id"], "query": r["query"], "answer": r["answer"]} for r in result]

    def get_raw_chat_turns_by_user(self, user_id: str) -> List[Dict]:
        cypher = """
        MATCH (u:User {id: $user_id})-[:HAS_TURN]->(t:ChatTurn)
        RETURN t.id AS id, t.raw_query AS query, t.raw_answer AS answer
        ORDER BY t.timestamp ASC
        """
        with self.driver.session() as session:
            result = session.run(cypher, user_id=user_id)
            return [{"id": r["id"], "query": r["query"], "answer": r["answer"]} for r in result]

    def save_chat_turn(self, user_id: str, turn_id: str, query: str, raw_answer: str, summary: str, concept_ids: list = None, target_file: str = "", target_section: str = ""):
        """
        - Lí do tại sao dùng: Lưu trữ Episodic Memory vào Neo4j làm Nguồn Sự Thật Duy Nhất (SSOT).
        - Chức năng: Lưu lượt chat, liên kết với User, NHIỀU Concept và gắn metadata Section.
        - Cách dùng: Gọi sau khi LLM xử lý xong một lượt học/hỏi đáp.
        """
        cypher = """
        MERGE (u:User {id: $user_id})
        CREATE (t:ChatTurn {
            id: $turn_id, 
            raw_query: $query, 
            raw_answer: $raw_answer, 
            summary: $summary, 
            file: $target_file,
            section: $target_section,
            timestamp: timestamp()
        })
        MERGE (u)-[:HAS_TURN]->(t)
        
        WITH t, u
        MATCH (u)-[:HAS_TURN]->(prev:ChatTurn)
        WHERE prev.id <> t.id
        WITH t, prev ORDER BY prev.timestamp DESC LIMIT 1
        MERGE (prev)-[:NEXT_TURN]->(t)
        """
        with self.driver.session() as session:
            session.run(cypher, parameters={"user_id": user_id, "turn_id": turn_id, "query": query, "raw_answer": raw_answer, "summary": summary, "target_file": target_file, "target_section": target_section})
            
            if concept_ids:
                link_cypher = """
                MATCH (t:ChatTurn {id: $turn_id})
                MERGE (c:Concept {id: $concept_id})
                MERGE (t)-[:DISCUSSED]->(c)
                """
                for cid in concept_ids:
                    if cid and str(cid).strip():
                        session.run(link_cypher, turn_id=turn_id, concept_id=str(cid).strip())

    def get_history_by_section(self, user_id: str, target_file: str, target_section: str, limit: int = 50) -> List[
        Dict]:

        cypher = """
        MATCH (u:User {id: $user_id})-[:HAS_TURN]->(t:ChatTurn)
        WHERE t.file = $target_file AND t.section = $target_section
        RETURN t.raw_query AS query, t.raw_answer AS answer
        ORDER BY t.timestamp DESC LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, user_id=user_id, target_file=target_file, target_section=target_section,
                                 limit=limit)
            # Reverse list to render oldest top, newest bottom
            return [{"query": r["query"], "answer": r["answer"]} for r in result][::-1]

    def delete_source(self, source_name: str) -> None:
        """
        - Function: Removes all knowledge graph nodes and chat turns associated with a file.
        """
        with self.driver.session() as session:
            # 1. Delete ChatTurns associated with this file
            session.run("MATCH (t:ChatTurn {file: $source}) DETACH DELETE t", source=source_name)

            # 2. Update Concepts: Remove locators from this file
            # If after removal, source_locators is empty, the node is orphaned from this perspective.
            session.run("""
            MATCH (n:Concept)
            WHERE any(loc IN n.source_locators WHERE loc STARTS WITH $prefix)
            SET n.source_locators = [loc IN n.source_locators WHERE NOT loc STARTS WITH $prefix]
            """, prefix=f"{source_name}::")

            # 3. Delete Concepts that no longer have any source locators
            session.run("""
            MATCH (n:Concept)
            WHERE size(n.source_locators) = 0
            DETACH DELETE n
            """)
            print(f"[*] Cleaned up Neo4j for source: {source_name}")
