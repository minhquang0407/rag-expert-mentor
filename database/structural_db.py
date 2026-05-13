import os
import uuid
import time
from typing import Dict, Any, List
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from core.interfaces import IVectorStore, ILLMService


# ==========================================
# QDRANT VECTOR STORE MANAGER
# ==========================================
class QdrantVectorStore(IVectorStore):
    def __init__(self, host: str = "localhost", port: int = 6333, api_key: str = None, collection_name: str = "math_curriculum_v4"):
        """
        - Reason: To manage vector embeddings and handle collection initialization conflicts.
        - Function: Initializes the Qdrant client, embedding model, and robustly ensures required collections exist.
        - Usage: Instantiated once at application startup.
        - Parameters:
            - collection_name (str): The base name for the collections (default: "math_curriculum_v4").
        - Returns: None.
        - Alternatives: Manual directory cleanup before startup.
        """
        self.vector_name = "fast-paraphrase-multilingual-minilm-l12-v2"
        self.parent_coll = collection_name
        self.child_coll = f"{collection_name}_questions"
        self.memory_coll = "user_memory_v1"

        # Connect to Qdrant (Cloud or Local)
        if host.startswith("http"):
            self.client = QdrantClient(url=host, api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port, api_key=api_key if api_key else None, https=False)

        # Extremely fast and lightweight local embedding model
        self.embed_model = FastEmbedEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        try:
            self.client.set_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        except Exception:
            pass

        # Robust creation of collections
        for coll in [self.parent_coll, self.child_coll, self.memory_coll]:
            try:
                # Check if collection is already registered in Qdrant's state
                if not self.client.collection_exists(coll):
                    print(f"[*] Creating new collection: {coll}")
                    self.client.create_collection(
                        collection_name=coll,
                        # [FIXED]: Declare vectors_config as a dictionary to create Named Vector from scratch
                        vectors_config={self.vector_name: VectorParams(size=384, distance=Distance.COSINE)},
                    )
                
                for field in ["source", "section", "type"]:
                    try:
                        self.client.create_payload_index(
                            collection_name=coll,
                            field_name=field,
                            field_schema=models.PayloadSchemaType.KEYWORD
                        )
                    except Exception:
                        pass # Index already exists or is being created

            except Exception as e:
                # If "File exists" error occurs, it means the directory is there but Qdrant was confused.
                # We log it and move on, as the existing directory will be used.
                if "File exists" in str(e):
                    print(f"[!] Warning: Directory for {coll} already exists on disk. Skipping creation.")
                else:
                    print(f"[X] Critical error creating collection {coll}: {e}")

                
            try:
                self.client.create_payload_index(
                    collection_name=coll,
                    field_name="source",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def upsert_section(self, text: str, metadata: dict, parent_id: str) -> None:
        """
        - Reason: To store the macro-context of a textbook section.
        - Function: Embeds raw text and saves it as a parent node.
        - Usage: Called during the data ingestion phase.
        - Parameters: text, metadata, parent_id.
        - Returns: None.
        """
        vector = self.embed_model.embed_query(text)
        payload = metadata.copy()
        payload["parent_id"] = parent_id
        payload["type"] = "section_anchor"
        payload["page_content"] = text

        self.client.upsert(
            collection_name=self.parent_coll,
            # [FIXED]: Pass vector as a dictionary matching the named vector schema
            points=[PointStruct(id=parent_id, vector={self.vector_name: vector}, payload=payload)]
        )

    def upsert_questions(self, qa_pairs: List[Dict[str, str]], parent_id: str, source_file: str) -> None:
        """
        - Function: Links multiple hypothetical questions and their key knowledge to a parent section.
        """
        if not qa_pairs:
            return

        questions = [qa["question"] for qa in qa_pairs]
        vectors = list(self.embed_model.embed_documents(questions))
        points = []

        for idx, (qa, vec) in enumerate(zip(qa_pairs, vectors)):
            valid_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_q_{idx}"))
            payload = {
                "page_content": qa["question"],
                "key_knowledge": qa.get("key_knowledge", ""),
                "parent_id": parent_id,
                "source": source_file,
                "type": "question"
            }
            # [FIXED]: Pass vector as a dictionary matching the named vector schema
            points.append(PointStruct(id=valid_id, vector={self.vector_name: vec}, payload=payload))

        self.client.upsert(
            collection_name=self.child_coll,
            points=points
        )

    def upsert_curriculum_group(self, group_data: dict, parent_id: str, source_file: str, chapter: str,
                                section: str) -> None:
        """
        - Reason: To store the AOT roadmap and agent queues.
        - Function: Extracts the pre-assigned agents and saves them into the payload.
        - Usage: Core function for the new roadmap-driven architecture.
        - Parameters: group_data, parent_id, source_file, chapter, section.
        - Returns: None.
        """
        # Prioritize content_focus from the new schema
        vector_text = group_data.get("content_focus", "Empty teaching step")

        vector = self.embed_model.embed_query(vector_text)

        # Use deterministic UUID based on parent section and sequence ID to enforce OVERWRITE.
        seq_id = group_data.get("seq_id", 0)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_step_{seq_id}"))
        # Expose agent queue for the orchestrator
        required_agents = group_data.get("required_agents", ["concept", "formula", "example"])

        payload = {
            "source": source_file,
            "chapter": chapter,
            "section": section,
            "parent_id": parent_id,
            "type": "curriculum_group",
            "required_agents": required_agents,
            "curriculum_data": group_data
        }

        self.client.upsert(
            collection_name=self.parent_coll,
            # [FIXED]: Pass vector as a dictionary matching the named vector schema
            points=[PointStruct(id=point_id, vector={self.vector_name: vector}, payload=payload)]
        )

    def get_curriculum_groups(self, target_file: str, target_section: str) -> List[Dict[str, Any]]:
        """
        - Function: Retrieves pre-compiled roadmap steps.
        - Returns: Sorted list of teaching steps.
        """
        try:
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
                if r.payload and "curriculum_data" in r.payload:
                    groups.append(r.payload["curriculum_data"])

            groups.sort(key=lambda x: x.get("seq_id", 0))
            return groups
        except Exception as e:
            print(f"[!] Qdrant Scroll Error: {str(e)}")
            # Return empty list to prevent UI crash
            return []

    def get_section_exact(self, target_file: str, target_section: str) -> List[Dict[str, Any]]:
        """
        - Function: Direct text retrieval by metadata match.
        """
        conditions = []
        if target_file:
            conditions.append(models.FieldCondition(key="source", match=models.MatchValue(value=target_file)))
        if target_section:
            conditions.append(models.FieldCondition(key="section", match=models.MatchValue(value=target_section)))

        filter_query = models.Filter(must=conditions) if conditions else None

        records, _ = self.client.scroll(
            collection_name=self.parent_coll,
            scroll_filter=filter_query,
            limit=1000,
            with_payload=True
        )

        return [{"page_content": r.payload.get("page_content", ""), "metadata": r.payload} for r in records]

    def search_candidates_and_fetch_parent(self, query: str, llm_service: ILLMService, target_file: str = "") -> List[Dict[str, Any]]:
        """
        - Function: Advanced HyDE search with semantic reranking.
        """
        conditions = []
        if target_file:
            conditions.append(models.FieldCondition(key="source", match=models.MatchValue(value=target_file)))

        filter_query = models.Filter(must=conditions) if conditions else None
        query_vector = self.embed_model.embed_query(query)

        response = self.client.query_points(
            collection_name=self.child_coll,
            query=query_vector,
            using=self.vector_name,
            query_filter=filter_query,
            limit=5
        )
        results = response.points

        if not results:
            return []

        candidates = [
            {
                "question": r.payload.get("page_content", ""), 
                "parent_id": r.payload.get("parent_id"),
                "key_knowledge": r.payload.get("key_knowledge", "")
            }
            for r in results
        ]

        best_parent_ids = llm_service.rerank_candidate_questions(query, candidates)
        if not best_parent_ids:
            return []

        # Fetch all unique parent sections (limit to top 2 for token safety)
        best_parent_ids = best_parent_ids[:2]
        
        parent_records, _ = self.client.scroll(
            collection_name=self.parent_coll,
            scroll_filter=models.Filter(must=[models.HasIdCondition(has_id=best_parent_ids)]),
            limit=len(best_parent_ids),
            with_payload=True
        )

        final_results = []
        for r in parent_records:
            meta = r.payload.copy()
            
            # Find any candidate that points to this parent to get its micro-context
            related_candidate = next((c for c in candidates if c["parent_id"] == r.id), candidates[0])
            meta["matched_knowledge"] = related_candidate.get("key_knowledge", "")
            
            final_results.append({"page_content": r.payload.get("page_content", ""), "metadata": meta})

        return final_results

    def upsert_user_memory(self, user_id: str, turn_id: str, query: str, answer: str, summary: str):
        # Embed the summary (or query + summary) for semantic matching
        vector = self.embed_model.embed_query(summary)
        payload = {
            "user_id": user_id,
            "turn_id": turn_id,
            "raw_query": query,
            "raw_answer": answer,
            "summary": summary,
            "type": "episodic_memory",
            "timestamp": int(time.time())
        }
        self.client.upsert(
            collection_name=self.memory_coll,
            points=[PointStruct(id=turn_id, vector={self.vector_name: vector}, payload=payload)]
        )

    def search_semantic_memory(self, user_id: str, query: str, limit: int = 5) -> List[Dict]:
        query_vector = self.embed_model.embed_query(query)
        # MUST filter by user_id for multi-tenant isolation
        user_filter = models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))])
        
        response = self.client.query_points(
            collection_name=self.memory_coll,
            query= query_vector,
            using=self.vector_name,
            query_filter=user_filter,
            limit=limit
        )
        
        results = response.points

        return [r.payload for r in results if hasattr(r, 'payload') and r.payload]

    def delete_user_history(self, user_id: str = "guest_01") -> None:

        delete_filter = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
            ]
        )
        try:
            self.client.delete(
                collection_name=self.memory_coll,
                points_selector=models.FilterSelector(filter=delete_filter)
            )
            print(f"[Qdrant] Removed episodic memory about: {user_id}")
        except Exception as e:
            print(f"[Qdrant] Error when removing history of {user_id}: {e}")

    def get_section_questions(self, parent_id: str) -> List[str]:
        """
        - Function: Retrieves all pre-generated questions for a specific section.
        """
        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="parent_id", match=models.MatchValue(value=parent_id))
            ]
        )
        try:
            records, _ = self.client.scroll(
                collection_name=self.child_coll,
                scroll_filter=query_filter,
                limit=20,
                with_payload=True
            )
            return [r.payload.get("page_content", "") for r in records if r.payload]
        except Exception as e:
            print(f"[!] Qdrant get_section_questions Error: {e}")
            return []

    def delete_source(self, source_name: str) -> None:
        """
        - Function: Deletes all points associated with a specific source file across collections.
        """
        delete_filter = models.Filter(
            must=[
                models.FieldCondition(key="source", match=models.MatchValue(value=source_name))
            ]
        )
        
        for coll in [self.parent_coll, self.child_coll]:
            try:
                self.client.delete(
                    collection_name=coll,
                    points_selector=models.FilterSelector(filter=delete_filter)
                )
                print(f"[*] Deleted records for {source_name} from {coll}")
            except Exception as e:
                print(f"[!] Error deleting {source_name} from {coll}: {e}")
