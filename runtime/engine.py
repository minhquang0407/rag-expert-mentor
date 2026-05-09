import json
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from core.schemas import QAResponse

import uuid

class SupportAgent:
    def __init__(self, llm: BaseChatModel):
        """
        - Reason: To handle out-of-band user queries without disrupting the main Learning Queue.
        - Function: Generates isolated, single-shot answers based on Graph and Chat context.
        - Usage: Called by RuntimeEngine during LOCAL_QA or GLOBAL_QA.
        """
        self.llm = llm

    def summarize_turn(self, query: str, answer: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", 'You are an expert technical summarizer. Summarize the following Q&A by extracting the core concepts and key technical details into 2-3 concise bullet points. You MUST output strictly valid JSON in this exact format: {{"summary": "- Bullet 1\\n- Bullet 2\\n- Bullet 3"}}'),
            ("human", "Query: {query}\nAnswer: {answer}")
        ])
        try:
            response = self.llm.invoke(prompt.format_messages(query=query, answer=answer))
            text = response.content
            # Try to parse JSON and extract 'summary'
            try:
                data = json.loads(text)
                if "summary" in data:
                    return str(data["summary"])
                elif "action" in data:
                    return str(data)
                return str(data)
            except json.JSONDecodeError:
                return text
        except Exception as e:
            return f"Summary unavailable: {e}"

    def route_and_answer(self, query: str, semantic_memory: list, recent_history: list, graph_context: list = None, macro_context: str = "", micro_context: str = "", raw_details: list = None) -> dict:
        system_prompt = """
        You are an Expert Academic Mentor. 
        You have access to [SEMANTIC MEMORY] (related past Q&A summaries) and [RECENT HISTORY] (latest chat context).
        If you need the exact full-text details of a past conversation to answer, return:
        {{"action": "fetch_raw", "turn_ids": ["turn_id_1", ...]}}
        
        Otherwise, answer the question using the available context. Output strictly valid JSON:
        {{"action": "answer", "response": "Your detailed explanation..."}}
        """

        human_prompt = """
        [SEMANTIC MEMORY (Qdrant)]:
        {semantic_memory}

        [RECENT HISTORY (Neo4j)]:
        {recent_history}
        
        [RAW DETAILS FETCHED]:
        {raw_details}

        [GRAPH CONTEXT]:
        {graph_context}

        [MICRO CONTEXT (Targeted Answer)]:
        {micro_context}

        [MACRO TEXT CONTEXT (Broad Background)]:
        {macro_context}

        [STUDENT QUESTION]:
        {query}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        # Truncate context to prevent token overflow (which causes empty responses or 40k-token runaways)
        safe_macro = (macro_context[:3000] + "...[truncated]") if macro_context and len(macro_context) > 3000 else (macro_context or "EMPTY")
        safe_micro = (micro_context[:1000] + "...[truncated]") if micro_context and len(micro_context) > 1000 else (micro_context or "EMPTY")
        safe_semantic = json.dumps(semantic_memory[:5], ensure_ascii=False) if semantic_memory else "EMPTY"
        safe_recent = json.dumps(recent_history[:5], ensure_ascii=False) if recent_history else "EMPTY"
        safe_graph = json.dumps(graph_context[:5], ensure_ascii=False) if graph_context else "EMPTY"
        safe_raw = json.dumps(raw_details[:3], ensure_ascii=False) if raw_details else "NONE"

        _input = prompt.format_messages(
            semantic_memory=safe_semantic,
            recent_history=safe_recent,
            raw_details=safe_raw,
            graph_context=safe_graph,
            macro_context=safe_macro,
            micro_context=safe_micro,
            query=query
        )

        try:
            response = self.llm.invoke(_input, max_tokens=2048)
            content = response.content.strip()
            start = content.find('{')
            end = content.rfind('}')
            
            if start == -1:
                # Fallback: if no JSON structure at all, treat as text answer
                return {"action": "answer", "response": content}
            
            if end == -1:
                # Handle truncated JSON: Add missing closing brace
                print("[!] Truncated JSON detected, attempting repair...")
                clean_json = content[start:]
                # Check if we need to close a quote first
                if clean_json.count('"') % 2 != 0:
                    clean_json += '"'
                clean_json += '}'
            else:
                clean_json = content[start:end+1]
                
            parsed = json.loads(clean_json)
            
            # Normalize: LLM may use 'answer', 'text', or 'content' instead of 'response'
            if "response" not in parsed and parsed.get("action") != "fetch_raw":
                for alt_key in ["answer", "text", "content", "reply"]:
                    if alt_key in parsed:
                        parsed["response"] = parsed.pop(alt_key)
                        break
                else:
                    # If no known key found, use the entire raw LLM output as the response
                    non_action_values = [v for k, v in parsed.items() if k != "action" and isinstance(v, str)]
                    if non_action_values:
                        parsed["response"] = non_action_values[0]
            
            return parsed
        except Exception as e:
            print(f"[!] SupportAgent Error: {e}")
            # Fallback: return the raw LLM text as the answer if JSON parsing fails entirely
            raw_text = response.content if 'response' in dir() else "I am currently unable to process your request."
            return {"action": "answer", "response": raw_text}

class RuntimeEngine:
    def __init__(self, orchestrator, vector_db, graph_db, support_agent):
        """
        - Reason: The Central Dispatcher (Air Traffic Controller) of the system.
        - Function: Routes actions to LEARNING or QA, manages memory, and handles DB retrievals.
        - Usage: Instantiated once at the application entry point (e.g., app.py).
        - Parameters: Instances of QueueOrchestrator, QdrantVectorStore, Neo4jGraphStore, SupportAgent.
        """
        self.orchestrator = orchestrator
        self.vector_db = vector_db
        self.graph_db = graph_db
        self.support_agent = support_agent

    def save_learning_turn(self, user_id: str, target_file: str, target_section: str, step_data: dict, full_lecture_text: str):
        """
        - Reason: Decouples lecture persistence logic from the UI streaming flow.
        - Function: Assigns a static summary, saves raw text to Qdrant, and links all entities in Neo4j.
        """
        turn_id = str(uuid.uuid4())
        seq_id = step_data.get("seq_id", 0)
        
        # 1. Format the raw query identifier
        raw_query = f"Learn {target_section}_{seq_id}"
        
        # 2. Static summary (saves LLM tokens by avoiding a summarize call)
        summary = f"[LECTURE] System completed teaching: {target_section} (Step {seq_id})"
        
        # 3. Save to Qdrant (full lecture text)
        self.vector_db.upsert_user_memory(user_id, turn_id, raw_query, full_lecture_text, summary)
        
        # 4. Save to Neo4j (link ALL main_entities)
        main_entities = step_data.get("main_entities", [])
        self.graph_db.save_chat_turn(
            user_id=user_id,
            turn_id=turn_id,
            query=raw_query,
            raw_answer=full_lecture_text,
            summary=summary,
            concept_ids=main_entities,
            target_file=target_file,
            target_section=target_section
        )

    def _stream_learning_action(self, target_file: str, target_section: str, step_data: dict, user_id: str):
        yield {"type": "status", "message": "Extracting Macro-context from the lesson..."}
        raw_context = self.vector_db.get_section_exact(target_file, target_section)

        macro_context_str = ""
        if raw_context and isinstance(raw_context, list):
            if isinstance(raw_context[0], dict): macro_context_str = raw_context[0].get("page_content", "")
            else: macro_context_str = str(raw_context[0])
        elif isinstance(raw_context, str):
            macro_context_str = raw_context

        yield {"type": "status", "message": "Scanning User Profile Graph (Learned/Unlearned) from Neo4j..."}
        main_entities = step_data.get("main_entities", [])
        unlearned, learned = [], []
        if main_entities:
            for ent in main_entities:
                unlearned.extend(self.graph_db.get_unlearned_prerequisites(ent))
                learned.extend(self.graph_db.get_learned_prerequisites(ent))
            unlearned, learned = list(set(unlearned)), list(set(learned))
        
        graph_context_str = f"Unlearned: {', '.join(unlearned) if unlearned else 'None'}\nLearned: {', '.join(learned) if learned else 'None'}"

        # [FIXED]: Accumulate chunks to build the full text before saving
        full_lecture_text = ""
        for event in self.orchestrator.execute_teaching_step(step_data, macro_context_str, graph_context_str):
            if event["type"] == "chunk":
                full_lecture_text += event["content"]
            elif event["type"] == "agent_end":
                # Add formatting separators to match UI display
                full_lecture_text += f"\n\n---\n\n"
            
            # Yield the event forward to the UI
            yield event

        # [FIXED]: Call the dedicated save logic with the fully accumulated text
        yield {"type": "status", "message": "Saving lecture to Neo4j and Qdrant..."}
        self.save_learning_turn(user_id, target_file, target_section, step_data, full_lecture_text)

        yield {"type": "status", "message": "Marking learning progress in User Profile..."}
        if main_entities:
            for ent in main_entities:
                self.graph_db.mark_concept_as_learned(ent, user_id=user_id)

    def process_action(self, action_mode: str, query: str = "", target_file: str = "", target_section: str = "", step_data: dict = None, user_id: str = "guest_01"):
        """
        - Reason: Unified entry point for all UI interactions.
        - Function: Executes the distinct logic flows for LEARNING, LOCAL_QA, and GLOBAL_QA.
        """
        if action_mode == "LEARNING":
            print("\n[ENGINE] Dispatching to LEARNING Queue (Streaming)...")
            return self._stream_learning_action(target_file, target_section, step_data, user_id)

        elif action_mode in ["LOCAL_QA", "GLOBAL_QA"]:
            print(f"\n[ENGINE] Executing {action_mode} for query: {query}")
            
            # 1. Fetch Hybrid Memory
            semantic_mem = self.vector_db.search_semantic_memory(user_id, query, limit=5)
            recent_mem = self.graph_db.get_recent_history(user_id, limit=5)

            # 2. Search Anchor using existing HyDE search
            anchor_nodes = []
            macro_context = ""
            micro_context = ""
            search_results = self.vector_db.search_candidates_and_fetch_parent(
                query=query,
                llm_service=self.orchestrator.llm_service,
                target_file=target_file
            )

            if search_results:
                macro_list = []
                micro_list = []
                all_anchors = set()
                
                for res in search_results:
                    macro_list.append(res.get("page_content", ""))
                    if "metadata" in res:
                        micro_list.append(res["metadata"].get("matched_knowledge", ""))
                        raw_anchors = res["metadata"].get("anchor_nodes", "")
                        if raw_anchors:
                            for node in raw_anchors.split(","):
                                all_anchors.add(node.strip())
                
                macro_context = "\n---\n".join(macro_list)
                micro_context = "\n---\n".join(micro_list)
                anchor_nodes = list(all_anchors)

            # 3. Neo4j Search
            search_mode = "search" if action_mode == "GLOBAL_QA" else "semi_search"
            graph_data = self.graph_db.get_graph_context(anchor_nodes, search_mode=search_mode) if anchor_nodes else []

            # 4. Hybrid Routing Phase 1
            route_res = self.support_agent.route_and_answer(
                query=query, 
                semantic_memory=semantic_mem, 
                recent_history=recent_mem, 
                graph_context=graph_data, 
                macro_context=macro_context,
                micro_context=micro_context
            )

            if route_res.get("action") == "fetch_raw":
                print(f"[*] Agent requested raw history for turns: {route_res.get('turn_ids')}")
                raw_details = self.graph_db.get_raw_chat_turns(route_res.get("turn_ids", []))
                # Hybrid Routing Phase 2 (with raw data)
                route_res = self.support_agent.route_and_answer(
                    query=query, 
                    semantic_memory=semantic_mem, 
                    recent_history=recent_mem, 
                    graph_context=graph_data, 
                    macro_context=macro_context,
                    micro_context=micro_context,
                    raw_details=raw_details
                )

            answer = route_res.get("response", "No answer generated.")

            # 5. Summarize and Save Memory
            turn_id = str(uuid.uuid4())
            summary = self.support_agent.summarize_turn(query, answer)

            self.vector_db.upsert_user_memory(user_id, turn_id, query, answer, summary)
            self.graph_db.save_chat_turn(
                user_id=user_id,
                turn_id=turn_id,
                query=query,
                raw_answer=answer,
                summary=summary,
                concept_ids=list(anchor_nodes) if anchor_nodes else [],
                target_file=target_file,
                target_section=target_section
            )

            return answer