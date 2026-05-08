import json
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from core.schemas import QAResponse

class MemoryManager:
    def __init__(self, max_history: int = 5):
        """
        - Reason: To maintain a rolling window of conversation history across all states.
        - Function: Stores and summarizes recent interactions to prevent prompt context overflow.
        - Usage: Instantiated inside RuntimeEngine.
        - Parameters: max_history (int) - Number of recent QA pairs to keep.
        - Returns: None
        - Alternatives: Advanced ViT Image memory rendering (Placeholder for future).
        """
        self.history = []
        self.max_history = max_history

    def add_interaction(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history * 2: # Keep last N pairs
            self.history = self.history[-(self.max_history * 2):]

    def get_history_string(self) -> str:
        if not self.history:
            return "No previous interaction in this session."
        return "\n".join([f"{item['role'].capitalize()}: {item['content']}" for item in self.history])

class SupportAgent:
    def __init__(self, llm: BaseChatModel):
        """
        - Reason: To handle out-of-band user queries without disrupting the main Learning Queue.
        - Function: Generates isolated, single-shot answers based on Graph and Chat context.
        - Usage: Called by RuntimeEngine during LOCAL_QA or GLOBAL_QA.
        """
        self.llm = llm

    def generate_answer(self, query: str, chat_history: str, graph_context: List[Dict], macro_context: str = "") -> str:
        """
        - Reason: Core logic for QA synthesis.
        - Function: Formats the Socratic prompt and invokes the LLM. Fallbacks gracefully if graph_context is empty.
        - Parameters:
            - query (str): User's question.
            - chat_history (str): Stringified recent memory.
            - graph_context (List): Data from Neo4j.
            - macro_context (str): Optional full section text (usually for GLOBAL_QA).
        - Returns: String answer.
        """
        system_prompt = """
                You are an Expert Academic Mentor. Answer the student's question directly, accurately, and pedagogically.

                [STRICT RULES - ANTI-HALLUCINATION]:
                1. GROUNDING RULE: You MUST construct your answer using ONLY the facts provided in [MACRO TEXT CONTEXT] and [GRAPH CONTEXT].
                2. OUT OF CONTEXT RULE: If the student's question asks for concepts or facts NOT present in the provided contexts, you MUST decline to answer by stating exactly: "Xin lỗi, tài liệu bài học hiện tại không đề cập đến thông tin này." DO NOT use your internal pre-trained knowledge to answer out-of-scope questions.
                3. If [GRAPH CONTEXT] is provided, use the entities and relationships to explain the underlying logic.
                4. If both contexts are EMPTY, rely purely on [CHAT HISTORY] to clarify or re-explain previous points. Do not introduce new external facts.
                5. Output strictly valid JSON.
                """

        human_prompt = """
                [CHAT HISTORY]:
                {chat_history}

                [GRAPH CONTEXT]:
                {graph_context}

                [MACRO TEXT CONTEXT]:
                {macro_context}

                [STUDENT QUESTION]:
                {query}

                [REQUIRED JSON FORMAT]:
                {{
                    "answer": "Your detailed explanation based STRICTLY on the context, or the exact refusal message if out of scope."
                }}
                """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        _input = prompt.format_messages(
            chat_history=chat_history,
            graph_context=json.dumps(graph_context, ensure_ascii=False) if graph_context else "EMPTY",
            macro_context=macro_context if macro_context else "EMPTY",
            query=query
        )

        try:
            response = self.llm.invoke(_input)
            # Basic JSON extraction (assuming robust output or using the helper from LocalLLMService)
            start = response.content.find('{')
            end = response.content.rfind('}')
            clean_json = response.content[start:end+1]
            data = json.loads(clean_json)
            return data.get("answer", "I apologize, but I encountered an error processing the answer.")
        except Exception as e:
            print(f"[!] SupportAgent Error: {e}")
            return "I am currently unable to process your request."

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
        self.memory = MemoryManager()
        self.support_agent = support_agent

    def _stream_learning_action(self, target_file: str, target_section: str, step_data: dict):
        yield {"type": "status", "message": "🔍 Đang trích xuất Macro-context từ bài học..."}
        raw_context = self.vector_db.get_section_exact(target_file, target_section)

        macro_context_str = ""
        if raw_context and isinstance(raw_context, list):
            if isinstance(raw_context[0], dict): macro_context_str = raw_context[0].get("page_content", "")
            else: macro_context_str = str(raw_context[0])
        elif isinstance(raw_context, str):
            macro_context_str = raw_context

        yield {"type": "status", "message": "🕸️ Đang quét Đồ thị User Profile (Learned/Unlearned) từ Neo4j..."}
        main_entities = step_data.get("main_entities", [])
        unlearned, learned = [], []
        if main_entities:
            for ent in main_entities:
                unlearned.extend(self.graph_db.get_unlearned_prerequisites(ent))
                learned.extend(self.graph_db.get_learned_prerequisites(ent))
            unlearned, learned = list(set(unlearned)), list(set(learned))
        
        graph_context_str = f"Unlearned: {', '.join(unlearned) if unlearned else 'None'}\nLearned: {', '.join(learned) if learned else 'None'}"

        yield from self.orchestrator.execute_teaching_step(step_data, macro_context_str, graph_context_str)

        yield {"type": "status", "message": "💾 Đang lưu tiến độ học vào User Profile Neo4j..."}
        if main_entities:
            for ent in main_entities:
                self.graph_db.mark_concept_as_learned(ent)

    def process_action(self, action_mode: str, query: str = "", target_file: str = "", target_section: str = "", step_data: dict = None):
        """
        - Reason: Unified entry point for all UI interactions.
        - Function: Executes the distinct logic flows for LEARNING, LOCAL_QA, and GLOBAL_QA.
        - Parameters:
            - action_mode (str): "LEARNING", "LOCAL_QA", or "GLOBAL_QA".
            - query (str): User input (for QA).
            - target_file/target_section: Metadata for context routing.
            - step_data: Data for the QueueOrchestrator (Learning only).
        """
        if action_mode == "LEARNING":
            print("\n[ENGINE] Dispatching to LEARNING Queue (Streaming)...")
            # Trả về Generator object thay vì yield trực tiếp trong hàm này
            return self._stream_learning_action(target_file, target_section, step_data)

        elif action_mode == "LOCAL_QA":
            print(f"\n[ENGINE] Executing LOCAL QA for query: {query}")
            self.memory.add_interaction("user", query)

            # 1. Fallback mechanism: Find anchor entities using existing HyDE search
            anchor_nodes = []
            search_results = self.vector_db.search_candidates_and_fetch_parent(
                query=query,
                llm_service=self.orchestrator.llm_service,
                target_file=target_file
            )

            if search_results and "metadata" in search_results[0]:
                raw_anchors = search_results[0]["metadata"].get("anchor_nodes", "")
                if raw_anchors:
                    anchor_nodes = [node.strip() for node in raw_anchors.split(",") if node.strip()]

            # 2. Neo4j Semi-Search (backwards 1 hop)
            graph_data = self.graph_db.get_graph_context(anchor_nodes, search_mode="semi_search") if anchor_nodes else []

            # 3. Generate Answer
            answer = self.support_agent.generate_answer(
                query=query,
                chat_history=self.memory.get_history_string(),
                graph_context=graph_data
            )
            self.memory.add_interaction("assistant", answer)
            return answer

        elif action_mode == "GLOBAL_QA":
            print(f"\n[ENGINE] Executing GLOBAL QA for query: {query}")
            self.memory.add_interaction("user", query)

            # 1. Search Anchor using existing HyDE search
            anchor_nodes = []
            macro_context = ""
            search_results = self.vector_db.search_candidates_and_fetch_parent(
                query=query,
                llm_service=self.orchestrator.llm_service,
                target_file=target_file
            )

            if search_results:
                macro_context = search_results[0].get("page_content", "")
                raw_anchors = search_results[0].get("metadata", {}).get("anchor_nodes", "")
                if raw_anchors:
                    anchor_nodes = [node.strip() for node in raw_anchors.split(",") if node.strip()]

            # 2. Neo4j Search (2 hops undirected)
            graph_data = self.graph_db.get_graph_context(anchor_nodes, search_mode="search") if anchor_nodes else []

            # 4. Generate Answer
            answer = self.support_agent.generate_answer(
                query=query,
                chat_history=self.memory.get_history_string(),
                graph_context=graph_data,
                macro_context=macro_context
            )
            self.memory.add_interaction("assistant", answer)
            return answer