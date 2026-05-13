import json
import re
from collections import deque
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from core.schemas import QueueState

# ==========================================
# PURE QUEUE ORCHESTRATOR
# ==========================================
class QueueOrchestrator:
    def __init__(self, llm_service):
        """
        - Reason: Replaces LangGraph to manage agent execution flow with zero framework overhead.
        - Function: Initializes the orchestrator with the local LLM service and an empty state.
        - Usage: Instantiated when a user starts studying a specific section.
        - Parameters:
            - llm_service (LocalLLMService): The instance to communicate with Qwen.
        - Returns: None.
        - Alternatives: LangGraph StateGraph, standard Python state machines.
        """
        self.llm_service = llm_service
        self.state = None

    def _clean_think_tags(self, text: str) -> str:
        """
        - Reason: reasoning models output thoughts in <think> tags.
        - Function: Removes everything between <think> and </think> inclusive.
        """
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def initialize_state(self, step_id: str, macro_context: str, graph_context: str = ""):
        """
        - Reason: To reset or initialize the partitioned memory (scratchpads) for a new teaching step.
        - Function: Creates a fresh QueueState object.
        - Usage: Called at the beginning of each new curriculum group/step.
        - Parameters:
            - step_id (str): The unique identifier of the current teaching step.
            - macro_context (str): The full section text.
            - graph_context (str): Learned vs unlearned prerequisites.
        - Returns: None.
        """
        self.state = QueueState(current_step_id=step_id, macro_context=macro_context, graph_context=graph_context)

    def execute_teaching_step(self, step_data: Dict[str, Any], macro_context: str, graph_context: str = ""):
        step_id = step_data.get("step_id", "unknown_step")
        self.initialize_state(step_id, macro_context, graph_context)

        raw_queue = step_data.get("required_agents", ["concept", "example"])
        agent_queue = deque(raw_queue)

        yield {"type": "status", "message": f"⚙️ Agent Queue established: {list(agent_queue)}"}

        while agent_queue:
            current_agent = agent_queue.popleft()
            yield {"type": "status", "message": f"🧠 Expert [{current_agent.upper()}] is analyzing Graph and Context..."}
            yield {"type": "agent_start", "agent": current_agent}

            lecture_content = ""
            for chunk in self._route_and_execute_agent(current_agent, step_data):
                lecture_content += chunk
                yield {"type": "chunk", "content": chunk}

            yield {"type": "agent_end", "agent": current_agent}

            # Update scratchpads
            if current_agent == "concept": self.state.concept_scratchpad.append(lecture_content)
            elif current_agent == "math": self.state.math_scratchpad.append(lecture_content)
            elif current_agent == "formula": self.state.formula_scratchpad.append(lecture_content)
            elif current_agent == "algorithm": self.state.algorithm_scratchpad.append(lecture_content)
            else: self.state.dynamic_scratchpad.append(lecture_content)

            yield {"type": "status", "message": f"📝 Summarizing [{current_agent.upper()}]'s key points into Global Memory..."}
            self._update_global_summary(current_agent, lecture_content)

    def mutate_queue(self, target_queue: deque, new_agents: List[str]):
        """
        - Reason: To allow dynamic agent injection at runtime.
        - Function: Pushes new agents to the FRONT of the queue.
        - Usage: Can be called by specific agents if they detect a gap in the student's knowledge.
        - Parameters:
            - target_queue (deque): The current running queue.
            - new_agents (List[str]): List of agent roles to inject.
        - Returns: None.
        """
        new_agents.reverse()
        target_queue.extendleft(new_agents)
        print(f"    [!] Queue Mutated! New Queue: {list(target_queue)}")

    # ---------------------------------------------------------
    # INTERNAL LLM HELPER
    # ---------------------------------------------------------
    def _stream_agent_llm(self, system_prompt: str, user_prompt: str, **kwargs):
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", user_prompt)])
        _input = prompt.format_messages(**kwargs)
        
        buffer = ""
        is_first_phase = True
        in_think_block = False
        
        try:
            for chunk in self.llm_service.chat_llm.stream(_input):
                content = chunk.content
                
                # Handle <think> blocks (DeepSeek-R1 style)
                if "<think>" in content:
                    in_think_block = True
                    # If there's text before <think>, keep it
                    content = content.split("<think>")[0]
                
                if in_think_block:
                    if "</think>" in content:
                        in_think_block = False
                        # Only keep text AFTER </think>
                        content = content.split("</think>")[-1]
                    else:
                        # Skip this chunk entirely while inside think block
                        continue

                if not content:
                    continue

                if is_first_phase:
                    buffer += content
                    if "\n" in buffer or len(buffer) > 15:
                        clean_buffer = re.sub(r'^\s*```[a-zA-Z]*\n?', '', buffer)
                        if clean_buffer:
                            yield clean_buffer
                        is_first_phase = False
                else:
                    # Strip any trailing backticks dynamically to prevent closing fences
                    if "```" in content:
                        content = content.replace("```", "")
                    yield content
        except Exception as e:
            print(f"[!] Streaming Error: {e}")
            yield "Sorry, I encountered an error while streaming the response."

    # ---------------------------------------------------------
    # AGENT ROUTING & EXECUTION
    # ---------------------------------------------------------
    def _route_and_execute_agent(self, agent_role: str, step_data: Dict[str, Any]) -> str:
        """Internal router to call the appropriate agent function based on the role."""
        content_focus = step_data.get("content_focus", "None")

        if agent_role == "concept":
            return self._run_concept_agent(content_focus)
        elif agent_role == "math":
            return self._run_math_agent(content_focus)
        elif agent_role == "formula":
            return self._run_formula_agent(content_focus)
        elif agent_role == "algorithm":
            return self._run_algorithm_agent(content_focus)
        elif agent_role == "example":
            return self._run_example_agent(content_focus)
        elif agent_role.startswith("dynamic:"):
            role_detail = agent_role.split(":")[1]
            return self._run_dynamic_agent(role_detail, content_focus)
        else:
            return f"Error: Unknown agent role '{agent_role}'"

    def _run_concept_agent(self, content_focus: str):
        system = "You are the Concept Expert. Your goal is to explain the intuition using metaphors. Return your lecture DIRECTLY IN MARKDOWN. Use $$...$$ for block math equations. CRITICAL RULE: Do NOT wrap your entire response in code blocks or triple backticks (```)."
        human = """
        [MACRO-CONTEXT (FULL TEXT)]: 
        {macro_context}
        
        [LEARNER'S GRAPH STATE]:
        {graph_context}
        
        [GLOBAL SUMMARY SO FAR]: {global_summary}
        [YOUR PREVIOUS MEMORY]: {scratchpad}

        [TASK]: Read the MACRO-CONTEXT. Your lecture MUST strictly focus on explaining:
        [MICRO-CONTEXT (FOCUS)]: "{content_focus}"

        [GRAPH INSTRUCTION]:
        If the Graph State shows UNLEARNED prerequisites, briefly explain them.
        If it shows LEARNED prerequisites, briefly remind the user.
        """
        yield from self._stream_agent_llm(
            system, human,
            macro_context=self.state.macro_context,
            graph_context=self.state.graph_context,
            global_summary=self.state.global_summary,
            scratchpad=" | ".join(self.state.concept_scratchpad),
            content_focus=content_focus
        )

    def _run_math_agent(self, content_focus: str):
        system = "You are the rigorous Math Expert. You provide step-by-step logical proofs and deep mathematical derivations. Return your lecture DIRECTLY IN MARKDOWN. Use $$...$$ for block math equations. CRITICAL RULE: Do NOT wrap your entire response in code blocks or triple backticks (```)."
        human = """
        [MACRO-CONTEXT (FULL TEXT)]: 
        {macro_context}
        
        [LEARNER'S GRAPH STATE]:
        {graph_context}
        
        [GLOBAL SUMMARY SO FAR]: {global_summary}
        [YOUR PREVIOUS MEMORY]: {scratchpad}

        [TASK]: Read the MACRO-CONTEXT. Your lecture MUST strictly focus on explaining:
        [MICRO-CONTEXT (FOCUS)]: "{content_focus}"

        [GRAPH INSTRUCTION]:
        If the Graph State shows UNLEARNED prerequisites, briefly explain them.
        If it shows LEARNED prerequisites, briefly remind the user.
        """
        yield from self._stream_agent_llm(
            system, human,
            macro_context=self.state.macro_context,
            graph_context=self.state.graph_context,
            global_summary=self.state.global_summary,
            scratchpad=" | ".join(self.state.math_scratchpad),
            content_focus=content_focus
        )

    def _run_formula_agent(self, content_focus: str):
        system = "You are the Formal Syntax Expert. You define variables, state formulas clearly using LaTeX, and establish naming conventions. Return your lecture DIRECTLY IN MARKDOWN. Use $$...$$ for block math equations. CRITICAL RULE: Do NOT wrap your entire response in code blocks or triple backticks (```)."
        human = """
        [MACRO-CONTEXT (FULL TEXT)]: 
        {macro_context}
        
        [LEARNER'S GRAPH STATE]:
        {graph_context}
        
        [GLOBAL SUMMARY SO FAR]: {global_summary}
        [YOUR PREVIOUS MEMORY]: {scratchpad}

        [TASK]: Read the MACRO-CONTEXT. Your lecture MUST strictly focus on explaining:
        [MICRO-CONTEXT (FOCUS)]: "{content_focus}"

        [GRAPH INSTRUCTION]:
        If the Graph State shows UNLEARNED prerequisites, briefly explain them.
        If it shows LEARNED prerequisites, briefly remind the user.
        """
        yield from self._stream_agent_llm(
            system, human,
            macro_context=self.state.macro_context,
            graph_context=self.state.graph_context,
            global_summary=self.state.global_summary,
            scratchpad=" | ".join(self.state.formula_scratchpad),
            content_focus=content_focus
        )

    def _run_algorithm_agent(self, content_focus: str):
        system = "You are the Algorithm Expert. You translate mathematical theory into computational logic, flowcharts, or pseudo-code. Return your lecture DIRECTLY IN MARKDOWN. Use $$...$$ for block math equations. CRITICAL RULE: Do NOT wrap your entire response in code blocks or triple backticks (```)."
        human = """
        [MACRO-CONTEXT (FULL TEXT)]: 
        {macro_context}
        
        [LEARNER'S GRAPH STATE]:
        {graph_context}
        
        [GLOBAL SUMMARY SO FAR]: {global_summary}
        [YOUR PREVIOUS MEMORY]: {scratchpad}

        [TASK]: Read the MACRO-CONTEXT. Your lecture MUST strictly focus on explaining:
        [MICRO-CONTEXT (FOCUS)]: "{content_focus}"

        [GRAPH INSTRUCTION]:
        If the Graph State shows UNLEARNED prerequisites, briefly explain them.
        If it shows LEARNED prerequisites, briefly remind the user.
        """
        yield from self._stream_agent_llm(
            system, human,
            macro_context=self.state.macro_context,
            graph_context=self.state.graph_context,
            global_summary=self.state.global_summary,
            scratchpad=" | ".join(self.state.algorithm_scratchpad),
            content_focus=content_focus
        )

    def _run_example_agent(self, content_focus: str):
        system = "You are the Practical Example Expert. You provide concrete, numerical, or real-world examples to apply the theory just discussed. Return your lecture DIRECTLY IN MARKDOWN. Use $$...$$ for block math equations. CRITICAL RULE: Do NOT wrap your entire response in code blocks or triple backticks (```)."
        human = """
        [MACRO-CONTEXT (FULL TEXT)]: 
        {macro_context}
        
        [LEARNER'S GRAPH STATE]:
        {graph_context}
        
        [GLOBAL SUMMARY SO FAR]: {global_summary}
        [YOUR PREVIOUS MEMORY]: {scratchpad}

        [TASK]: Read the MACRO-CONTEXT. Your lecture MUST strictly focus on explaining:
        [MICRO-CONTEXT (FOCUS)]: "{content_focus}"

        [GRAPH INSTRUCTION]:
        If the Graph State shows UNLEARNED prerequisites, briefly explain them.
        If it shows LEARNED prerequisites, briefly remind the user.
        """
        yield from self._stream_agent_llm(
            system, human,
            macro_context=self.state.macro_context,
            graph_context=self.state.graph_context,
            global_summary=self.state.global_summary,
            scratchpad=" | ".join(self.state.dynamic_scratchpad),
            content_focus=content_focus
        )

    def _run_dynamic_agent(self, role_detail: str, content_focus: str):
        system = f"You are an expert acting as: {role_detail}. Provide insights entirely from this specific viewpoint. Return your lecture DIRECTLY IN MARKDOWN. Use $$...$$ for block math equations. CRITICAL RULE: Do NOT wrap your entire response in code blocks or triple backticks (```)."
        human = """
        [MACRO-CONTEXT (FULL TEXT)]: 
        {macro_context}
        
        [LEARNER'S GRAPH STATE]:
        {graph_context}
        
        [GLOBAL SUMMARY SO FAR]: {global_summary}
        [YOUR PREVIOUS MEMORY]: {scratchpad}

        [TASK]: Read the MACRO-CONTEXT. Your lecture MUST strictly focus on explaining:
        [MICRO-CONTEXT (FOCUS)]: "{content_focus}"

        [GRAPH INSTRUCTION]:
        If the Graph State shows UNLEARNED prerequisites, briefly explain them.
        If it shows LEARNED prerequisites, briefly remind the user.
        """
        yield from self._stream_agent_llm(
            system, human,
            macro_context=self.state.macro_context,
            graph_context=self.state.graph_context,
            global_summary=self.state.global_summary,
            scratchpad=" | ".join(self.state.dynamic_scratchpad),
            content_focus=content_focus
        )

    def _update_global_summary(self, agent_role: str, recent_content: str):
        system = "You are a Summarization API. Output EXACTLY ONE short, concise sentence summarizing the text. Do not output anything else."
        human = """
        Read the following lecture snippet by the {role} expert. 
        Compress its core meaning into EXACTLY ONE short, concise sentence.

        [SNIPPET]: {content}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", human)
        ])

        _input = prompt.format_messages(role=agent_role, content=recent_content[:1500])
        
        try:
            raw_response = self.llm_service.chat_llm.invoke(_input)
            content = raw_response.content.strip()
            compressed_thought = self._clean_think_tags(content)
            self.state.global_summary += f"[{agent_role.upper()}]: {compressed_thought}\n"
        except Exception as e:
            print(f"[!] Summarization Error: {e}")
            self.state.global_summary += f"[{agent_role.upper()} completed their part]\n"