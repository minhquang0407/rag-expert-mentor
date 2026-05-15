from core.schemas import AgentTask, BlackboardState
from agents.base import BaseAgent


class SpecialistAgent(BaseAgent):
    """Reusable base class for teaching specialist agents."""

    teaching_style: str = "Explain clearly and accurately."

    def _build_prompts(self, task: AgentTask, blackboard: BlackboardState):
        system_prompt = (
            f"You are {self.name}, a {self.role}. "
            f"Goal: {self.goal}\n"
            "Return the lecture directly in Markdown. "
            "Use Streamlit/KaTeX-compatible LaTeX only: inline math must use $...$ and display equations must use $$ on its own line before and after the equation. "
            "For matrices, always output a complete block like $$\\begin{pmatrix} ... \\end{pmatrix}$$ with both begin and end tags. "
            "Do not emit partial LaTeX fragments, raw ampersand rows, or orphan closing delimiters. "
            "The CURRENT SECTION and CURRENT FOCUS are authoritative. Never switch to a previous topic from memory or recent history. "
            "Do not wrap the whole response in triple backticks."
        )
        human_prompt = f"""
[CURRENT SECTION]
{blackboard.target_section or 'EMPTY'}

[TEACHING STYLE]
{self.teaching_style}

[ROLE-SPECIFIC FOCUS]
You are the {self.name} specialist. Only cover the part of the lesson that matches this role.
- concept: intuition, definitions, metaphors, misconceptions. Do NOT include formulas, derivations, pseudocode, or worked examples.
- formula: symbols, notation, equations, variable meanings. Do NOT repeat broad intuition or examples.
- math: derivations, proof logic, assumptions. Do NOT provide broad conceptual metaphors.
- algorithm: procedures, pseudocode, complexity, edge cases. Do NOT repeat conceptual introduction.
- example: worked examples, counterexamples, exercises. Do NOT introduce new theory beyond what is needed for the example.

[TOPIC ANCHOR]
You must teach ONLY the current section/focus above. RECENT HISTORY is background for continuity only; if it mentions another topic, ignore it.

[AVAILABLE TOOLS]
Tool-call validation mode is enabled. For this test, request exactly one relevant tool call when the current topic contains any matrix, graph, function, curve, algorithm, runtime, complexity, or benchmark concept.
Use exactly this XML wrapper at the end of your response:
<tool_call>
{{"tool_name": "plot_matrix_heatmap", "arguments": {{"matrix": [[2,0],[0,1]], "title": "Example Matrix"}}}}
</tool_call>
Do not invent artifact paths. The runtime will create artifacts. Only request tools that fit your role.
Allowed tools by role:
- concept/example: plot_graph or plot_matrix_heatmap
- formula: plot_matrix_heatmap
- math: plot_function or plot_matrix_heatmap
- algorithm: run_algorithm_benchmark
For degree matrix topics, prefer plot_matrix_heatmap with a small 3x3 diagonal degree matrix. For graph intuition, prefer plot_graph with 3-5 nodes. For algorithm topics, prefer run_algorithm_benchmark with algorithm='degree_count'.

[LESSON GOAL]
{blackboard.lesson_goal or task.instruction}

[MACRO CONTEXT]
{blackboard.macro_context or 'EMPTY'}

[MICRO CONTEXT]
{blackboard.micro_context or 'EMPTY'}

[GRAPH CONTEXT]
{blackboard.graph_context or 'EMPTY'}

[SEMANTIC MEMORY]
{blackboard.semantic_memory or 'EMPTY'}

[RECENT HISTORY]
{blackboard.recent_history or 'EMPTY'}

[AGENT TASK]
{task.instruction}
"""
        return system_prompt, human_prompt

    def run(self, task: AgentTask, blackboard: BlackboardState):
        try:
            system_prompt, human_prompt = self._build_prompts(task, blackboard)
            content = self._llm_invoke_text(system_prompt, human_prompt)
            return self._build_result(task, content, confidence=0.75)
        except Exception as exc:
            return self._build_result(
                task,
                f"Agent {self.name} failed: {exc}",
                success=False,
                error=str(exc),
            )

    def stream_run(self, task: AgentTask, blackboard: BlackboardState):
        try:
            system_prompt, human_prompt = self._build_prompts(task, blackboard)
            chunks = []
            for chunk in self._llm_stream_text(system_prompt, human_prompt):
                chunks.append(chunk)
                yield chunk
            return "".join(chunks)
        except Exception as exc:
            error_text = f"Agent {self.name} failed: {exc}"
            yield error_text
            return error_text


class ConceptAgent(SpecialistAgent):
    name = "concept"
    role = "Concept Expert"
    goal = "Explain core intuition, metaphors, and common misconceptions."
    teaching_style = (
        "Prioritize intuition before formalism. Use analogies and identify likely misconceptions. "
        "If GRAPH CONTEXT contains unlearned prerequisites, add a short 'Prerequisite Refresh' section with at most 3 bullets. "
        "Use those prerequisites only as bridges into the current topic; do not teach them deeply."
    )


class MathAgent(SpecialistAgent):
    name = "math"
    role = "Math Expert"
    goal = "Provide rigorous derivations, proofs, and formal reasoning."
    teaching_style = "Use step-by-step logic, define assumptions, and make mathematical dependencies explicit."


class FormulaAgent(SpecialistAgent):
    name = "formula"
    role = "Formula and Notation Expert"
    goal = "Define variables, notation, formulas, and symbolic conventions."
    teaching_style = "Focus on precise notation and explain every symbol before using it."


class AlgorithmAgent(SpecialistAgent):
    name = "algorithm"
    role = "Algorithm Expert"
    goal = "Explain procedures, pseudocode, complexity, and edge cases."
    teaching_style = "Use algorithmic steps, pseudocode when useful, and discuss complexity and failure modes."


class ExampleAgent(SpecialistAgent):
    name = "example"
    role = "Example Expert"
    goal = "Create worked examples, counterexamples, and concrete exercises."
    teaching_style = "Use concrete examples and walk through them step by step."
