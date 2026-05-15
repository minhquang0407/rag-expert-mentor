import json
import uuid
from typing import Any, Dict, List

from core.schemas import (
    AgentTask,
    BlackboardState,
    GradingResult,
    QuizPayload,
    QuizQuestion,
    RemediationPlan,
)
from agents.base import BaseAgent
from runtime.protocols import parse_json_object


class QuizAgent(BaseAgent):
    """Assessment agent that generates concept-grounded multiple-choice quizzes."""

    name = "quiz"
    role = "Assessment Quiz Generator"
    goal = "Generate diagnostic questions for the current lesson concepts."

    def run(self, task: AgentTask, blackboard: BlackboardState):
        concepts = task.metadata.get("concepts") or blackboard.artifacts.get("target_concepts") or []
        question_count = int(task.metadata.get("question_count", 5))

        system_prompt = """
You are a rigorous assessment generator. Output strictly valid JSON only.
Schema:
{
  "title": "Short quiz title",
  "questions": [
    {
      "question": "Question text",
      "options": ["A", "B", "C", "D"],
      "answer_idx": 0,
      "explanation": "Why the answer is correct",
      "concept_id": "Concept name",
      "difficulty": "easy|medium|hard"
    }
  ]
}
Rules:
- Generate exactly the requested number of questions.
- Each question must have exactly 4 options.
- answer_idx must be 0, 1, 2, or 3.
- Focus on conceptual understanding, formulas, and common misconceptions.
- Do not use raw LaTeX backslashes in JSON strings; write formulas in plain Unicode/text form when needed.
"""
        human_prompt = f"""
[LESSON GOAL]
{blackboard.lesson_goal}

[LESSON CONTEXT]
{blackboard.macro_context or blackboard.micro_context}

[TARGET CONCEPTS]
{concepts}

[QUESTION COUNT]
{question_count}
"""
        try:
            raw = self._llm_invoke_text(system_prompt, human_prompt)
            parsed = parse_json_object(raw)
            questions = [QuizQuestion(**q) for q in parsed.get("questions", [])]
            quiz = QuizPayload(
                quiz_id=str(uuid.uuid4()),
                title=parsed.get("title", blackboard.lesson_goal or "Assessment"),
                questions=questions,
                target_concepts=list(concepts),
            )
            blackboard.artifacts["quiz"] = quiz.model_dump()
            return self._build_result(
                task,
                f"Generated quiz with {len(quiz.questions)} questions.",
                confidence=0.85,
                metadata={"quiz": quiz.model_dump()},
            )
        except Exception as exc:
            return self._build_result(task, f"QuizAgent failed: {exc}", success=False, error=str(exc))


class GraderAgent(BaseAgent):
    """Assessment agent that grades quiz submissions and plans remediation."""

    name = "grader"
    role = "Assessment Grader"
    goal = "Grade quiz answers, identify weak concepts, and propose remediation."

    def run(self, task: AgentTask, blackboard: BlackboardState):
        quiz_data = task.metadata.get("quiz") or blackboard.artifacts.get("quiz") or {}
        answers: Dict[str, Any] = task.metadata.get("answers", {})
        quiz = QuizPayload(**quiz_data)

        total = len(quiz.questions)
        correct = 0
        weak_concepts: List[str] = []

        for idx, question in enumerate(quiz.questions):
            submitted = answers.get(str(idx), answers.get(idx))
            if submitted == question.answer_idx:
                correct += 1
            else:
                weak_concepts.append(question.concept_id or f"Question {idx + 1}")

        score = correct / total if total else 0.0
        weak_concepts = sorted(set(weak_concepts))
        result = GradingResult(
            score=score,
            correct_count=correct,
            total_count=total,
            weak_concepts=weak_concepts,
            remediation_required=score < float(task.metadata.get("pass_threshold", 0.75)),
            feedback=f"Score: {correct}/{total}." if total else "No questions were available to grade.",
        )

        remediation = RemediationPlan(
            weak_concepts=weak_concepts,
            recommended_agents=["concept", "example"] if weak_concepts else [],
            instruction=(
                "Review the weak concepts with intuition-first explanation and worked examples: "
                + ", ".join(weak_concepts)
                if weak_concepts else "No remediation required."
            ),
        )

        blackboard.artifacts["grading_result"] = result.model_dump()
        blackboard.artifacts["remediation_plan"] = remediation.model_dump()
        return self._build_result(
            task,
            result.feedback,
            confidence=0.95,
            metadata={
                "grading_result": result.model_dump(),
                "remediation_plan": remediation.model_dump(),
            },
        )
