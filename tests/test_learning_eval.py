import sys
import os
import json
import uuid

import pytest
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from core.container import Container
from config.settings import settings

def initialize_engine():
    """Initialize engine without Streamlit dependencies."""
    container = Container()
    # Force a higher temperature for evaluation to ensure variation between runs
    settings.learning_temperature = 0.7
    settings.qa_temperature = 0.7
    settings.ingestion_temperature = 0.7

    container.config.from_pydantic(settings)
    return container.runtime_engine()

# Configuration
TEXTBOOK_FILE = "datalab-output-a.pdf.md"
TARGET_SECTION = "Distributed Versus Single-Node Systems"
NUM_RUNS = 3  # Number of iterations to get a stable average score

def run_learning_evaluation(roadmap, main_entities, llm):
    """
    Evaluates a teaching roadmap using LLM-as-Judge.
    """
    judge_prompt = f"""
    You are an Expert Pedagogical Auditor. Your task is to evaluate the quality of a 'Teaching Roadmap' generated for a textbook section.
    
    [MAIN ENTITIES IN SECTION]:
    {main_entities}
    
    [GENERATED TEACHING ROADMAP]:
    {json.dumps(roadmap, indent=2)}
    
    Evaluate the full session based on these 4 criteria (Score 0-10 each):
    1. **Pedagogical Depth**: Does the explanation provide enough detail or is it too shallow?
    2. **Concept Coverage**: Does the roadmap cover all the main entities listed above?
    3. **Tone & Engagement**: Is the mentor's tone appropriate for a student?
    4. **Coherence**: Does the lecture flow smoothly from one step to the next?

    Output ONLY a JSON object:
    {{
        "scores": {{
            "pedagogical_depth": <0-10>,
            "concept_coverage": <0-10>,
            "tone_engagement": <0-10>,
            "coherence": <0-10>
        }},
        "feedback": "Concise feedback on the lecture quality"
    }}
    """
    
    try:
        response = llm.invoke(judge_prompt)
        content = response.content
        # Extract JSON
        parsed = json.loads(content[content.find('{'):content.rfind('}')+1])
        return parsed
    except Exception as e:
        return {"error": str(e)}

def test_learning_roadmap_quality():
    """
    Main test to evaluate Learning Mode roadmap quality.
    """
    engine = initialize_engine()
    
    print(f"\n\n{'='*80}")
    print(f"{'LEARNING MODE EVALUATION (CURRICULUM QUALITY)':^80}")
    print(f"{'='*80}")
    
    # 1. Retrieve the roadmap from Qdrant
    roadmap = engine.vector_db.get_curriculum_groups(TEXTBOOK_FILE, TARGET_SECTION)
    
    if not roadmap:
        pytest.fail(f"No roadmap found for {TARGET_SECTION}. Ensure data is ingested first.")

    # 2. Retrieve main entities for coverage check
    # We'll fetch the parent section metadata to get these
    parent_data = engine.vector_db.get_section_exact(TEXTBOOK_FILE, TARGET_SECTION)
    main_entities = parent_data[0]["metadata"].get("main_entities", "Unknown") if parent_data else "Unknown"

    print(f"[*] Section: {TARGET_SECTION}")
    print(f"[*] Main Entities: {main_entities}")
    print(f"[*] Roadmap Steps: {len(roadmap)}")
    for step in roadmap:
        print(f"    - Step {step.get('seq_id')}: {step.get('step_title')} [{', '.join(step.get('required_agents', []))}]")

    # 3. Multi-Run Evaluation Loop
    all_run_scores = []
    
    for run_idx in range(NUM_RUNS):
        # Create a unique student for each run to prevent Episodic Memory leakage
        run_user_id = f"eval_student_{uuid.uuid4().hex[:6]}"
        
        print(f"\n{'='*40}")
        print(f"  RUN {run_idx+1}/{NUM_RUNS} (User: {run_user_id})")
        print(f"{'='*40}")

        # Simulate Learning Session
        print(f"[*] Simulating session...")
        full_lecture_history = ""
        for step in roadmap:
            step_explanation = ""
            # The orchestrator is stateless regarding users, but we ensure state is reset
            for output in engine.orchestrator.execute_teaching_step(
                step_data=step,
                macro_context=parent_data[0]["page_content"] if parent_data else ""
            ):
                if output["type"] == "chunk":
                    step_explanation += output["content"]
            
            full_lecture_history += f"### STEP: {step.get('step_title')}\n{step_explanation}\n\n"

        # Judge this run
        print("[*] Judging this run...")
        eval_res = run_learning_evaluation(full_lecture_history, main_entities, engine.support_agent.llm)
        if "scores" in eval_res:
            all_run_scores.append(eval_res["scores"])
            print(f"    -> Partial Avg: {sum(eval_res['scores'].values())/len(eval_res['scores']):.2f}")

    # 4. Final Aggregation
    if not all_run_scores:
        pytest.fail("No successful evaluation runs.")

    # Calculate mean for each metric
    final_avg_scores = {}
    metrics = all_run_scores[0].keys()
    for m in metrics:
        final_avg_scores[m] = sum(run[m] for run in all_run_scores) / len(all_run_scores)

    # 5. Print Results
    print(f"\n{'='*60}")
    print(f"{'FINAL AGGREGATED QUALITY SCORES (Avg over ' + str(NUM_RUNS) + ' runs)':^60}")
    print(f"{'='*60}")
    
    for k, v in final_avg_scores.items():
        stars = "★" * int(round(v/2)) + "☆" * (5 - int(round(v/2))) # Adjusting for 0-10 scale
        print(f"{k.replace('_', ' ').title():<25} | {v:>5.2f}/10.0 {stars}")
    
    print(f"{'-'*60}")
    total_avg = sum(final_avg_scores.values()) / len(final_avg_scores)
    print(f"{'OVERALL PEDAGOGICAL SCORE':<25} | {total_avg:>5.2f}/10.0")
    print(f"{'='*60}")
    
    assert total_avg >= 6.0, "Average quality is below threshold."

if __name__ == "__main__":
    test_learning_roadmap_quality()
