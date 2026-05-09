"""
RAGAS Evaluation — AI Professor QA Pipeline
============================================
Evaluates the RAG pipeline using RAGAS (Retrieval Augmented Generation Assessment)
framework with 4 standard metrics:

  - Faithfulness:      Is the answer grounded in the retrieved context? (no hallucination)
  - Answer Relevancy:  Does the answer address the question asked?
  - Context Precision:  Are the retrieved contexts actually relevant?
  - Context Recall:     Does the retrieved context cover the ground truth?

Dataset:
  Load from tests/ragas_dataset.json — a list of {question, ground_truth, target_file}.
  You can replace this file with your own data at any time.

Prerequisites:
  pip install ragas

Usage:
  pytest tests/test_ragas_eval.py -s --tb=short
"""

import pytest
import json
import uuid
import time
import sys
import os
from pathlib import Path
from typing import Dict, List

current_dir = Path(__file__).parent.parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from core.container import Container
from core.data_ingestion import run_ingestion_pipeline
from config.settings import settings


# ==========================================
# CONSTANTS
# ==========================================
TEXTBOOK_FILE = "datalab-output-a.pdf.md"
TEXTBOOK_PATH = os.path.join(current_dir, "data", TEXTBOOK_FILE)
DATASET_PATH = os.path.join(Path(__file__).parent, "ragas_dataset.json")


# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture(scope="session")
def engine():
    """Initialize engine + ingest textbook (once per session)."""
    container = Container()
    container.config.from_pydantic(settings)
    _engine = container.runtime_engine()

    existing = _engine.vector_db.get_curriculum_groups(TEXTBOOK_FILE, "Message Passing Framework")
    if existing:
        print(f"\n[SETUP] Data already ingested, skipping.")
    else:
        print(f"\n[SETUP] Ingesting '{TEXTBOOK_FILE}'...")
        with open(TEXTBOOK_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        run_ingestion_pipeline(content, TEXTBOOK_FILE, _engine.vector_db,
                               _engine.orchestrator.llm_service, _engine.graph_db)
        print("[SETUP] Ingestion complete!")
    return _engine


@pytest.fixture(scope="session")
def dataset():
    """Load the evaluation dataset from JSON."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


# ==========================================
# PIPELINE RUNNER — captures retrieved contexts
# ==========================================
def run_qa_and_capture(engine, question: str, target_file: str, user_id: str) -> Dict:
    """
    Run the full QA pipeline and capture:
      - answer: the LLM response
      - contexts: list of retrieved text chunks used as context
    """
    # 1. Hybrid Memory
    semantic_mem = engine.vector_db.search_semantic_memory(user_id, question, limit=5)
    recent_mem = engine.graph_db.get_recent_history(user_id, limit=5)

    # 2. HyDE Search + Reranking
    anchor_nodes = []
    macro_context = ""
    search_results = engine.vector_db.search_candidates_and_fetch_parent(
        query=question,
        llm_service=engine.orchestrator.llm_service,
        target_file=target_file
    )
    if search_results:
        macro_context = search_results[0].get("page_content", "")
        if "metadata" in search_results[0]:
            raw = search_results[0]["metadata"].get("anchor_nodes", "")
            if raw:
                anchor_nodes = [n.strip() for n in raw.split(",") if n.strip()]

    # 3. Graph Context
    graph_data = []
    if anchor_nodes:
        graph_data = engine.graph_db.get_graph_context(anchor_nodes, search_mode="semi_search")

    # 4. LLM Route + Answer
    route_res = engine.support_agent.route_and_answer(
        query=question,
        semantic_memory=semantic_mem,
        recent_history=recent_mem,
        graph_context=graph_data,
        macro_context=macro_context
    )

    # 5. Self-Routing
    if route_res.get("action") == "fetch_raw":
        raw_details = engine.graph_db.get_raw_chat_turns(route_res.get("turn_ids", []))
        route_res = engine.support_agent.route_and_answer(
            query=question,
            semantic_memory=semantic_mem,
            recent_history=recent_mem,
            graph_context=graph_data,
            macro_context=macro_context,
            raw_details=raw_details
        )

    answer = route_res.get("response", "No answer generated.")

    # Collect all context pieces that were fed to the LLM
    contexts = []
    if macro_context:
        contexts.append(macro_context)
    if graph_data:
        for g in graph_data:
            ctx_str = f"{g.get('node', '')}: {json.dumps(g, ensure_ascii=False)}"
            contexts.append(ctx_str)
    if semantic_mem:
        for m in semantic_mem:
            contexts.append(m.get("summary", str(m)))

    # Save turn for accumulating memory
    turn_id = str(uuid.uuid4())
    summary = engine.support_agent.summarize_turn(question, answer)
    engine.vector_db.upsert_user_memory(user_id, turn_id, question, answer, summary)
    engine.graph_db.save_chat_turn(
        user_id=user_id, turn_id=turn_id, query=question,
        raw_answer=answer, summary=summary,
        concept_ids=list(anchor_nodes)[:3] if anchor_nodes else [],
        target_file=target_file, target_section=""
    )

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }


# ==========================================
# RAGAS EVALUATION
# ==========================================
def run_ragas_evaluation(samples: List[Dict], llm):
    """
    Run RAGAS evaluation on collected samples.
    Returns a dict with averaged scores.

    NOTE: We skip ResponseRelevancy because it requires embeddings wrapper
    that causes EmbeddingUsageEvent validation errors with FastEmbed.
    We evaluate 3 metrics: Faithfulness, ContextPrecision, ContextRecall.
    """
    try:
        from ragas import evaluate
        # Compatibility handling for different RAGAS versions
        try:
            from ragas import EvaluationDataset, SingleTurnSample
        except ImportError:
            from ragas.dataset import EvaluationDataset
            from ragas.models import SingleTurnSample

        try:
            from ragas.metrics.collections import Faithfulness, ContextPrecision, LLMContextRecall
        except ImportError:
            from ragas.metrics import Faithfulness, ContextPrecision, LLMContextRecall
    except ImportError as e:
        print(f"\n[ERROR] RAGAS library issue: {e}. Fallback to manual evaluation...")
        return run_manual_evaluation(samples, llm)

    ragas_samples = []
    for s in samples:
        ragas_samples.append(
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"] if s["contexts"] else ["No context retrieved."],
                reference=s["ground_truth"],
            )
        )

    eval_dataset = EvaluationDataset(samples=ragas_samples)

    # Wrap for RAGAS
    from ragas.llms import LangchainLLMWrapper
    ragas_llm = LangchainLLMWrapper(llm)

    # 3 metrics only
    metrics = [
        Faithfulness(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm),
    ]

    print(f"\n[RAGAS] Evaluating {len(samples)} samples with 3 metrics...")
    result = evaluate(dataset=eval_dataset, metrics=metrics)

    # Extract averaged scores from RAGAS Result object
    try:
        df = result.to_pandas()
        scores = {}
        for col in df.columns:
            if col in ["user_input", "response", "retrieved_contexts", "reference"]:
                continue
            vals = df[col].dropna().tolist()
            if vals:
                scores[col] = sum(vals) / len(vals)
        return {"scores": scores, "per_sample_df": df}
    except Exception:
        # Fallback: try dict access
        return {"scores": dict(result) if hasattr(result, '__iter__') else {}, "per_sample_df": None}


def run_manual_evaluation(samples: List[Dict], llm) -> Dict:
    """
    Fallback: LLM-as-Judge evaluation when RAGAS is not installed.
    Scores each sample on 4 dimensions matching RAGAS metrics.
    """
    from langchain_core.prompts import ChatPromptTemplate

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluation judge. Score the answer on 4 metrics (0.0 to 1.0).
Output ONLY valid JSON:
{{"faithfulness": <0-1>, "answer_relevancy": <0-1>, "context_precision": <0-1>, "context_recall": <0-1>}}

- faithfulness: Is the answer grounded in the given context? (1.0 = fully grounded)
- answer_relevancy: Does the answer address the question? (1.0 = perfectly relevant)
- context_precision: Are the retrieved contexts relevant to the question? (1.0 = all relevant)
- context_recall: Do the contexts cover the information in the ground truth? (1.0 = full coverage)"""),
        ("human", """[QUESTION]: {question}
[RETRIEVED CONTEXTS]: {contexts}
[ANSWER]: {answer}
[GROUND TRUTH]: {ground_truth}

Score as JSON:""")
    ])

    all_scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": [], "context_recall": []}

    for i, s in enumerate(samples):
        try:
            msgs = eval_prompt.format_messages(
                question=s["question"],
                contexts="\n---\n".join(s["contexts"][:3]) if s["contexts"] else "NONE",
                answer=s["answer"],
                ground_truth=s["ground_truth"]
            )
            resp = llm.invoke(msgs)
            content = resp.content
            parsed = json.loads(content[content.find('{'):content.rfind('}')+1])
            for k in all_scores:
                all_scores[k].append(float(parsed.get(k, 0)))
            print(f"  [{i+1}/{len(samples)}] ✓ F:{parsed.get('faithfulness', 0):.2f} "
                  f"AR:{parsed.get('answer_relevancy', 0):.2f} "
                  f"CP:{parsed.get('context_precision', 0):.2f} "
                  f"CR:{parsed.get('context_recall', 0):.2f}")
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] ✗ Error: {e}")
            for k in all_scores:
                all_scores[k].append(0.0)

    # Compute averages
    avg = {k: sum(v)/len(v) if v else 0 for k, v in all_scores.items()}
    return {"scores": avg, "per_sample": all_scores}


# ==========================================
# OUTPUT FORMATTER
# ==========================================
def print_results(samples: List[Dict], eval_result):
    """Print formatted evaluation results."""
    # Per-sample table
    print(f"\n{'='*100}")
    print(f"{'RAGAS EVALUATION — PER QUESTION':^100}")
    print(f"{'='*100}")
    print(f"{'#':<3} | {'Question':<50} | {'Ans Len':>7} | {'Contexts':>8}")
    print(f"{'-'*3}-+-{'-'*50}-+-{'-'*7}-+-{'-'*8}")

    for i, s in enumerate(samples):
        q_short = s["question"][:48] + ".." if len(s["question"]) > 50 else s["question"]
        print(f"{i+1:<3} | {q_short:<50} | {len(s['answer']):>7} | {len(s['contexts']):>8}")

    # Extract scores dict
    scores = {}
    if isinstance(eval_result, dict) and "scores" in eval_result:
        scores = eval_result["scores"]
    elif isinstance(eval_result, dict):
        scores = eval_result
    elif hasattr(eval_result, 'to_pandas'):
        try:
            df = eval_result.to_pandas()
            for col in df.columns:
                if col not in ["user_input", "response", "retrieved_contexts", "reference"]:
                    vals = df[col].dropna().tolist()
                    if vals:
                        scores[col] = sum(vals) / len(vals)
        except Exception:
            scores = {}

    print(f"\n{'='*60}")
    print(f"{'RAGAS SCORES — AGGREGATED':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<30} | {'Score':>10}")
    print(f"{'-'*30}-+-{'-'*10}")

    total = 0
    count = 0
    # Print ALL metrics found (not just hardcoded names)
    for key, val in scores.items():
        if isinstance(val, (int, float)):
            display = key.replace("_", " ").title()
            print(f"{display:<30} | {val:>9.3f}")
            total += val
            count += 1

    if count > 0:
        print(f"{'-'*30}-+-{'-'*10}")
        print(f"{'AVERAGE':>30} | {total/count:>9.3f}")
    else:
        print(f"{'(no scores available)':<30} |")
    print(f"{'='*60}")


# ==========================================
# MAIN TEST
# ==========================================
def test_ragas_evaluation(engine, dataset):
    """
    Full RAGAS evaluation pipeline:
      1. Ingest textbook (fixture)
      2. Run each question through the QA pipeline
      3. Capture: question, answer, retrieved contexts, ground truth
      4. Run RAGAS evaluation (or LLM-as-Judge fallback)
      5. Output formatted results + save JSON
    """
    user_id = f"ragas_{uuid.uuid4().hex[:6]}"

    print(f"\n{'='*80}")
    print(f"  RAGAS EVALUATION — {len(dataset)} questions")
    print(f"{'='*80}")

    # ── Phase 1: Run QA pipeline for each question ──
    samples = []
    for i, item in enumerate(dataset):
        question = item["question"]
        target_file = item.get("target_file", "")
        ground_truth = item["ground_truth"]

        print(f"\n  [{i+1}/{len(dataset)}] {question[:60]}...", end=" ", flush=True)

        start = time.perf_counter()
        result = run_qa_and_capture(engine, question, target_file, user_id)
        elapsed = (time.perf_counter() - start) * 1000

        result["ground_truth"] = ground_truth
        samples.append(result)
        print(f"({elapsed:.0f}ms, {len(result['contexts'])} contexts)")

    # ── Phase 2: Run RAGAS evaluation ──
    try:
        # Pass the RAW LLM to run_ragas_evaluation
        eval_result = run_ragas_evaluation(samples, engine.support_agent.llm)
    except Exception as e:
        print(f"\n[INFO] RAGAS evaluation failed ({e}), using LLM-as-Judge fallback...")
        eval_result = run_manual_evaluation(samples, engine.support_agent.llm)

    # ── Phase 3: Output ──
    print_results(samples, eval_result)

    # Save full results to JSON
    output_path = Path(__file__).parent / "ragas_results.json"
    json_out = {
        "metadata": {
            "dataset": str(DATASET_PATH),
            "textbook": TEXTBOOK_FILE,
            "num_questions": len(samples),
        },
        "scores": eval_result.get("scores", {}) if isinstance(eval_result, dict) else str(eval_result),
        "samples": [
            {
                "question": s["question"],
                "ground_truth": s["ground_truth"],
                "answer_preview": s["answer"],
                "num_contexts": len(s["contexts"]),
            }
            for s in samples
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {output_path}")

    print("\n✅ RAGAS evaluation completed.")
