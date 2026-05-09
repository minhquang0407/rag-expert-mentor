"""
Ablation Study — AI Professor QA Pipeline
==========================================
Systematically disables individual components of the QA pipeline
to measure their contribution to ANSWER QUALITY using LLM-as-Judge scoring.

Metrics (scored 1-10 by the LLM evaluator):
  - Correctness:  Are the facts accurate relative to the ground truth?
  - Completeness: Does the answer cover all key points from the reference?
  - Faithfulness:  Does the answer stay grounded (no hallucination)?

Experiments:
  A) Full System (baseline)          — all components enabled
  B) No HyDE                        — direct vector search, no LLM reranking
  C) No Graph Context               — skip Neo4j graph traversal
  D) No Hybrid Memory               — no semantic/episodic memory
  E) No Self-Routing (fetch_raw)    — disable Phase 2 raw retrieval
  F) Force Global Search for Local  — use full search instead of semi_search

Usage:
  pytest tests/test_ablation_study.py -s --tb=short
"""

import pytest
import time
import json
import uuid
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field

current_dir = Path(__file__).parent.parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from core.container import Container
from core.data_ingestion import run_ingestion_pipeline
from config.settings import settings


# ==========================================
# DATA
# ==========================================
@dataclass
class AblationResult:
    experiment: str
    query: str
    answer: str
    latency_ms: float
    context_sources: Dict[str, bool]
    answer_length: int = 0
    has_macro_context: bool = False
    has_graph_data: bool = False
    has_memory: bool = False
    # LLM-as-Judge scores (1-10)
    correctness: float = 0.0
    completeness: float = 0.0
    faithfulness: float = 0.0

    def __post_init__(self):
        self.answer_length = len(self.answer)

    @property
    def avg_score(self) -> float:
        return round((self.correctness + self.completeness + self.faithfulness) / 3, 1)


# ==========================================
# TEST QUERIES WITH GROUND TRUTH
# ==========================================
TEXTBOOK_FILE = "gnn_textbook.md"
TEXTBOOK_PATH = os.path.join(current_dir, "data", TEXTBOOK_FILE)

TEST_QUERIES = [
    {
        "query": "What is the message passing mechanism in Graph Neural Networks?",
        "category": "conceptual",
        "target_file": TEXTBOOK_FILE,
        "ground_truth": (
            "Message passing is the core paradigm in GNN where each node updates its feature vector "
            "by: (1) AGGREGATE — collecting messages from neighboring nodes N(v) using functions like "
            "SUM, MEAN, or MAX; (2) UPDATE — combining the aggregated message with the node's own "
            "features. SUM aggregation is the most expressive as proven by Xu et al. (2019). "
            "A key limitation is the over-smoothing problem: as depth increases, node representations "
            "converge to indistinguishable values, limiting practical models to 2-4 layers."
        ),
    },
    {
        "query": "Explain the symmetric normalization formula in GCN and why it prevents feature explosion.",
        "category": "technical",
        "target_file": TEXTBOOK_FILE,
        "ground_truth": (
            "GCN by Kipf & Welling (2017) uses the propagation rule: "
            "H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l)), where Ã = A + I_N is the adjacency "
            "matrix with self-loops, D̃ is the corresponding degree matrix, W^(l) is a trainable "
            "weight matrix, and σ is a nonlinear activation (typically ReLU). The symmetric "
            "normalization D̃^(-1/2) Ã D̃^(-1/2) prevents feature explosion for high-degree nodes "
            "by normalizing the aggregated features proportionally to the geometric mean of the "
            "degrees, ensuring numerical stability during training."
        ),
    },
    {
        "query": "How does Graph Attention Network differ from GCN? What are the advantages?",
        "category": "comparison",
        "target_file": TEXTBOOK_FILE,
        "ground_truth": (
            "GAT (Veličković et al., 2018) differs from GCN by computing learned attention "
            "coefficients α_ij instead of fixed equal-weight aggregation. The attention is computed "
            "as: α_ij = softmax(LeakyReLU(a^T[Wh_i || Wh_j])). GAT uses multi-head attention "
            "for stability. Key advantages over GCN: (1) does not require knowing full graph "
            "structure upfront (works transductively), (2) can handle dynamic graphs, and "
            "(3) learns edge importance implicitly rather than using fixed normalization."
        ),
    },
    {
        "query": "What are the main applications of GNN in link prediction and recommendation systems?",
        "category": "applied",
        "target_file": TEXTBOOK_FILE,
        "ground_truth": (
            "Link prediction in GNN predicts missing or future edges. For recommendation systems, "
            "it predicts user-item interactions using collaborative filtering enhanced with graph "
            "structure. For knowledge graph completion, methods like TransE, DistMult, and RotatE "
            "predict missing entity relations. For drug discovery, GNN predicts drug-target "
            "interactions using molecular graphs. The common approach is: encode nodes with GNN, "
            "then score candidate edges using dot product, bilinear form, or learned decoders."
        ),
    },
]


# ==========================================
# LLM-AS-JUDGE EVALUATOR
# ==========================================
def llm_evaluate(llm, query: str, answer: str, ground_truth: str) -> Dict[str, float]:
    """
    Use the LLM to score the answer against ground truth on 3 dimensions.
    Returns dict with correctness, completeness, faithfulness (each 1-10).
    """
    from langchain_core.prompts import ChatPromptTemplate

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict academic evaluator. Score the STUDENT ANSWER against the REFERENCE ANSWER.
Output ONLY valid JSON with exactly these 3 integer scores (1-10):
{{"correctness": <1-10>, "completeness": <1-10>, "faithfulness": <1-10>}}

Scoring criteria:
- correctness: Are the facts in the student answer accurate? (10 = all facts correct, 1 = mostly wrong)
- completeness: Does the student answer cover all key points from the reference? (10 = covers everything, 1 = missing most points)
- faithfulness: Does the student answer stay grounded in facts without hallucinating? (10 = fully grounded, 1 = heavy hallucination)"""),
        ("human", """[QUESTION]: {query}

[REFERENCE ANSWER]: {ground_truth}

[STUDENT ANSWER]: {answer}

Score the student answer. Output ONLY JSON:""")
    ])

    try:
        messages = eval_prompt.format_messages(
            query=query, ground_truth=ground_truth, answer=answer
        )
        response = llm.invoke(messages)
        content = response.content
        start = content.find('{')
        end = content.rfind('}')
        parsed = json.loads(content[start:end+1])
        return {
            "correctness": float(parsed.get("correctness", 0)),
            "completeness": float(parsed.get("completeness", 0)),
            "faithfulness": float(parsed.get("faithfulness", 0)),
        }
    except Exception as e:
        print(f"\n  [WARN] Evaluation failed: {e}")
        return {"correctness": 0, "completeness": 0, "faithfulness": 0}


# ==========================================
# REPORT GENERATOR
# ==========================================
@dataclass
class AblationReport:
    results: List[AblationResult] = field(default_factory=list)

    def add(self, result: AblationResult):
        self.results.append(result)

    def final_table(self) -> str:
        """The main comparison table with quality scores."""
        from collections import defaultdict

        lines = []
        lines.append(f"\n{'='*110}")
        lines.append(f"{'ABLATION STUDY — QUALITY COMPARISON':^110}")
        lines.append(f"{'='*110}")
        lines.append(
            f"{'Experiment':<28} | {'Correct':>7} | {'Complete':>8} | {'Faithful':>8} | "
            f"{'Avg Score':>9} | {'Latency':>9} | {'Δ vs Full':>9}"
        )
        lines.append(f"{'-'*28}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")

        # Group by experiment
        exp_groups = defaultdict(list)
        for r in self.results:
            exp_groups[r.experiment].append(r)

        baseline_scores = {}
        for exp_name in ["A) Full System", "B) No HyDE", "C) No Graph Context",
                         "D) No Hybrid Memory", "E) No Self-Routing", "F) Force Global Search"]:
            if exp_name not in exp_groups:
                continue
            group = exp_groups[exp_name]
            n = len(group)
            avg_cor = sum(r.correctness for r in group) / n
            avg_com = sum(r.completeness for r in group) / n
            avg_fai = sum(r.faithfulness for r in group) / n
            avg_all = (avg_cor + avg_com + avg_fai) / 3
            avg_lat = sum(r.latency_ms for r in group) / n

            if exp_name.startswith("A)"):
                baseline_scores = {"avg": avg_all, "lat": avg_lat}
                delta_str = "BASELINE"
            else:
                delta = avg_all - baseline_scores.get("avg", 0)
                delta_str = f"{delta:+.1f}"

            lines.append(
                f"{exp_name:<28} | {avg_cor:>7.1f} | {avg_com:>8.1f} | {avg_fai:>8.1f} | "
                f"{avg_all:>9.1f} | {avg_lat:>7.0f}ms | {delta_str:>9}"
            )

        lines.append(f"{'='*110}")
        return "\n".join(lines)

    def per_query_table(self) -> str:
        """Detailed per-query breakdown."""
        lines = []
        lines.append(f"\n{'='*130}")
        lines.append(f"{'DETAILED PER-QUERY RESULTS':^130}")
        lines.append(f"{'='*130}")
        lines.append(
            f"{'Query':<45} | {'Experiment':<28} | {'Cor':>4} | {'Com':>4} | {'Fai':>4} | "
            f"{'Avg':>4} | {'ms':>7} | {'Ctx':<12}"
        )
        lines.append(f"{'-'*45}-+-{'-'*28}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*7}-+-{'-'*12}")

        for r in self.results:
            q_short = r.query[:43] + ".." if len(r.query) > 45 else r.query
            ctx_flags = []
            if r.has_macro_context: ctx_flags.append("M")
            if r.has_graph_data: ctx_flags.append("G")
            if r.has_memory: ctx_flags.append("H")
            ctx_str = "+".join(ctx_flags) if ctx_flags else "—"

            lines.append(
                f"{q_short:<45} | {r.experiment:<28} | {r.correctness:>4.0f} | {r.completeness:>4.0f} | "
                f"{r.faithfulness:>4.0f} | {r.avg_score:>4.1f} | {r.latency_ms:>5.0f}ms | {ctx_str:<12}"
            )

        lines.append(f"{'='*130}")
        return "\n".join(lines)


# ==========================================
# INGESTION FIXTURE
# ==========================================
@pytest.fixture(scope="session")
def engine():
    container = Container()
    container.config.from_pydantic(settings)
    _engine = container.runtime_engine()

    existing = _engine.vector_db.get_curriculum_groups(TEXTBOOK_FILE, "Message Passing Framework")
    if existing:
        print(f"\n[SETUP] Data already ingested for '{TEXTBOOK_FILE}', skipping.")
    else:
        print(f"\n[SETUP] Ingesting '{TEXTBOOK_FILE}'...")
        with open(TEXTBOOK_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        run_ingestion_pipeline(content, TEXTBOOK_FILE, _engine.vector_db,
                               _engine.orchestrator.llm_service, _engine.graph_db)
        print("[SETUP] Ingestion complete!")

    return _engine


# ==========================================
# QA RUNNER WITH TOGGLES
# ==========================================
def run_qa_with_config(
    engine, query: str, user_id: str, target_file: str = "",
    *, use_hyde=True, use_graph=True, use_memory=True,
    use_fetch_raw=True, force_search_mode=None,
) -> AblationResult:
    start_time = time.perf_counter()

    # 1. Hybrid Memory
    semantic_mem, recent_mem = [], []
    if use_memory:
        semantic_mem = engine.vector_db.search_semantic_memory(user_id, query, limit=5)
        recent_mem = engine.graph_db.get_recent_history(user_id, limit=5)

    # 2. HyDE Search
    anchor_nodes, macro_context = [], ""
    if use_hyde:
        results = engine.vector_db.search_candidates_and_fetch_parent(
            query=query, llm_service=engine.orchestrator.llm_service, target_file=target_file
        )
        if results:
            macro_context = results[0].get("page_content", "")
            if "metadata" in results[0]:
                raw = results[0]["metadata"].get("anchor_nodes", "")
                if raw:
                    anchor_nodes = [n.strip() for n in raw.split(",") if n.strip()]
    else:
        vec = engine.vector_db.embed_model.embed_query(query)
        try:
            resp = engine.vector_db.client.query_points(
                collection_name=engine.vector_db.parent_coll,
                query=vec, using=engine.vector_db.vector_name, limit=1
            )
            if resp.points:
                macro_context = resp.points[0].payload.get("page_content", "")
                raw = resp.points[0].payload.get("anchor_nodes", "")
                if raw:
                    anchor_nodes = [n.strip() for n in raw.split(",") if n.strip()]
        except Exception:
            pass

    # 3. Graph Context
    graph_data = []
    if use_graph and anchor_nodes:
        mode = force_search_mode or "semi_search"
        graph_data = engine.graph_db.get_graph_context(anchor_nodes, search_mode=mode)

    # 4. LLM Phase 1
    route_res = engine.support_agent.route_and_answer(
        query=query, semantic_memory=semantic_mem, recent_history=recent_mem,
        graph_context=graph_data, macro_context=macro_context
    )

    # 5. Self-Routing Phase 2
    if use_fetch_raw and route_res.get("action") == "fetch_raw":
        raw = engine.graph_db.get_raw_chat_turns(route_res.get("turn_ids", []))
        route_res = engine.support_agent.route_and_answer(
            query=query, semantic_memory=semantic_mem, recent_history=recent_mem,
            graph_context=graph_data, macro_context=macro_context, raw_details=raw
        )

    answer = route_res.get("response", "No answer generated.")
    elapsed = (time.perf_counter() - start_time) * 1000

    return AblationResult(
        experiment="", query=query, answer=answer, latency_ms=elapsed,
        context_sources={"HyDE": use_hyde, "Graph": use_graph, "Memory": use_memory, "FetchRaw": use_fetch_raw},
        has_macro_context=bool(macro_context), has_graph_data=bool(graph_data),
        has_memory=bool(semantic_mem or recent_mem),
    )


# ==========================================
# ABLATION CONFIGS
# ==========================================
ABLATION_CONFIGS = [
    {"name": "A) Full System",         "use_hyde": True,  "use_graph": True,  "use_memory": True,  "use_fetch_raw": True},
    {"name": "B) No HyDE",             "use_hyde": False, "use_graph": True,  "use_memory": True,  "use_fetch_raw": True},
    {"name": "C) No Graph Context",    "use_hyde": True,  "use_graph": False, "use_memory": True,  "use_fetch_raw": True},
    {"name": "D) No Hybrid Memory",    "use_hyde": True,  "use_graph": True,  "use_memory": False, "use_fetch_raw": True},
    {"name": "E) No Self-Routing",     "use_hyde": True,  "use_graph": True,  "use_memory": True,  "use_fetch_raw": False},
    {"name": "F) Force Global Search", "use_hyde": True,  "use_graph": True,  "use_memory": True,  "use_fetch_raw": True, "force_search_mode": "search"},
]


# ==========================================
# MAIN TEST
# ==========================================
def test_ablation_study(engine):
    """
    Full ablation pipeline:
      1. Ingest GNN textbook (fixture)
      2. Seed chat history
      3. Run 4 queries × 6 configs = 24 experiments
      4. LLM-as-Judge scores each answer against ground truth
      5. Output quality comparison + per-query tables
    """
    user_id = f"ablation_{uuid.uuid4().hex[:6]}"

    # ── Phase 0: Seed memory ──
    print("\n[PHASE 0] Seeding chat history...")
    seeds = [
        ("What is spectral graph theory?",
         "Spectral graph theory studies graphs via eigenvalues of the Laplacian L = D - A."),
        ("What is the over-smoothing problem in GNN?",
         "Over-smoothing: stacking too many GNN layers makes node representations converge."),
    ]
    for sq, sa in seeds:
        tid = str(uuid.uuid4())
        engine.vector_db.upsert_user_memory(user_id, tid, sq, sa, f"{sq[:40]}→{sa[:40]}")
        engine.graph_db.save_chat_turn(user_id=user_id, turn_id=tid, query=sq,
            raw_answer=sa, summary=f"{sq[:40]}→{sa[:40]}", concept_ids=[],
            target_file=TEXTBOOK_FILE, target_section="")

    # ── Phase 1: Run all experiments ──
    report = AblationReport()

    for q_data in TEST_QUERIES:
        query = q_data["query"]
        print(f"\n{'='*80}\n  QUERY [{q_data['category'].upper()}]: {query}\n{'='*80}")

        for cfg in ABLATION_CONFIGS:
            name = cfg["name"]
            run_cfg = {k: v for k, v in cfg.items() if k != "name"}
            print(f"  ► {name}...", end=" ", flush=True)

            result = run_qa_with_config(engine, query, user_id, q_data["target_file"], **run_cfg)
            result.experiment = name
            print(f"{result.latency_ms:.0f}ms", end="")

            # ── LLM-as-Judge ──
            scores = llm_evaluate(engine.support_agent.llm, query, result.answer, q_data["ground_truth"])
            result.correctness = scores["correctness"]
            result.completeness = scores["completeness"]
            result.faithfulness = scores["faithfulness"]
            print(f" → Cor:{result.correctness:.0f} Com:{result.completeness:.0f} Fai:{result.faithfulness:.0f}")

            report.add(result)

    # ── Phase 2: Output ──
    print(report.final_table())
    print(report.per_query_table())

    # Save JSON
    output_path = Path(__file__).parent / "ablation_results.json"
    json_out = []
    for r in report.results:
        json_out.append({
            "experiment": r.experiment, "query": r.query,
            "correctness": r.correctness, "completeness": r.completeness,
            "faithfulness": r.faithfulness, "avg_score": r.avg_score,
            "latency_ms": round(r.latency_ms, 1),
            "has_macro_context": r.has_macro_context,
            "has_graph_data": r.has_graph_data, "has_memory": r.has_memory,
            "answer_preview": r.answer[:300],
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {output_path}")

    # ── Assertions ──
    full = [r for r in report.results if r.experiment.startswith("A)")]
    for r in full:
        assert r.avg_score > 0, f"Full system scored 0 for: {r.query}"

    print("\n✅ Ablation study completed.")


# ==========================================
# MEMORY STRESS TEST
# ==========================================
# 15 diverse past conversations — the recall question targets turn #3 (eigenvalues)
# which is buried deep in history. Only semantic search can find it efficiently.
MEMORY_FLOOD_TURNS = [
    ("What is an adjacency matrix?",
     "An adjacency matrix A is a square matrix where A_ij = 1 if edge exists between i and j. For weighted graphs, A_ij stores edge weight. Space complexity is O(V^2)."),
    ("How does degree matrix work?",
     "Degree matrix D is diagonal where D_ii = sum of edges connected to vertex i. Used in Laplacian L = D - A."),
    ("What are the eigenvalues of the graph Laplacian and what do they tell us?",
     "The smallest eigenvalue is always 0. The number of zero eigenvalues equals connected components. The second-smallest eigenvalue (Fiedler value) measures graph connectivity — larger values mean more connected. The eigenvectors form the basis for the Graph Fourier Transform: x_hat = U^T x."),
    ("What is an edge list representation?",
     "Edge list is a flat list of (u,v) pairs. Simple but O(E) for neighbor lookups. Good for sparse graphs with few queries."),
    ("What is the GCN propagation rule?",
     "H^(l+1) = sigma(D_tilde^{-1/2} A_tilde D_tilde^{-1/2} H^(l) W^(l)) where A_tilde = A + I, sigma is ReLU."),
    ("Explain multi-head attention in GAT.",
     "GAT concatenates K attention heads for stability. Each head computes independent attention coefficients alpha_ij and aggregates neighbors separately."),
    ("What is node classification?",
     "Predicting labels for unlabeled nodes using labeled nodes + graph structure. Semi-supervised: small labeled fraction, full graph for propagation."),
    ("How does link prediction work?",
     "Predict missing/future edges. Encode with GNN, score edges via dot product or bilinear form. Used in recommender systems and knowledge graphs."),
    ("What datasets are common for GNN benchmarks?",
     "Cora, CiteSeer, PubMed for citation networks. OGB for large-scale benchmarks. QM9, ZINC for molecular graphs."),
    ("What is graph pooling?",
     "Pooling reduces graph to fixed-size representation. Methods: global mean/sum, DiffPool (hierarchical), SAGPool (attention-based), Set2Set."),
    ("Explain the over-smoothing problem in detail.",
     "As GNN depth increases beyond 2-4 layers, node features converge to indistinguishable values. Solutions: residual connections, JK-Net, DropEdge."),
    ("What is Graph Fourier Transform?",
     "Given signal x on vertices, GFT is x_hat = U^T x where U are Laplacian eigenvectors. Inverse is x = U x_hat. Foundation for spectral convolutions."),
    ("Compare GCN vs GAT.",
     "GCN uses fixed symmetric normalization. GAT learns attention weights per edge. GAT handles dynamic graphs, doesn't need full structure upfront."),
    ("What is DistMult for knowledge graphs?",
     "DistMult scores triples (h,r,t) as h^T diag(r) t. Simple bilinear model for KG completion. Symmetric so can't model asymmetric relations."),
    ("How is GNN used in drug discovery?",
     "Molecular graphs: atoms=nodes, bonds=edges. GNN predicts drug-target interactions, molecular properties (solubility, toxicity). Used with QM9 dataset."),
]

RECALL_QUERY = {
    "query": "Earlier I asked you about eigenvalues of the graph Laplacian. Can you recall exactly what you told me about the Fiedler value and the Graph Fourier Transform formula?",
    "target_file": TEXTBOOK_FILE,
    "ground_truth": (
        "The smallest eigenvalue is always 0. The number of zero eigenvalues equals the number of "
        "connected components. The second-smallest eigenvalue is called the Fiedler value and measures "
        "graph connectivity — larger values indicate more connected graphs. The eigenvectors form the "
        "basis for the Graph Fourier Transform: x_hat = U^T x, and the inverse is x = U x_hat."
    ),
}


def test_memory_stress(engine):
    """
    Memory Stress Test:
    Simulate a user with 15 past conversations, then ask a RECALL question
    that targets turn #3 (eigenvalues/Fiedler value), buried deep in history.

    This is the scenario where:
    - Full System: semantic search finds turn #3 by meaning → high score
    - No Memory: LLM has zero history context → must guess from textbook only
    - No Self-Routing: has summaries but can't fetch_raw for exact details
    """
    user_id = f"stress_{uuid.uuid4().hex[:6]}"

    # ── Flood memory with 15 turns ──
    print(f"\n{'='*80}")
    print(f"  MEMORY STRESS TEST — Seeding {len(MEMORY_FLOOD_TURNS)} turns")
    print(f"{'='*80}")

    for i, (sq, sa) in enumerate(MEMORY_FLOOD_TURNS):
        tid = str(uuid.uuid4())
        summary = engine.support_agent.summarize_turn(sq, sa)
        engine.vector_db.upsert_user_memory(user_id, tid, sq, sa, summary)
        engine.graph_db.save_chat_turn(
            user_id=user_id, turn_id=tid, query=sq, raw_answer=sa,
            summary=summary, concept_ids=[], target_file=TEXTBOOK_FILE, target_section=""
        )
        print(f"  [{i+1:2d}/{len(MEMORY_FLOOD_TURNS)}] Seeded: {sq[:50]}...")

    # ── Run recall query with 3 configs ──
    stress_configs = [
        {"name": "A) Full System",      "use_hyde": True, "use_graph": True, "use_memory": True,  "use_fetch_raw": True},
        {"name": "D) No Hybrid Memory", "use_hyde": True, "use_graph": True, "use_memory": False, "use_fetch_raw": True},
        {"name": "E) No Self-Routing",  "use_hyde": True, "use_graph": True, "use_memory": True,  "use_fetch_raw": False},
    ]

    query = RECALL_QUERY["query"]
    gt = RECALL_QUERY["ground_truth"]

    print(f"\n  RECALL QUERY: {query}")
    print(f"  (Target: turn #3 — eigenvalues, Fiedler value, GFT formula)\n")

    report = AblationReport()
    for cfg in stress_configs:
        name = cfg["name"]
        run_cfg = {k: v for k, v in cfg.items() if k != "name"}
        print(f"  ► {name}...", end=" ", flush=True)

        result = run_qa_with_config(engine, query, user_id, RECALL_QUERY["target_file"], **run_cfg)
        result.experiment = name

        scores = llm_evaluate(engine.support_agent.llm, query, result.answer, gt)
        result.correctness = scores["correctness"]
        result.completeness = scores["completeness"]
        result.faithfulness = scores["faithfulness"]
        print(f"{result.latency_ms:.0f}ms → Cor:{result.correctness:.0f} Com:{result.completeness:.0f} Fai:{result.faithfulness:.0f}")

        report.add(result)

    print(report.final_table())

    # The key assertion: Full System should outscore No Memory on completeness
    full = next(r for r in report.results if r.experiment.startswith("A)"))
    no_mem = next(r for r in report.results if r.experiment.startswith("D)"))
    print(f"\n  Full System avg: {full.avg_score} vs No Memory avg: {no_mem.avg_score}")
    print(f"  Memory contribution: +{full.avg_score - no_mem.avg_score:.1f} points")
    print("\n✅ Memory stress test completed.")
