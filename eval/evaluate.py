"""MediAgent Evaluation — quantitative pipeline assessment.

Runs golden test cases through the full pipeline and measures:
1. Extraction accuracy (medications, conditions)
2. DDI detection recall (did we find expected interactions?)
3. Guideline retrieval relevance (did we find expected topics?)
4. Citation coverage (are recommendations grounded in evidence?)
5. End-to-end latency (per-agent and total)

Run: uv run python -m eval.evaluate
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph.workflow import build_graph
from src.graph.state import MediAgentState


GOLDEN_DATASET_PATH = Path("data/test_cases/golden_dataset.json")
RESULTS_DIR = Path("eval/results")


def load_golden_dataset(dataset_path: Path) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_case(app, case: dict, thread_id: str) -> dict:
    """Run a single test case through the pipeline.

    Automatically approves all HITL checkpoints (evaluation mode).
    Tracks latency per agent.
    """
    initial_state: MediAgentState = {
        "messages": [],
        "raw_case": case["raw_text"],
        "patient_profile": None,
        "drug_interactions": [],
        "risk_level": None,
        "medications_checked": [],
        "normalization_failures": [],
        "guideline_matches": [],
        "clinical_report": None,
        "agent_errors": [],
    }

    config = {"configurable": {"thread_id": thread_id}}
    timings = {}
    total_start = time.time()

    # Run the graph
    for event in app.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "__interrupt__":
                continue
            timings[node_name] = time.time() - total_start

    # Auto-approve HITL interrupts
    snapshot = app.get_state(config)
    while snapshot.next:
        for event in app.stream(
            Command(resume="approve"), config=config, stream_mode="updates"
        ):
            for node_name, node_output in event.items():
                if node_name == "__interrupt__":
                    continue
                timings[node_name] = time.time() - total_start
        snapshot = app.get_state(config)

    total_time = time.time() - total_start
    timings["total"] = total_time

    # Get final state
    final_state = app.get_state(config)
    return {
        "state": final_state.values,
        "timings": timings,
    }


def score_extraction(state: dict, expected: dict) -> dict:
    """Score medication and condition extraction accuracy."""
    profile = state.get("patient_profile")
    if not profile:
        return {"med_precision": 0, "med_recall": 0, "cond_recall": 0}

    # Medication extraction
    extracted_meds = {m.name.lower() for m in profile.medications}
    expected_meds = {m.lower() for m in expected.get("medications_extracted", [])}

    if expected_meds:
        med_recall = len(extracted_meds & expected_meds) / len(expected_meds)
        med_precision = len(extracted_meds & expected_meds) / len(extracted_meds) if extracted_meds else 0
    else:
        med_recall = 1.0
        med_precision = 1.0

    # Condition extraction (fuzzy: check if expected condition substring is in any extracted condition)
    extracted_conds = [c.lower() for c in profile.conditions]
    expected_conds = [c.lower() for c in expected.get("conditions", [])]

    if expected_conds:
        matched = sum(
            1 for ec in expected_conds
            if any(ec in ext or ext in ec for ext in extracted_conds)
        )
        cond_recall = matched / len(expected_conds)
    else:
        cond_recall = 1.0

    return {
        "med_precision": round(med_precision, 3),
        "med_recall": round(med_recall, 3),
        "cond_recall": round(cond_recall, 3),
    }


def score_ddi_detection(state: dict, expected: dict) -> dict:
    """Score DDI detection recall — did we find expected interactions?"""
    found_interactions = state.get("drug_interactions", [])
    expected_ddis = expected.get("expected_ddi", [])

    if not expected_ddis:
        # No expected DDIs — check we didn't hallucinate critical ones
        return {"ddi_recall": 1.0, "ddi_found": len(found_interactions), "ddi_expected": 0}

    found_pairs = set()
    for interaction in found_interactions:
        pair = frozenset([interaction.drug_a.lower(), interaction.drug_b.lower()])
        found_pairs.add(pair)

    matched = 0
    for expected_ddi in expected_ddis:
        expected_pair = frozenset([d.lower() for d in expected_ddi["drugs"]])
        if expected_pair in found_pairs:
            matched += 1

    recall = matched / len(expected_ddis)

    return {
        "ddi_recall": round(recall, 3),
        "ddi_found": len(found_interactions),
        "ddi_expected": len(expected_ddis),
    }


def score_guidelines(state: dict, expected: dict) -> dict:
    """Score guideline retrieval — did we find expected topics?"""
    guideline_matches = state.get("guideline_matches", [])
    expected_topics = expected.get("expected_guideline_topics", [])

    if not expected_topics:
        return {"guideline_topic_recall": 1.0, "chunks_retrieved": len(guideline_matches)}

    # Check if expected topics appear in retrieved chunk text or section names
    all_text = " ".join(
        f"{g.section} {g.text} {g.condition}".lower()
        for g in guideline_matches
    )

    matched = sum(
        1 for topic in expected_topics
        if topic.lower() in all_text
    )

    return {
        "guideline_topic_recall": round(matched / len(expected_topics), 3),
        "chunks_retrieved": len(guideline_matches),
    }


def score_citations(state: dict) -> dict:
    """Score citation coverage — are recommendations grounded?"""
    report = state.get("clinical_report")
    if not report or not report.recommendations:
        return {"citation_coverage": 0, "total_recommendations": 0}

    cited = sum(1 for r in report.recommendations if r.citation_chunk_ids)
    total = len(report.recommendations)

    return {
        "citation_coverage": round(cited / total, 3) if total else 0,
        "total_recommendations": total,
        "cited_recommendations": cited,
    }


def evaluate_all(dataset_path: Path):
    """Run full evaluation across all golden/silver test cases."""
    print("=== MediAgent Evaluation ===\n")

    # Load test cases
    cases = load_golden_dataset(dataset_path)
    print(f"Loaded {len(cases)} test cases from {dataset_path.name}\n")

    # Build graph (fresh checkpointer per evaluation)
    graph = build_graph()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    all_results = []

    for i, case in enumerate(cases):
        case_id = case.get("case_id", f"TC-{i+1:03d}")
        print(f"Running {case_id}...", end=" ", flush=True)

        try:
            run_result = run_single_case(app, case, thread_id=f"eval-{case_id}")
            state = run_result["state"]
            expected = case.get("expected_output", {})

            # Score all dimensions
            scores = {
                "case_id": case_id,
                **score_extraction(state, expected),
                **score_ddi_detection(state, expected),
                **score_guidelines(state, expected),
                **score_citations(state),
                "risk_level_match": (
                    state.get("risk_level", "").lower() == expected.get("expected_risk_level", "").lower()
                ),
                "timings": run_result["timings"],
                "errors": state.get("agent_errors", []),
            }
            all_results.append(scores)
            print(f"SUCCESS ({run_result['timings']['total']:.1f}s)")

        except Exception as e:
            print(f"Error: {e}")
            all_results.append({
                "case_id": case_id,
                "error": str(e),
            })

    # Aggregate and display results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    successful = [r for r in all_results if "error" not in r]

    if successful:
        # Per-case table
        print(f"\n{'Case':<10} {'Med R':<8} {'Cond R':<8} {'DDI R':<8} {'Guide R':<8} {'Cite %':<8} {'Risk':<6} {'Time':<8}")
        print("-" * 68)
        for r in successful:
            risk_mark = "PASS" if r.get("risk_level_match") else "FAIL"
            print(
                f"{r['case_id']:<10} "
                f"{r['med_recall']:<8.1%} "
                f"{r['cond_recall']:<8.1%} "
                f"{r['ddi_recall']:<8.1%} "
                f"{r.get('guideline_topic_recall', 0):<8.1%} "
                f"{r.get('citation_coverage', 0):<8.1%} "
                f"{risk_mark:<6} "
                f"{r['timings']['total']:<8.1f}s"
            )

        # Aggregates
        print(f"\n--- Averages (n={len(successful)}) ---")
        for metric in ["med_recall", "med_precision", "cond_recall", "ddi_recall",
                       "guideline_topic_recall", "citation_coverage"]:
            values = [r.get(metric, 0) for r in successful]
            avg = sum(values) / len(values) if values else 0
            print(f"  {metric}: {avg:.1%}")

        # Latency
        total_times = [r["timings"]["total"] for r in successful]
        print(f"\n--- Latency ---")
        print(f"  Mean: {sum(total_times)/len(total_times):.1f}s")
        total_times_sorted = sorted(total_times)
        p50_idx = len(total_times_sorted) // 2
        p95_idx = int(len(total_times_sorted) * 0.95)
        print(f"  P50:  {total_times_sorted[p50_idx]:.1f}s")
        print(f"  P95:  {total_times_sorted[min(p95_idx, len(total_times_sorted)-1)]:.1f}s")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        # Convert non-serializable types
        serializable = []
        for r in all_results:
            sr = {}
            for k, v in r.items():
                if isinstance(v, bool):
                    sr[k] = v
                elif isinstance(v, (int, float, str, list, dict)):
                    sr[k] = v
                else:
                    sr[k] = str(v)
            serializable.append(sr)
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MediAgent")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="data/test_cases/golden_dataset.json",
        help="Path to the JSON dataset to evaluate"
    )
    args = parser.parse_args()
    
    evaluate_all(Path(args.dataset))
