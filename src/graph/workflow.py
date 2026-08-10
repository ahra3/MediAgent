"""MediAgent workflow — the full LangGraph pipeline.

This module constructs the StateGraph that orchestrates all agents:

  START → intake → ddi → [conditional: critical? → HITL DDI review] 
       → guidelines_rag → HITL pre-synthesis → synthesis → END

HITL (Human-in-the-Loop) interrupts:
1. hitl_ddi_review: fires ONLY when risk_level is "critical" or "high"
2. hitl_pre_synthesis: fires ALWAYS before final report generation

The graph uses LangGraph's interrupt() for HITL and Command for routing.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

from src.graph.state import MediAgentState
from src.agents.intake_agent import intake_agent
from src.agents.ddi_agent import ddi_agent
from src.agents.guidelines_rag_agent import guidelines_rag_agent
from src.agents.synthesis_agent import synthesis_agent


def hitl_ddi_review(state: MediAgentState) -> dict:
    """Human-in-the-loop checkpoint for critical/high DDI risk.

    Pauses the pipeline and presents DDI findings to the clinician.
    The clinician can approve, modify, or override the findings.
    """
    interactions = state.get("drug_interactions", [])
    risk_level = state.get("risk_level", "unknown")

    # Build a summary for the clinician
    review_summary = [
        f" DDI REVIEW REQUIRED — Risk Level: {risk_level.upper()}",
        f"Found {len(interactions)} drug interaction(s):",
    ]
    for i in interactions:
        review_summary.append(f"  • {i.drug_a} ↔ {i.drug_b} | {i.severity.value}")
        if i.mechanism:
            review_summary.append(f"    Mechanism: {i.mechanism}")
        if i.management:
            review_summary.append(f"    Suggested: {i.management}")

    review_summary.append("\nPlease review and type 'approve' to continue, or provide modifications.")

    # This pauses the graph and waits for human input
    human_response = interrupt(value="\n".join(review_summary))

    # After clinician responds, continue the pipeline
    return {
        "messages": [
            {"role": "user", "content": f"[Clinician DDI Review] {human_response}"},
            {"role": "assistant", "content": f"[System] DDI review acknowledged. Proceeding with guidelines retrieval."},
        ],
    }


def hitl_pre_synthesis(state: MediAgentState) -> dict:
    """Human-in-the-loop checkpoint before final synthesis.

    Always fires — gives the clinician a chance to review the full
    evidence package before the report is generated.
    """
    profile = state.get("patient_profile")
    interactions = state.get("drug_interactions", [])
    guidelines = state.get("guideline_matches", [])

    review_parts = [
        "📋 PRE-SYNTHESIS REVIEW — Full evidence package:",
        f"\nPatient: {profile.age}yo {profile.sex}" if profile else "Patient: Unknown",
        f"Conditions: {', '.join(profile.conditions)}" if profile else "",
        f"\nDDIs: {len(interactions)} interaction(s) | Risk: {state.get('risk_level', 'unknown')}",
        f"Guidelines: {len(guidelines)} relevant chunk(s) retrieved",
    ]

    if guidelines:
        review_parts.append("\nTop guideline matches:")
        for g in guidelines[:5]:
            review_parts.append(f"  • [{g.condition}] {g.source_document} → {g.section} (score: {g.relevance_score:.3f})")

    review_parts.append("\nType 'approve' to generate the final report, or provide feedback.")

    human_response = interrupt(value="\n".join(review_parts))

    return {
        "messages": [
            {"role": "user", "content": f"[Clinician Pre-Synthesis Review] {human_response}"},
            {"role": "assistant", "content": "[System] Pre-synthesis review acknowledged. Generating clinical report."},
        ],
    }


def route_after_ddi(state: MediAgentState):
    """Conditional routing after DDI Agent.

    If risk is critical or high → HITL DDI review first
    Otherwise → skip straight to guidelines retrieval
    """
    risk = state.get("risk_level", "low")
    if risk in ("critical", "high"):
        return "hitl_ddi_review"
    return "guidelines_rag_agent"


def build_graph() -> StateGraph:
    """Construct the full MediAgent workflow graph.

    Returns:
        Compiled LangGraph StateGraph ready to invoke.
    """
    graph = StateGraph(MediAgentState)

    # Add all nodes
    graph.add_node("intake_agent", intake_agent)
    graph.add_node("ddi_agent", ddi_agent)
    graph.add_node("hitl_ddi_review", hitl_ddi_review)
    graph.add_node("guidelines_rag_agent", guidelines_rag_agent)
    graph.add_node("hitl_pre_synthesis", hitl_pre_synthesis)
    graph.add_node("synthesis_agent", synthesis_agent)

    # Define edges
    graph.add_edge(START, "intake_agent")
    graph.add_edge("intake_agent", "ddi_agent")

    # Conditional: high/critical risk → HITL review, else → skip to RAG
    graph.add_conditional_edges("ddi_agent", route_after_ddi)

    # After HITL DDI review → continue to guidelines
    graph.add_edge("hitl_ddi_review", "guidelines_rag_agent")

    # After guidelines → always HITL pre-synthesis review
    graph.add_edge("guidelines_rag_agent", "hitl_pre_synthesis")

    # After pre-synthesis review → generate report
    graph.add_edge("hitl_pre_synthesis", "synthesis_agent")

    # Synthesis → END
    graph.add_edge("synthesis_agent", END)

    return graph


def compile_graph():
    """Build and compile the graph with a checkpointer for HITL.

    The MemorySaver checkpointer is required for interrupt() to work.
    In production, replace with SqliteSaver or PostgresSaver.
    """
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_graph()
    checkpointer = MemorySaver()
    compiled_graph = graph.compile(checkpointer=checkpointer)
    compiled_graph.get_graph().draw_mermaid_png(output_file_path="mediagent_workflow.png")
    return compiled_graph


# --- CLI test: full end-to-end pipeline ---
# --- CLI test: full end-to-end pipeline ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=== MediAgent Full Pipeline Test ===\n")

    app = compile_graph()

    # Use TC-001 as the test case
    test_case = (
        "65-year-old male presenting with fatigue and dizziness. "
        "Past medical history significant for Type 2 Diabetes Mellitus diagnosed 10 years ago, "
        "Atrial Fibrillation, and Hypertension. Current medications include "
        "Metformin 1000mg twice daily, Glipizide 5mg once daily in the morning, "
        "and Warfarin 5mg daily (started 3 months ago for AF). "
        "Patient reports no known drug allergies. "
        "Recent labs show HbA1c 7.8%, Creatinine 1.8 mg/dL (elevated), "
        "eGFR 38 mL/min, INR 3.2 (above therapeutic range), Potassium 4.9 mEq/L. "
        "Blood pressure today: 155/95 mmHg."
    )

    initial_state = {
        "messages": [],
        "raw_case": test_case,
        "patient_profile": None,
        "drug_interactions": [],
        "risk_level": None,
        "medications_checked": [],
        "normalization_failures": [],
        "guideline_matches": [],
        "clinical_report": None,
        "agent_errors": [],
    }

    config = {"configurable": {"thread_id": "test-run-001"}}

    print("Starting pipeline...\n")

    # Run the graph — it will pause at HITL interrupts
    for event in app.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            # Skip interrupt events (handled below via get_state)
            if node_name == "__interrupt__":
                continue

            print(f"--- {node_name} ---")
            if isinstance(node_output, dict):
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    if hasattr(msg, "content"):
                        print(msg.content)
                    elif isinstance(msg, dict):
                        print(f"  [{msg.get('role', '?')}] {msg.get('content', '')}")
            print()

    # Check if we're interrupted (HITL)
    snapshot = app.get_state(config)
    while snapshot.next:
        interrupted_node = snapshot.next[0]
        print(f"🛑 Pipeline paused at: {interrupted_node}")

        # Show the interrupt value (review summary)
        if snapshot.tasks and snapshot.tasks[0].interrupts:
            print(snapshot.tasks[0].interrupts[0].value)

        # Get clinician input
        human_input = input("\nYour response (or just press Enter for 'approve'): ").strip()
        if not human_input:
            human_input = "approve"

        # Resume the graph with the human response
        print(f"\n[Resuming with: '{human_input}']\n")
        for event in app.stream(
            Command(resume=human_input), config=config, stream_mode="updates"
        ):
            for node_name, node_output in event.items():
                if node_name == "__interrupt__":
                    continue

                print(f"--- {node_name} ---")
                if isinstance(node_output, dict):
                    msgs = node_output.get("messages", [])
                    for msg in msgs:
                        if hasattr(msg, "content"):
                            print(msg.content)
                        elif isinstance(msg, dict):
                            print(f"  [{msg.get('role', '?')}] {msg.get('content', '')}")
                print()

        # Check if there's another interrupt
        snapshot = app.get_state(config)

    # Pipeline complete — show the final report
    final_state = app.get_state(config)
    report = final_state.values.get("clinical_report")

    if report:
        print("=" * 60)
        print("FINAL CLINICAL REPORT")
        print("=" * 60)
        print(f"\nRisk Level: {report.risk_level}")
        print(f"\nPatient Summary:\n  {report.patient_summary}")
        print(f"\nInteraction Warnings ({len(report.interaction_warnings)}):")
        for w in report.interaction_warnings:
            print(f"  ⚠️ {w}")
        print(f"\nRecommendations ({len(report.recommendations)}):")
        for r in report.recommendations:
            print(f"  📋 [{r.confidence}] {r.recommendation}")
            print(f"     Citations: {r.citation_chunk_ids}")
        print(f"\nFlags for Clinician ({len(report.flags_for_clinician)}):")
        for f in report.flags_for_clinician:
            print(f"  🚩 {f}")
        print(f"\n{report.disclaimer}")
    else:
        print("❌ No report generated")
        errors = final_state.values.get("agent_errors", [])
        if errors:
            print(f"Errors: {errors}")
