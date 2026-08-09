"""DDI Agent — orchestrates drug name normalization and interaction checking.

This is the second node in the MediAgent graph. It:
1. Extracts medication names from the PatientProfile
2. Normalizes them via RxNorm (brand → generic, fixes misspellings)
3. Checks all pairwise combinations for drug-drug interactions
4. Classifies overall risk level (drives HITL routing downstream)

The agent combines two tools: RxNorm API (normalization) and DDI Checker
(DDInter + fallback lookup). Medical knowledge comes from databases,
never from the LLM's parametric memory.

No LLM call — the DDI Agent is pure orchestration. Medical facts come from databases (DDInter/fallback),
name resolution comes from RxNorm. The LLM is not involved at all. This is the core architectural
principle: intelligence in the system, not the model
"""

import asyncio

from langchain_core.messages import AIMessage

from src.graph.state import MediAgentState
from src.tools.ddi_checker import check_all_interactions
from src.tools.rxnorm_client import normalize_medication_list


def ddi_agent(state: MediAgentState) -> dict:
    """Check drug-drug interactions for the patient's medications.

    This is a LangGraph node function. It reads the patient_profile
    from the Intake Agent's output, normalizes drug names, and checks
    all pairwise interactions.

    Args:
        state: Current graph state with patient_profile populated.

    Returns:
        Partial state update with drug_interactions, risk_level,
        medications_checked, and normalization_failures.
    """
    profile = state.get("patient_profile")

    # Guard: if Intake Agent failed, skip gracefully
    if profile is None:
        return {
            "drug_interactions": [],
            "risk_level": "unknown",
            "medications_checked": [],
            "normalization_failures": [],
            "messages": [
                AIMessage(content="[DDI Agent] ⚠️ Skipped — no patient profile available (Intake Agent may have failed).")
            ],
            "agent_errors": ["DDI Agent: skipped due to missing patient_profile."],
        }

    # Extract medication names from the profile
    raw_med_names = [med.name for med in profile.medications]

    if not raw_med_names:
        return {
            "drug_interactions": [],
            "risk_level": "low",
            "medications_checked": [],
            "normalization_failures": [],
            "messages": [
                AIMessage(content="[DDI Agent] No medications found in patient profile. No interactions to check.")
            ],
        }

    # Step 1: Normalize drug names via RxNorm
    try:
        normalization = asyncio.run(normalize_medication_list(raw_med_names))
        normalized_names = normalization["normalized_names"]
        failures = normalization["failures"]
        mappings = normalization["mappings"]
    except Exception as e:
        # RxNorm is down — fall back to raw names (lowercase)
        normalized_names = [name.lower().strip() for name in raw_med_names]
        failures = []
        mappings = []
        state_errors = state.get("agent_errors", [])
        state_errors.append(f"DDI Agent: RxNorm normalization failed ({e}). Using raw drug names.")

    # Build a readable normalization summary for the message log
    norm_summary_lines = []
    for m in mappings:
        if m["original"].lower() != m["normalized"].lower():
            norm_summary_lines.append(f"  {m['original']} → {m['normalized']} (via {m['source']})")

    # Step 2: Check all pairwise interactions
    ddi_result = check_all_interactions(
        normalized_names=normalized_names,
        original_names=raw_med_names,
    )

    # Override normalization_failures with RxNorm results
    ddi_result.normalization_failures = failures

    # Build the status message
    msg_parts = [
        f"[DDI Agent] Checked {len(raw_med_names)} medications.",
    ]
    if norm_summary_lines:
        msg_parts.append("Drug name resolutions:")
        msg_parts.extend(norm_summary_lines)
    if failures:
        msg_parts.append(f"⚠️ Could not resolve: {', '.join(failures)}")
    msg_parts.append(f"Found {len(ddi_result.interactions)} interaction(s). Risk level: {ddi_result.risk_level}.")

    if ddi_result.interactions:
        msg_parts.append("Interactions:")
        for i in ddi_result.interactions:
            msg_parts.append(
                f"  ⚠️ {i.drug_a} ↔ {i.drug_b} | {i.severity.value} | Source: {i.source}"
            )

    return {
        "drug_interactions": ddi_result.interactions,
        "risk_level": ddi_result.risk_level,
        "medications_checked": ddi_result.medications_checked,
        "normalization_failures": ddi_result.normalization_failures,
        "messages": [AIMessage(content="\n".join(msg_parts))],
    }


# --- CLI test ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from src.models.patient import Medication, PatientProfile

    print("=== DDI Agent Test ===\n")

    # Simulate a patient profile from TC-003 (brand names + polypharmacy)
    mock_profile = PatientProfile(
        name="Test Patient",
        age=71,
        sex="Male",
        conditions=["COPD", "Type 2 Diabetes", "Depression", "Chronic back pain"],
        symptoms=["persistent cough", "shortness of breath", "leg swelling"],
        medications=[
            Medication(name="Glucophage", dosage="500mg", frequency="BID"),
            Medication(name="Advil", dosage="400mg", frequency="TID"),
            Medication(name="Zoloft", dosage="100mg", frequency="daily"),
            Medication(name="Lasix", dosage="40mg", frequency="daily"),
            Medication(name="Lisinopril", dosage="20mg", frequency="daily"),
        ],
        allergies=["Penicillin", "Sulfa drugs"],
        lab_results=[],
        medical_history=[],
    )

    mock_state: MediAgentState = {
        "messages": [],
        "raw_case": "",
        "patient_profile": mock_profile,
        "drug_interactions": [],
        "risk_level": None,
        "medications_checked": [],
        "normalization_failures": [],
        "guideline_matches": [],
        "clinical_report": None,
        "agent_errors": [],
    }

    result = ddi_agent(mock_state)

    print(f"Risk Level: {result['risk_level']}")
    print(f"Meds Checked: {result['medications_checked']}")
    print(f"Normalization Failures: {result['normalization_failures']}")
    print(f"\nInteractions ({len(result['drug_interactions'])}):")
    for i in result["drug_interactions"]:
        print(f"  ⚠️  {i.drug_a} ↔ {i.drug_b} | {i.severity.value} | {i.source}")
        if i.mechanism:
            print(f"      Mechanism: {i.mechanism}")
    print(f"\nAgent message:\n{result['messages'][0].content}")
