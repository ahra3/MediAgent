"""Synthesis Agent — generates the final clinical report with citations.

This is the final agent in the MediAgent pipeline. It:
1. Receives all upstream analysis (patient profile, DDIs, guidelines)
2. Synthesizes a comprehensive ClinicalReport
3. Enforces citation grounding — every recommendation must cite a chunk_id
4. Declines to recommend if no evidence is available

This is the ONLY agent where we use temperature > 0 (0.3) because
we want natural clinical language, not robotic JSON extraction.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.config.prompt_loader import load_prompt
from src.config.settings import settings
from src.graph.state import MediAgentState
from src.models.report import ClinicalReport


def _build_evidence_context(state: MediAgentState) -> str:
    """Build a structured evidence package from all upstream outputs.

    This formats the patient profile, DDI results, and guideline
    matches into a clear text block the LLM can reason over.
    """
    parts = []

    # Patient Profile
    profile = state.get("patient_profile")
    if profile:
        parts.append("=== PATIENT PROFILE ===")
        parts.append(f"Age: {profile.age} | Sex: {profile.sex}")
        parts.append(f"Conditions: {', '.join(profile.conditions)}")
        parts.append(f"Symptoms: {', '.join(profile.symptoms)}")
        parts.append(f"Allergies: {', '.join(profile.allergies) if profile.allergies else 'None reported'}")
        parts.append("Medications:")
        for med in profile.medications:
            parts.append(f"  - {med.name} {med.dosage or ''} {med.frequency or ''}")
        if profile.lab_results:
            parts.append("Lab Results:")
            for lab in profile.lab_results:
                flag = " [ABNORMAL]" if lab.is_abnormal else ""
                parts.append(f"  - {lab.test_name}: {lab.value}{flag}")

    # Drug Interactions
    interactions = state.get("drug_interactions", [])
    parts.append(f"\n=== DRUG INTERACTIONS ({len(interactions)} found) ===")
    if interactions:
        for i in interactions:
            parts.append(f"   {i.drug_a} ↔ {i.drug_b} | Severity: {i.severity.value}")
            if i.mechanism:
                parts.append(f"     Mechanism: {i.mechanism}")
            if i.management:
                parts.append(f"     Management: {i.management}")
            parts.append(f"     Source: {i.source}")
    else:
        parts.append("  No drug-drug interactions detected.")

    parts.append(f"\nOverall DDI Risk Level: {state.get('risk_level', 'unknown')}")

    # Retrieved Guidelines
    guidelines = state.get("guideline_matches", [])
    parts.append(f"\n=== RETRIEVED CLINICAL GUIDELINES ({len(guidelines)} chunks) ===")
    if guidelines:
        for g in guidelines:
            parts.append(f"\n[chunk_id: {g.chunk_id}]")
            parts.append(f"Source: {g.source_document} → {g.section}")
            parts.append(f"Condition: {g.condition}")
            parts.append(f"Relevance Score: {g.relevance_score:.3f}")
            parts.append(f"Text:\n{g.text}")
            parts.append("---")
    else:
        parts.append("  No guideline chunks retrieved.")

    return "\n".join(parts)


def synthesis_agent(state: MediAgentState) -> dict:
    """Generate the final clinical report from all upstream evidence.

    This is a LangGraph node function. It uses the Synthesis Agent's
    prompt with temperature=0.3 for natural clinical language.

    Args:
        state: Full graph state with all upstream outputs populated.

    Returns:
        Partial state update with clinical_report and messages.
    """
    system_prompt = load_prompt("synthesis_agent")
    evidence_context = _build_evidence_context(state)

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature_creative,  # 0.3 for natural language
        google_api_key=settings.google_api_key,
    )

    structured_llm = llm.with_structured_output(ClinicalReport, method="json_schema")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate a clinical decision support report based on the following evidence:\n\n{evidence_context}"),
    ]

    # Attempt 1
    try:
        report = structured_llm.invoke(messages)
        return {
            "clinical_report": report,
            "messages": [
                AIMessage(
                    content=(
                        f"[Synthesis Agent]  Report generated. "
                        f"Risk: {report.risk_level} | "
                        f"{len(report.interaction_warnings)} warnings | "
                        f"{len(report.recommendations)} recommendations | "
                        f"{len(report.flags_for_clinician)} flags."
                    )
                )
            ],
        }
    except Exception as e:
        first_error = str(e)

    # Attempt 2: retry with error feedback
    try:
        retry_messages = messages + [
            HumanMessage(
                content=(
                    f"Your previous response could not be parsed. Error: {first_error}\n"
                    "Please respond with a valid JSON object matching the ClinicalReport schema."
                )
            ),
        ]
        report = structured_llm.invoke(retry_messages)
        return {
            "clinical_report": report,
            "messages": [
                AIMessage(content=f"[Synthesis Agent] Report generated (retry succeeded). Risk: {report.risk_level}.")
            ],
            "agent_errors": [f"Synthesis Agent: first attempt failed ({first_error}), retry succeeded."],
        }
    except Exception as e:
        error_msg = f"Synthesis Agent failed after 2 attempts. Error 1: {first_error} | Error 2: {e}"
        return {
            "clinical_report": None,
            "messages": [
                AIMessage(content=f"[Synthesis Agent]  Failed to generate report: {error_msg}")
            ],
            "agent_errors": [error_msg],
        }


# --- CLI test ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from src.models.patient import Medication, LabResult, PatientProfile
    from src.models.interactions import DrugInteraction, Severity
    from src.models.guidelines import GuidelineMatch

    print("=== Synthesis Agent Test ===\n")

    # Build a realistic mock state from TC-001
    mock_state: MediAgentState = {
        "messages": [],
        "raw_case": "",
        "patient_profile": PatientProfile(
            name="John Doe",
            age=65,
            sex="Male",
            conditions=["Type 2 Diabetes Mellitus", "Atrial Fibrillation", "Hypertension", "Chronic Kidney Disease"],
            symptoms=["fatigue", "dizziness"],
            medications=[
                Medication(name="Metformin", dosage="1000mg", frequency="twice daily"),
                Medication(name="Glipizide", dosage="5mg", frequency="once daily"),
                Medication(name="Warfarin", dosage="5mg", frequency="daily"),
            ],
            allergies=[],
            lab_results=[
                LabResult(test_name="HbA1c", value="7.8%", is_abnormal=True),
                LabResult(test_name="eGFR", value="38 mL/min", is_abnormal=True),
                LabResult(test_name="INR", value="3.2", is_abnormal=True),
                LabResult(test_name="Creatinine", value="1.8 mg/dL", is_abnormal=True),
            ],
            medical_history=[],
        ),
        "drug_interactions": [
            DrugInteraction(
                drug_a="glipizide", drug_b="warfarin",
                severity=Severity.MAJOR,
                mechanism="Warfarin may enhance hypoglycemic effect of sulfonylureas",
                management="Monitor blood glucose closely. Consider dose reduction.",
                source="local_fallback",
            ),
            DrugInteraction(
                drug_a="metformin", drug_b="glipizide",
                severity=Severity.MODERATE,
                mechanism="Additive hypoglycemic effect",
                management="Monitor blood glucose regularly",
                source="DDInter 2.0",
            ),
        ],
        "risk_level": "high",
        "medications_checked": ["Metformin", "Glipizide", "Warfarin"],
        "normalization_failures": [],
        "guideline_matches": [
            GuidelineMatch(
                chunk_id="chunk_abc123",
                source_document="ADA Standards of Care in Diabetes — 2025",
                section="Section 3: Diabetes and Chronic Kidney Disease",
                text="Metformin should be discontinued or dose reduced when eGFR falls below 30 mL/min. Dose reduction should be considered at eGFR <45 mL/min.",
                relevance_score=0.92,
                condition="Type 2 Diabetes",
                matched_query="metformin dose adjustment renal function eGFR",
            ),
            GuidelineMatch(
                chunk_id="chunk_def456",
                source_document="JNC 8 and ACC/AHA Hypertension Guidelines",
                section="Section 1: Blood Pressure Targets",
                text="For patients with diabetes: blood pressure target is <130/80 mmHg per ACC/AHA 2024 update.",
                relevance_score=0.87,
                condition="Hypertension",
                matched_query="blood pressure target for diabetic patients",
            ),
            GuidelineMatch(
                chunk_id="chunk_ghi789",
                source_document="KDIGO Clinical Practice Guidelines for CKD",
                section="Section 3: Medication Adjustments in CKD",
                text="Metformin: use with caution in CKD. Dose reduce at eGFR 30-45 mL/min. Discontinue at eGFR <30 mL/min. NSAIDs: avoid in CKD stages G3-G5.",
                relevance_score=0.85,
                condition="Chronic Kidney Disease",
                matched_query="CKD staging eGFR management medication adjustment",
            ),
        ],
        "clinical_report": None,
        "agent_errors": [],
    }

    result = synthesis_agent(mock_state)

    if result["clinical_report"]:
        report = result["clinical_report"]
        print(f"Risk Level: {report.risk_level}")
        print(f"\nPatient Summary:\n  {report.patient_summary}")
        print(f"\nInteraction Warnings ({len(report.interaction_warnings)}):")
        for w in report.interaction_warnings:
            print(f"   {w}")
        print(f"\nRecommendations ({len(report.recommendations)}):")
        for r in report.recommendations:
            print(f"   [{r.confidence}] {r.recommendation}")
            print(f"     Citations: {r.citation_chunk_ids}")
        print(f"\nFlags for Clinician ({len(report.flags_for_clinician)}):")
        for f in report.flags_for_clinician:
            print(f"   {f}")
        print(f"\nDisclaimer: {report.disclaimer[:80]}...")
    else:
        print(" Report generation failed")
        print(f"Errors: {result.get('agent_errors')}")
