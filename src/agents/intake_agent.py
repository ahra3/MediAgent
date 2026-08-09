"""Intake Agent — extracts structured patient data from raw clinical text.

This is the first node in the MediAgent graph. It receives the raw
patient case text and extracts a structured PatientProfile using
Gemini's native structured output.

The agent includes a retry mechanism: if Pydantic validation fails,
it re-prompts once before logging the error and moving on.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config.prompt_loader import load_prompt
from src.config.settings import settings
from src.graph.state import MediAgentState
from src.models.patient import PatientProfile


def _build_llm() -> ChatGoogleGenerativeAI:
    """Build the LLM instance configured for structured extraction."""
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature_deterministic,
        google_api_key=settings.google_api_key,
    )


def intake_agent(state: MediAgentState) -> dict:
    """Extract structured patient profile from raw case text.

    This is a LangGraph node function. It receives the full graph state,
    processes the raw_case field, and returns a partial state update
    with the extracted PatientProfile.

    Args:
        state: Current graph state containing raw_case text.

    Returns:
        Partial state update with patient_profile and messages.
    """
    raw_case = state["raw_case"]
    system_prompt = load_prompt("intake_agent")
    llm = _build_llm()

    # Use Gemini's native structured output — guarantees valid Pydantic
    structured_llm = llm.with_structured_output(PatientProfile, method="json_schema")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract the patient profile from this clinical text:\n\n{raw_case}"),
    ]

    # Attempt 1
    try:
        profile = structured_llm.invoke(messages)
        return {
            "patient_profile": profile,
            "messages": [
                {"role": "assistant", "content": f"[Intake Agent] Successfully extracted profile for patient: {profile.name or 'unnamed'}. Found {len(profile.medications)} medications, {len(profile.conditions)} conditions."}
            ],
        }
    except Exception as e:
        first_error = str(e)

    # Attempt 2: retry with explicit error feedback
    try:
        retry_messages = messages + [
            HumanMessage(
                content=(
                    f"Your previous response could not be parsed. Error: {first_error}\n"
                    "Please try again, ensuring your output is a valid JSON object "
                    "matching the PatientProfile schema exactly."
                )
            ),
        ]
        profile = structured_llm.invoke(retry_messages)
        return {
            "patient_profile": profile,
            "messages": [
                {"role": "assistant", "content": f"[Intake Agent] Extracted profile (retry succeeded). Found {len(profile.medications)} medications, {len(profile.conditions)} conditions."}
            ],
            "agent_errors": [f"Intake Agent: first attempt failed ({first_error}), retry succeeded."],
        }
    except Exception as e:
        # Both attempts failed — log error but don't crash the pipeline
        error_msg = f"Intake Agent failed after 2 attempts. Error 1: {first_error} | Error 2: {e}"
        return {
            "patient_profile": None,
            "messages": [
                {"role": "assistant", "content": f"[Intake Agent] ❌ Failed to extract patient profile: {error_msg}"}
            ],
            "agent_errors": [error_msg],
        }


    
    
# --- CLI test ---
if __name__ == "__main__":
    import json 
    from pathlib import Path
    from src.config.settings import Settings
    from dotenv import load_dotenv, find_dotenv
    
    load_dotenv(find_dotenv())
    data_path = Path(__file__).parent.parent.parent / "data" / "test_cases" / "golden_dataset.json"
    with open(data_path, "r", encoding="utf-8") as f:
      cases = json.load(f)

    # Re-load settings after .env is loaded
    
    test_settings = Settings()

    print("=== Intake Agent Test ===\n")

    # Use test case TC-001
    test_case =cases[0]["raw_text"]

    # Simulate the graph state
    mock_state: MediAgentState = {
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

    result = intake_agent(mock_state)

    if result["patient_profile"]:
        profile = result["patient_profile"]
        print(f"Patient: {profile.name}, {profile.age}yo {profile.sex}")
        print(f"Conditions: {profile.conditions}")
        print(f"Symptoms: {profile.symptoms}")
        print(f"Medications:")
        for med in profile.medications:
            print(f"  - {med.name} {med.dosage or ''} {med.frequency or ''}")
        print(f"Allergies: {profile.allergies}")
        print(f"Lab Results:")
        for lab in profile.lab_results:
            abnormal = " ⚠️" if lab.is_abnormal else ""
            print(f"  - {lab.test_name}: {lab.value}{abnormal}")
    else:
        print("❌ Extraction failed")
        print(f"Errors: {result.get('agent_errors')}")
