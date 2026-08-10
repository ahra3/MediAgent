"""Guidelines RAG Agent — retrieves relevant clinical guidelines for the patient.

This is the third node in the MediAgent graph. It:
1. Builds targeted search queries from patient conditions and symptoms
2. Runs hybrid retrieval (BM25 + vector) for each query
3. Reranks results with a cross-encoder for precision
4. Deduplicates across queries
5. Returns structured GuidelineMatch objects

No LLM call here — the retrieval pipeline IS the intelligence.
The LLM is only used in the Synthesis Agent to interpret these results.
"""

from langchain_core.messages import AIMessage

from src.graph.state import MediAgentState
from src.models.guidelines import GuidelineMatch, RAGResult
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import rerank
from src.config.settings import settings


# Lazy-loaded retriever singleton
_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    """Get or create the hybrid retriever (lazy singleton)."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _build_queries(state: MediAgentState) -> list[str]:
    """Build targeted search queries from the patient profile.

    Strategy: generate one query per condition, plus queries that
    combine conditions with relevant context (medications, labs).
    This ensures we retrieve guidelines for each condition AND
    cross-cutting topics (e.g., "diabetes management in CKD").
    """
    profile = state.get("patient_profile")
    if not profile:
        return []

    queries = []

    # One query per condition
    for condition in profile.conditions:
        queries.append(f"{condition} clinical guidelines and management")

    # Cross-cutting queries based on condition combinations
    conditions_lower = [c.lower() for c in profile.conditions]

    if any("diabet" in c for c in conditions_lower):
        if any("kidney" in c or "ckd" in c or "renal" in c for c in conditions_lower):
            queries.append("diabetes management in chronic kidney disease eGFR")
        if any("hypertens" in c for c in conditions_lower):
            queries.append("blood pressure target for diabetic patients")

    # Medication-specific queries
    med_names = [med.name.lower() for med in profile.medications]
    if "metformin" in med_names or any("glucophage" in m for m in med_names):
        queries.append("metformin dose adjustment renal function eGFR")

    # Lab-driven queries for abnormal results
    for lab in profile.lab_results:
        if lab.is_abnormal:
            if "egfr" in lab.test_name.lower():
                queries.append("CKD staging eGFR management medication adjustment")
            elif "inr" in lab.test_name.lower():
                queries.append("warfarin INR management anticoagulation")
            elif "hba1c" in lab.test_name.lower() or "a1c" in lab.test_name.lower():
                queries.append("glycemic targets HbA1c management")

    # Deduplicate while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_queries.append(q)

    return unique_queries


def guidelines_rag_agent(state: MediAgentState) -> dict:
    """Retrieve relevant clinical guidelines for the patient.

    This is a LangGraph node function. It builds queries from the
    patient profile, runs hybrid retrieval + reranking, and returns
    structured GuidelineMatch objects.

    Args:
        state: Current graph state with patient_profile populated.

    Returns:
        Partial state update with guideline_matches and messages.
    """
    profile = state.get("patient_profile")

    if profile is None:
        return {
            "guideline_matches": [],
            "messages": [
                AIMessage(content="[RAG Agent]  Skipped — no patient profile available.")
            ],
            "agent_errors": ["RAG Agent: skipped due to missing patient_profile."],
        }

    queries = _build_queries(state)

    if not queries:
        return {
            "guideline_matches": [],
            "messages": [
                AIMessage(content="[RAG Agent] No conditions found to search guidelines for.")
            ],
        }

    try:
        retriever = _get_retriever()
    except Exception as e:
        return {
            "guideline_matches": [],
            "messages": [
                AIMessage(content=f"[RAG Agent]  Failed to initialize retriever: {e}")
            ],
            "agent_errors": [f"RAG Agent: retriever init failed ({e})"],
        }

    # Retrieve and rerank for each query
    all_matches: dict[str, dict] = {}  # chunk_id → best result (dedup)

    for query in queries:
        try:
            candidates = retriever.search(query, top_k=settings.retrieval_top_k)
            reranked = rerank(query, candidates, top_k=settings.rerank_top_k)

            for result in reranked:
                cid = result["chunk_id"]
                # Keep the highest-scoring version if chunk appears across queries
                if cid not in all_matches or result["rerank_score"] > all_matches[cid]["rerank_score"]:
                    result["matched_query"] = query
                    all_matches[cid] = result
        except Exception as e:
            # Log but don't fail — other queries may still work
            pass

    # Convert to GuidelineMatch objects
    guideline_matches = []
    for cid, result in sorted(all_matches.items(), key=lambda x: x[1]["rerank_score"], reverse=True):
        match = GuidelineMatch(
            chunk_id=cid,
            source_document=result["metadata"].get("source_document", "Unknown"),
            section=result["metadata"].get("section", "Unknown"),
            text=result["text"],
            relevance_score=result["rerank_score"],
            condition=result["metadata"].get("condition", "General"),
            matched_query=result.get("matched_query", ""),
        )
        guideline_matches.append(match)

    # Build status message
    msg_parts = [
        f"[RAG Agent] Searched {len(queries)} queries across guideline knowledge base.",
        f"Retrieved {len(guideline_matches)} unique relevant chunks.",
    ]
    if guideline_matches:
        msg_parts.append("Top matches:")
        for m in guideline_matches[:5]:
            msg_parts.append(f"   [{m.condition}] {m.source_document} → {m.section} (score: {m.relevance_score:.3f})")

    return {
        "guideline_matches": guideline_matches,
        "messages": [AIMessage(content="\n".join(msg_parts))],
    }


# --- CLI test ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from src.models.patient import Medication, LabResult, PatientProfile

    print("=== Guidelines RAG Agent Test ===\n")

    # Use TC-001: elderly diabetic with CKD and hypertension
    mock_profile = PatientProfile(
        name="Test Patient",
        age=65,
        sex="Male",
        conditions=[
            "Type 2 Diabetes Mellitus",
            "Atrial Fibrillation",
            "Hypertension",
            "Chronic Kidney Disease",
        ],
        symptoms=["fatigue", "dizziness"],
        medications=[
            Medication(name="Metformin", dosage="1000mg", frequency="twice daily"),
            Medication(name="Glipizide", dosage="5mg", frequency="once daily"),
            Medication(name="Warfarin", dosage="5mg", frequency="daily"),
        ],
        allergies=[],
        lab_results=[
            LabResult(test_name="HbA1c", value="7.8%", is_abnormal=True),
            LabResult(test_name="Creatinine", value="1.8 mg/dL", is_abnormal=True),
            LabResult(test_name="eGFR", value="38 mL/min", is_abnormal=True),
            LabResult(test_name="INR", value="3.2", is_abnormal=True),
            LabResult(test_name="Blood pressure", value="155/95 mmHg", is_abnormal=True),
        ],
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

    # Show queries that would be generated
    queries = _build_queries(mock_state)
    print(f"Generated {len(queries)} search queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    print()

    # Run the full agent
    result = guidelines_rag_agent(mock_state)
    print(result["messages"][0].content)

    print(f"\nTotal guideline matches: {len(result['guideline_matches'])}")
    for m in result["guideline_matches"][:5]:
        print(f"\n   {m.source_document} → {m.section}")
        print(f"     Condition: {m.condition} | Score: {m.relevance_score:.3f}")
        print(f"     Text: {m.text[:150]}...")
