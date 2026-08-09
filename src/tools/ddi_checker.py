"""Drug-Drug Interaction checker — combined DDInter + fallback lookup.

This module is the single entry point for DDI checking. It:
1. Checks DDInter 2.0 (160K+ interactions from Parquet)
2. Falls back to curated JSON for common interactions
3. Assigns overall risk level based on severity findings
4. Returns structured results for the DDI Agent

The DDI Agent calls check_all_interactions() and gets back
everything it needs to populate the graph state.
"""

import json
from itertools import combinations
from pathlib import Path

import polars as pl

from src.config.settings import settings
from src.models.interactions import DDIResult, DrugInteraction, Severity
from src.tools.ddinter_loader import load_processed, lookup_interaction


def _load_fallback_db() -> list[dict]:
    """Load the curated fallback interaction database from JSON."""
    fallback_path = settings.fallback_json
    if not fallback_path.exists():
        return []
    return json.loads(fallback_path.read_text(encoding="utf-8"))


def _lookup_fallback(drug_a: str, drug_b: str, fallback_db: list[dict]) -> dict | None:
    """Check two drugs against the fallback JSON database.

    Args:
        drug_a: First drug name (lowercase).
        drug_b: Second drug name (lowercase).
        fallback_db: The loaded fallback interaction list.

    Returns:
        Matching interaction dict, or None.
    """
    a, b = drug_a.lower().strip(), drug_b.lower().strip()

    for entry in fallback_db:
        entry_a = entry["drug_a"].lower().strip()
        entry_b = entry["drug_b"].lower().strip()
        if (entry_a == a and entry_b == b) or (entry_a == b and entry_b == a):
            return entry

    return None


def _classify_risk(interactions: list[DrugInteraction]) -> str:
    """Determine overall risk level from a list of interactions.

    Logic:
    - critical: any MAJOR interaction with life-threatening potential
    - high: multiple MODERATE or one MAJOR
    - moderate: one or more MODERATE
    - low: only MINOR or no interactions
    """
    if not interactions:
        return "low"

    severities = [i.severity for i in interactions]
    major_count = severities.count(Severity.MAJOR)
    moderate_count = severities.count(Severity.MODERATE)

    if major_count >= 1:
        return "critical" if major_count >= 2 else "high"
    if moderate_count >= 3:
        return "high"
    if moderate_count >= 1:
        return "moderate"
    return "low"


def check_all_interactions(
    normalized_names: list[str],
    original_names: list[str] | None = None,
) -> DDIResult:
    """Check all pairwise drug interactions — main entry point.

    Checks DDInter first, then fallback JSON for pairs not found in DDInter.
    This two-layer approach ensures comprehensive coverage.

    Args:
        normalized_names: List of normalized (generic) drug names.
        original_names: Optional list of original drug names (for audit trail).

    Returns:
        DDIResult with all interactions found and overall risk level.
    """
    ddinter_df = load_processed()
    fallback_db = _load_fallback_db()

    interactions: list[DrugInteraction] = []
    checked_pairs: set[tuple[str, str]] = set()

    for drug_a, drug_b in combinations(normalized_names, 2):
        # Skip if already checked (order-independent)
        pair_key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
        if pair_key in checked_pairs:
            continue
        checked_pairs.add(pair_key)

        # Layer 1: Check DDInter
        ddinter_matches = lookup_interaction(ddinter_df, drug_a, drug_b)

        if ddinter_matches:
            for match in ddinter_matches:
                severity_raw = match.get("severity", "unknown")
                try:
                    severity = Severity(severity_raw)
                except ValueError:
                    severity = Severity.UNKNOWN

                interactions.append(DrugInteraction(
                    drug_a=drug_a,
                    drug_b=drug_b,
                    severity=severity,
                    mechanism=match.get("mechanism"),
                    clinical_effect=match.get("clinical_effect"),
                    management=match.get("management"),
                    source="DDInter 2.0",
                ))
            continue  # Found in DDInter, skip fallback

        # Layer 2: Check fallback JSON
        fallback_match = _lookup_fallback(drug_a, drug_b, fallback_db)
        if fallback_match:
            try:
                severity = Severity(fallback_match["severity"])
            except ValueError:
                severity = Severity.UNKNOWN

            interactions.append(DrugInteraction(
                drug_a=fallback_match.get("drug_a", drug_a),
                drug_b=fallback_match.get("drug_b", drug_b),
                severity=severity,
                mechanism=fallback_match.get("mechanism"),
                clinical_effect=fallback_match.get("clinical_effect"),
                management=fallback_match.get("management"),
                source="local_fallback",
            ))

    risk_level = _classify_risk(interactions)

    return DDIResult(
        interactions=interactions,
        risk_level=risk_level,
        medications_checked=original_names or normalized_names,
        normalization_failures=[],  # Filled by the DDI Agent from RxNorm results
    )


# --- CLI test ---
if __name__ == "__main__":
    print("=== DDI Checker Test ===\n")

    # Test case from TC-003: polypharmacy with brand names already normalized
    test_meds = ["metformin", "ibuprofen", "sertraline", "furosemide", "lisinopril"]

    print(f"Checking: {test_meds}\n")
    result = check_all_interactions(test_meds)

    print(f"Risk level: {result.risk_level}")
    print(f"Interactions found: {len(result.interactions)}\n")

    for i in result.interactions:
        print(f"  ⚠️  {i.drug_a} ↔ {i.drug_b}")
        print(f"     Severity: {i.severity.value}")
        print(f"     Mechanism: {i.mechanism or 'N/A'}")
        print(f"     Management: {i.management or 'N/A'}")
        print(f"     Source: {i.source}")
        print()
