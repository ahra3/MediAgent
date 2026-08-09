import  httpx
from src.config.settings import settings


async def get_rxcui_by_name(drug_name: str) -> dict | None: # get the RxCUIT from the drug name 
    
    url = f"{settings.rxnorm_base_url}/rxcui.json"
    params = {"name": drug_name, "search": 2}  
    async with httpx.AsyncClient(timeout=settings.rxnorm_timeout) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            id_group = data.get("idGroup", {})
            rxnorm_ids = id_group.get("rxnormId")
            if rxnorm_ids:
                return {
                    "rxcui": rxnorm_ids[0],
                    "name": id_group.get("name", drug_name),
                    "source": "rxnorm_exact",
                }
        except (httpx.HTTPError, httpx.TimeoutException, KeyError):
            pass
    return None


async def get_approximate_match(drug_name: str) -> dict | None:
    
    url = f"{settings.rxnorm_base_url}/approximateTerm.json"
    params = {"term": drug_name, "maxEntries": 5}
    
    
    async with httpx.AsyncClient(timeout=settings.rxnorm_timeout) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            candidates = (
                data
                .get("approximateGroup", {})
                .get("candidate", [])
            )
            if candidates:
                best = candidates[0]  # Already ranked by match quality
                # Resolve the RxCUI to get the actual drug name
                name = await _get_name_by_rxcui(best["rxcui"])
                return {
                    "rxcui": best["rxcui"],
                    "name": name or drug_name,
                    "score": best.get("score", "N/A"),
                    "source": "rxnorm_approximate",
                }
        except (httpx.HTTPError, httpx.TimeoutException, KeyError):
            pass
    return None


async def _get_name_by_rxcui(rxcui: str) -> str | None:

    url = f"{settings.rxnorm_base_url}/rxcui/{rxcui}/properties.json"
    async with httpx.AsyncClient(timeout=settings.rxnorm_timeout) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get("properties", {}).get("name")
        except (httpx.HTTPError, httpx.TimeoutException, KeyError):
            return None


async def _get_ingredient_by_rxcui(rxcui: str) -> str | None:
    """Get the active ingredient (TTY=IN) for a given RxCUI.
    
    This converts brand names (like Glucophage) into their generic 
    ingredients (like metformin) which are required for DDInter lookups.
    """
    url = f"{settings.rxnorm_base_url}/rxcui/{rxcui}/allrelated.json"
    async with httpx.AsyncClient(timeout=settings.rxnorm_timeout) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            groups = data.get("allRelatedGroup", {}).get("conceptGroup", [])
            for group in groups:
                if group.get("tty") == "IN":  # Ingredient term type
                    concepts = group.get("conceptProperties", [])
                    if concepts:
                        return concepts[0]["name"].lower()
        except (httpx.HTTPError, httpx.TimeoutException, KeyError):
            pass
            
    return None


async def normalize_drug_name(drug_name: str) -> dict:
    cleaned = drug_name.strip()
    
    # Try 1: Exact match
    result = await get_rxcui_by_name(cleaned)
    
    # Try 2: Approximate match if exact fails
    if not result:
        result = await get_approximate_match(cleaned)

    if result:
        rxcui = result["rxcui"]
        # Convert whatever we found into its base ingredient (e.g., brand -> generic)
        ingredient = await _get_ingredient_by_rxcui(rxcui)
        
        # Fall back to the returned name if ingredient lookup fails
        normalized_name = ingredient or result["name"].lower()
        
        return {
            "original": drug_name,
            "normalized": normalized_name.lower(),
            "rxcui": rxcui,
            "source": result["source"],
            "resolved": True,
        }
        
    # Failed: return original name so pipeline doesn't break
    return {
        "original": drug_name,
        "normalized": cleaned.lower(),
        "rxcui": None,
        "source": "unresolved",
        "resolved": False,
    }

    

async def normalize_medication_list(drug_names: list[str]) -> dict:

    mappings = []
    normalized_names = []
    failures = []
    for name in drug_names:
        result = await normalize_drug_name(name)
        mappings.append(result)
        normalized_names.append(result["normalized"])
        if not result["resolved"]:
            failures.append(name)
    return {
        "mappings": mappings,
        "normalized_names": normalized_names,
        "failures": failures,
    }
    
    
# CLI test entry point :
if __name__ == "__main__":
    import asyncio
    async def test():
        print("=== RxNorm Normalization Test ===\n")
        test_names = [
            "Metformin",       # Exact generic
            "Advil",           # Brand → Ibuprofen
            "Glucophage",      # Brand → Metformin
            "Zoloft",          # Brand → Sertraline
            "Lasix",           # Brand → Furosemide
            "aspirin",         # Common name
            "Rantidine",       # Misspelling → Ranitidine
            "FakeNonDrug123",  # Should fail gracefully
        ]
        for name in test_names:
            result = await normalize_drug_name(name)
            status = "✅" if result["resolved"] else "❌"
            print(f"  {status} {name:20s} → {result['normalized']:25s} (via {result['source']})")
    asyncio.run(test())