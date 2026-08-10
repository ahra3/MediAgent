"""Cross-encoder reranker — precision boost for RAG retrieval.

Why rerank? The hybrid retriever (BM25 + vector) optimizes for RECALL —
casting a wide net. The cross-encoder optimizes for PRECISION — it reads
the query and each chunk TOGETHER to judge true relevance.

This two-stage approach (retrieve many → rerank few) is the standard
production RAG pattern. It's how Google Search works internally.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- 22MB, runs on CPU in ~50ms per query
- Trained on MS MARCO passage ranking (500K real search queries)
"""

from sentence_transformers import CrossEncoder

from src.config.settings import settings

# Lazy-loaded singleton — avoid loading the model on every import
_reranker_model: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Load the cross-encoder model (lazy singleton)."""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )
    return _reranker_model


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """Rerank candidate chunks using cross-encoder scoring.

    Args:
        query: The search query.
        candidates: List of result dicts from the hybrid retriever.
            Each must have a 'text' key.
        top_k: Number of top results to return (defaults to settings.rerank_top_k).

    Returns:
        Reranked list of result dicts with 'rerank_score' added, sorted
        by cross-encoder score (highest first).
    """
    top_k = top_k or settings.rerank_top_k

    if not candidates:
        return []

    model = _get_reranker()

    # Build query-document pairs for the cross-encoder
    pairs = [(query, candidate["text"]) for candidate in candidates]

    # Score all pairs
    scores = model.predict(pairs)

    # Attach scores and sort
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_k]


# --- CLI test ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from src.rag.hybrid_retriever import HybridRetriever

    print("=== Reranker Test ===\n")

    retriever = HybridRetriever()

    query = "metformin dose adjustment in chronic kidney disease"
    print(f"Query: '{query}'\n")

    # Step 1: Hybrid retrieval (top 10)
    candidates = retriever.search(query, top_k=10)
    print(f"Hybrid retrieval returned {len(candidates)} candidates")
    print("Before reranking:")
    for r in candidates[:5]:
        print(f"  [{r['rank']}] RRF: {r['rrf_score']:.4f} | {r['metadata']['section']}")

    # Step 2: Rerank (top 5)
    reranked = rerank(query, candidates, top_k=5)
    print(f"\nAfter reranking (top {len(reranked)}):")
    for i, r in enumerate(reranked):
        print(f"  [{i+1}] CE Score: {r['rerank_score']:.4f} | {r['metadata']['source_document']} → {r['metadata']['section']}")
        print(f"      {r['text'][:120]}...")
    print()
