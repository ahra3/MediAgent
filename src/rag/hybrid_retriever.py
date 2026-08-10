"""Hybrid retriever — BM25 keyword search + ChromaDB vector search.

Why hybrid? Medical text is terminology-heavy. Pure vector search
might miss "eGFR < 30" (exact term), while pure BM25 might miss
"kidney problems" → CKD (conceptual match). Combining both gives
dramatically better recall.

Fusion strategy: Reciprocal Rank Fusion (RRF) — merges two ranked
lists into one, weighting by rank position rather than raw scores
(which aren't comparable across BM25 and cosine similarity).
"""

from rank_bm25 import BM25Okapi

from src.rag.ingest import get_chroma_collection, get_embedding_function
from src.config.settings import settings


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


class HybridRetriever:
    """Combines BM25 keyword search with ChromaDB vector search.

    Usage:
        retriever = HybridRetriever()
        results = retriever.search("metformin dose in CKD", top_k=10)
    """

    def __init__(self):
        """Load the ChromaDB collection and build BM25 index."""
        self.collection = get_chroma_collection()
        self.embedder = get_embedding_function()

        # Load all documents from ChromaDB for BM25 indexing
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.doc_ids = all_data["ids"]
        self.doc_texts = all_data["documents"]
        self.doc_metadatas = all_data["metadatas"]

        # Build BM25 index
        tokenized_corpus = [_tokenize(doc) for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """Keyword search using BM25."""
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            if scores[idx] > 0:  # Only include actual matches
                results.append({
                    "chunk_id": self.doc_ids[idx],
                    "text": self.doc_texts[idx],
                    "metadata": self.doc_metadatas[idx],
                    "score": float(scores[idx]),
                    "rank": rank + 1,
                    "source": "bm25",
                })
        return results

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """Semantic search using ChromaDB embeddings."""
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        output = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            output.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - distance,  # Convert distance to similarity
                "rank": i + 1,
                "source": "vector",
            })
        return output

    def search(
        self,
        query: str,
        top_k: int | None = None,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> list[dict]:
        """Hybrid search with Reciprocal Rank Fusion (RRF).

        Args:
            query: Search query string.
            top_k: Number of results to return (defaults to settings.retrieval_top_k).
            bm25_weight: Weight for BM25 results in fusion (default 0.4).
            vector_weight: Weight for vector results in fusion (default 0.6).

        Returns:
            List of result dicts sorted by fused RRF score.
        """
        top_k = top_k or settings.retrieval_top_k

        # Get results from both sources
        bm25_results = self._bm25_search(query, top_k=top_k * 2)
        vector_results = self._vector_search(query, top_k=top_k * 2)

        # Reciprocal Rank Fusion
        # RRF score = sum of 1/(k + rank) across all lists where doc appears
        k = 60  # Standard RRF constant
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        for result in bm25_results:
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + bm25_weight * (1.0 / (k + result["rank"]))
            chunk_data[cid] = result

        for result in vector_results:
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + vector_weight * (1.0 / (k + result["rank"]))
            if cid not in chunk_data:
                chunk_data[cid] = result

        # Sort by fused score
        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        fused_results = []
        for rank, cid in enumerate(ranked_ids):
            entry = chunk_data[cid].copy()
            entry["rrf_score"] = rrf_scores[cid]
            entry["rank"] = rank + 1
            entry["source"] = "hybrid"
            fused_results.append(entry)

        return fused_results


# --- CLI test ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=== Hybrid Retriever Test ===\n")

    retriever = HybridRetriever()
    print(f"Indexed {len(retriever.doc_ids)} chunks\n")

    test_queries = [
        "metformin dose adjustment in chronic kidney disease",
        "blood pressure target for diabetic patients",
        "COPD exacerbation treatment with corticosteroids",
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        results = retriever.search(query, top_k=3)
        for r in results:
            print(f"  [{r['rank']}] RRF: {r['rrf_score']:.4f} | {r['metadata']['source_document']} → {r['metadata']['section']}")
            print(f"      {r['text'][:120]}...")
        print()
