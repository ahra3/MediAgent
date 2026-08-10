"""Clinical guideline models — output contract for the Guidelines RAG Agent."""

from pydantic import BaseModel, Field


class GuidelineMatch(BaseModel):
    """A single retrieved clinical guideline chunk.

    Returned by the hybrid retriever (BM25 + vector) after
    cross-encoder reranking. Each match traces back to a specific
    source document and section.
    """

    chunk_id: str = Field(description="Unique identifier for this chunk in the vector store")
    source_document: str = Field(
        description="Source guideline document (e.g., 'ADA Standards of Care 2025')"
    )
    section: str | None = Field(
        default=None,
        description="Section or chapter within the guideline (e.g., 'Pharmacologic Treatment')",
    )
    text: str = Field(description="The actual guideline text content")
    condition: str | None = Field(
        default=None,
        description="The medical condition this guideline addresses (e.g., 'Type 2 Diabetes')",
    )
    relevance_score: float = Field(
        description="Relevance score from the reranker (0.0 to 1.0, higher = more relevant)"
    )
    matched_query: str = ""


class RAGResult(BaseModel):
    """Complete output of the Guidelines RAG Agent."""

    matches: list[GuidelineMatch] = Field(
        default_factory=list,
        description="Retrieved guideline chunks, ordered by relevance (highest first)",
    )
    query_used: str = Field(
        description="The search query constructed by the agent from the patient profile",
    )
    retrieval_method: str = Field(
        default="hybrid_bm25_vector",
        description="Which retrieval method was used (for observability)",
    )
