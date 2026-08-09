"""Guidelines ingestion pipeline — structure-aware chunking + ChromaDB embedding.

This module:
1. Reads markdown guideline files from data/guidelines/processed/
2. Splits by ## headers (document-structure-aware chunking)
3. Applies size safety net (splits oversized sections, merges tiny ones)
4. Embeds chunks using Google's text-embedding-004
5. Stores in ChromaDB with rich metadata for retrieval


"""

import hashlib
import re
from pathlib import Path

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config.settings import settings


# --- ChromaDB setup ---
CHROMA_PERSIST_DIR = settings.data_dir / "vectorstore"
COLLECTION_NAME = "clinical_guidelines"


def get_chroma_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection for clinical guidelines."""
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Clinical guideline chunks for MediAgent RAG"},
    )
    return collection


def get_embedding_function() -> GoogleGenerativeAIEmbeddings:
    """Create the embedding function using Google's text-embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{settings.embedding_model}",
        google_api_key=settings.google_api_key,
    )


# --- Structure-aware chunking ---

def _extract_title(content: str) -> str | None:
    """Extract the top-level # title from markdown content."""
    match = re.match(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else None


def _extract_metadata_header(content: str) -> dict:
    """Extract metadata from the header block of a guideline file.

    Looks for Source:, URL:, and License: lines at the top.
    """
    metadata = {}
    for line in content.split("\n")[:10]:  # Only check first 10 lines
        if line.startswith("Source:"):
            metadata["source_org"] = line.replace("Source:", "").strip()
        elif line.startswith("URL:"):
            metadata["source_url"] = line.replace("URL:", "").strip()
    return metadata


def _detect_condition(filename: str) -> str:
    """Infer the primary medical condition from the filename."""
    condition_map = {
        "diabetes": "Type 2 Diabetes",
        "hypertension": "Hypertension",
        "copd": "COPD",
        "heart_failure": "Heart Failure",
        "ckd": "Chronic Kidney Disease",
    }
    for key, condition in condition_map.items():
        if key in filename.lower():
            return condition
    return "General"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate — ~4 chars per token for English text."""
    return len(text) // 4


def chunk_markdown_by_sections(
    content: str,
    source_filename: str,
    max_tokens: int = 800,
    min_tokens: int = 100,
) -> list[dict]:
    """Split markdown content by ## headers into semantically coherent chunks.

    Strategy:
    1. Split on ## headers — each section is a candidate chunk
    2. If a section is > max_tokens, split at paragraph boundaries
    3. If a section is < min_tokens, merge with the next section
    4. Attach rich metadata to each chunk

    Args:
        content: Full markdown file content.
        source_filename: Filename for metadata.
        max_tokens: Max chunk size before splitting further.
        min_tokens: Min chunk size — smaller sections get merged.

    Returns:
        List of chunk dicts with 'text', 'metadata', and 'chunk_id'.
    """
    title = _extract_title(content) or source_filename
    file_metadata = _extract_metadata_header(content)
    condition = _detect_condition(source_filename)

    # Split by ## headers
    sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)

    chunks = []
    buffer_text = ""
    buffer_section = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract section title from the ## header
        header_match = re.match(r"^##\s+(.+)$", section, re.MULTILINE)
        section_title = header_match.group(1).strip() if header_match else "Introduction"

        # Skip the preamble (title + metadata before first ## section)
        if not header_match and not chunks and _estimate_tokens(section) < min_tokens:
            continue

        token_count = _estimate_tokens(section)

        # Case 1: Section too small — buffer it for merging
        if token_count < min_tokens:
            buffer_text += "\n\n" + section if buffer_text else section
            buffer_section = buffer_section or section_title
            continue

        # Flush any buffered content first
        if buffer_text:
            merged = buffer_text + "\n\n" + section
            if _estimate_tokens(merged) <= max_tokens:
                # Merge buffer with current section
                buffer_text = merged
                buffer_section = buffer_section or section_title
                continue
            else:
                # Buffer is big enough on its own — emit it
                chunks.append(_make_chunk(
                    text=buffer_text,
                    section_title=buffer_section,
                    source_document=title,
                    condition=condition,
                    source_filename=source_filename,
                    chunk_index=len(chunks),
                    extra_metadata=file_metadata,
                ))
                buffer_text = ""
                buffer_section = ""

        # Case 2: Section fits within max_tokens — emit as-is
        if token_count <= max_tokens:
            chunks.append(_make_chunk(
                text=section,
                section_title=section_title,
                source_document=title,
                condition=condition,
                source_filename=source_filename,
                chunk_index=len(chunks),
                extra_metadata=file_metadata,
            ))
        else:
            # Case 3: Section too large — split at paragraph/bullet boundaries
            sub_chunks = _split_large_section(
                section, section_title, max_tokens
            )
            for i, sub_text in enumerate(sub_chunks):
                chunks.append(_make_chunk(
                    text=sub_text,
                    section_title=f"{section_title} (part {i + 1})",
                    source_document=title,
                    condition=condition,
                    source_filename=source_filename,
                    chunk_index=len(chunks),
                    extra_metadata=file_metadata,
                ))

    # Flush remaining buffer
    if buffer_text:
        chunks.append(_make_chunk(
            text=buffer_text,
            section_title=buffer_section,
            source_document=title,
            condition=condition,
            source_filename=source_filename,
            chunk_index=len(chunks),
            extra_metadata=file_metadata,
        ))

    return chunks


def _split_large_section(
    text: str, section_title: str, max_tokens: int
) -> list[str]:
    """Split an oversized section at paragraph or bullet boundaries.

    Tries to keep bullets and paragraphs intact rather than
    cutting mid-sentence.
    """
    # Split by double newline (paragraphs) or bullet point patterns
    paragraphs = re.split(r"\n\n+", text)

    sub_chunks = []
    current = ""

    for para in paragraphs:
        candidate = current + "\n\n" + para if current else para
        if _estimate_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                sub_chunks.append(current)
            current = para

    if current:
        sub_chunks.append(current)

    return sub_chunks


def _make_chunk(
    text: str,
    section_title: str,
    source_document: str,
    condition: str,
    source_filename: str,
    chunk_index: int,
    extra_metadata: dict | None = None,
) -> dict:
    """Create a chunk dict with text, metadata, and a deterministic ID."""
    chunk_id = hashlib.md5(
        f"{source_filename}:{section_title}:{chunk_index}".encode()
    ).hexdigest()[:12]

    metadata = {
        "source_document": source_document,
        "section": section_title,
        "condition": condition,
        "source_filename": source_filename,
        "chunk_index": chunk_index,
        "token_estimate": _estimate_tokens(text),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "chunk_id": f"chunk_{chunk_id}",
        "text": text,
        "metadata": metadata,
    }


# --- Embedding + ChromaDB storage ---

def ingest_all_guidelines() -> dict:
    """Process all guideline files and store in ChromaDB.

    This is the main entry point. Run it once to build the
    knowledge base. Subsequent runs will reset and rebuild.

    Returns:
        Summary dict with counts and chunk details.
    """
    guidelines_dir = settings.guidelines_dir
    md_files = sorted(guidelines_dir.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {guidelines_dir}")

    print(f"Found {len(md_files)} guideline files\n")

    # Chunk all files
    all_chunks = []
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        chunks = chunk_markdown_by_sections(
            content=content,
            source_filename=md_file.name,
            max_tokens=settings.chunk_max_tokens,
            min_tokens=settings.chunk_min_tokens,
        )
        print(f"  {md_file.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Embed and store
    print("\nEmbedding chunks with Google text-embedding-004...")
    embedder = get_embedding_function()

    texts = [c["text"] for c in all_chunks]
    # Embed in batches to avoid API limits
    batch_size = 20
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

    # Store in ChromaDB
    print("\nStoring in ChromaDB...")
    collection = get_chroma_collection()

    # Clear existing data (full rebuild)
    existing = collection.count()
    if existing > 0:
        print(f"  Clearing {existing} existing chunks...")
        collection.delete(ids=collection.get()["ids"])

    collection.add(
        ids=[c["chunk_id"] for c in all_chunks],
        documents=texts,
        embeddings=all_embeddings,
        metadatas=[c["metadata"] for c in all_chunks],
    )

    print(f"  Stored {collection.count()} chunks in ChromaDB")

    # Summary
    summary = {
        "total_chunks": len(all_chunks),
        "files_processed": len(md_files),
        "chunks_by_condition": {},
        "avg_tokens": sum(c["metadata"]["token_estimate"] for c in all_chunks) // len(all_chunks),
    }
    for c in all_chunks:
        cond = c["metadata"]["condition"]
        summary["chunks_by_condition"][cond] = summary["chunks_by_condition"].get(cond, 0) + 1

    return summary


# --- CLI entry point ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=== Guidelines Ingestion Pipeline ===\n")

    summary = ingest_all_guidelines()

    print(f"\n=== Ingestion Complete ===")
    print(f"  Files: {summary['files_processed']}")
    print(f"  Total chunks: {summary['total_chunks']}")
    print(f"  Avg tokens/chunk: {summary['avg_tokens']}")
    print(f"  By condition:")
    for condition, count in summary["chunks_by_condition"].items():
        print(f"    {condition}: {count} chunks")

    # Quick retrieval test
    print("\n=== Quick Retrieval Test ===")
    collection = get_chroma_collection()
    embedder = get_embedding_function()

    test_query = "metformin dose adjustment in chronic kidney disease"
    query_embedding = embedder.embed_query(test_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    print(f"Query: '{test_query}'")
    print(f"Top 3 results:")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"\n  [{i + 1}] Score: {1 - dist:.3f} | {meta['source_document']} → {meta['section']}")
        print(f"      {doc[:150]}...")
