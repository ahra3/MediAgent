# MediAgent

A multi-agent clinical decision support framework built on LangGraph. MediAgent decomposes clinical reasoning into four specialized, tool-grounded agents that consult external medical databases and enforce citation traceability across all generated recommendations.

The system was designed with production-grade engineering practices:

- **Typed state management** — All inter-agent data flows through a `TypedDict` state schema with Pydantic-validated models, eliminating unstructured data propagation between pipeline stages.
- **Schema-enforced extraction** — The Patient Intake Agent uses Pydantic model validation with automatic re-prompting on schema violations, ensuring downstream agents receive well-formed input.
- **Versioned prompt templates** — Agent prompts are externalized as versioned text files (`src/config/prompts/v1/`), decoupling prompt engineering from application logic.
- **Centralized configuration** — All model parameters, API endpoints, retrieval settings, and file paths are managed through a single `pydantic-settings` configuration class with `.env` override support.
- **Graceful degradation** — The DDI pipeline falls back to a local interaction database when external APIs (RxNorm, DDInter) are unavailable.
- **Human-in-the-loop safety gates** — LangGraph `interrupt()` checkpoints pause execution for clinician review at critical decision points (DDI risk escalation, pre-synthesis approval).
- **Reproducible evaluation** — A structured evaluation harness with automated HITL approval runs the full pipeline against annotated test cases and computes per-case and aggregate metrics.

## System Architecture

MediAgent is implemented as a stateful directed graph using LangGraph, enabling cyclic workflows, persistent state, and native HITL interrupt support.

<!-- Replace the path below with the actual architecture diagram once attached -->
<!-- ![System Architecture](docs/architecture.png) -->

**Pipeline flow:**

```
START → Patient Intake Agent → DDI Agent → [Critical? → HITL DDI Review]
      → Clinical Guidelines RAG Agent → HITL Pre-Synthesis Review
      → Synthesis Agent → END
```

**Patient Intake Agent.** Accepts unstructured clinical text (admission notes, discharge summaries) and produces a validated `PatientProfile` containing demographics, conditions, medications with dosages, allergies, and vital signs. Extraction failures trigger automatic re-prompting with the validation error message.

**DDI Agent.** Takes the extracted medication list, normalizes drug names via the RxNorm API, and queries the DDInter 2.0 database for known interactions. Returns severity levels (`minor`, `moderate`, `major`, `critical`) and clinical management recommendations. If any critical-severity interaction is detected, the pipeline triggers an HITL interrupt for clinician review before proceeding.

**Clinical Guidelines RAG Agent.** Retrieves relevant guideline passages using a hybrid retrieval strategy:
1. BM25 keyword search over chunked guideline text
2. Dense vector retrieval via Gemini embeddings stored in ChromaDB
3. Reciprocal Rank Fusion (RRF) to merge both result sets
4. Cross-encoder re-ranking for final precision filtering

The knowledge base comprises curated clinical practice guidelines covering common chronic conditions (diabetes, hypertension, COPD, asthma, heart failure, CKD).

**Synthesis Agent.** Receives all upstream outputs and generates a structured `ClinicalReport`. The agent prompt enforces a citation format (`[REF:chunk_id]`) for every recommendation. Post-generation validation checks that all cited chunk IDs map to actually retrieved evidence; recommendations lacking valid citations are flagged as ungrounded.

## Evaluation

### Dataset

The evaluation dataset consists of 50 clinical cases derived from the [MTSamples](https://mtsamples.com/) corpus. Each case is annotated with:

- Expected extracted medications and conditions
- Expected drug-drug interaction pairs
- Expected guideline topics
- Expected risk level classification

The dataset generation pipeline (`scripts/generate_silver_dataset.py`) programmatically constructs annotated cases from raw MTSamples transcriptions.

### Results

Evaluated across 50 cases with automated HITL approval:

| Metric | Score |
|---|---|
| Condition Extraction Recall | 81.8% |
| Guideline Retrieval Relevance (Topic Recall) | 100.0% |
| Citation Coverage (of generated recommendations) | 92.2% |
| Median End-to-End Latency | 23s |

The evaluation harness (`eval/evaluate.py`) measures extraction accuracy, DDI detection recall, guideline retrieval relevance, citation coverage, and per-agent latency.

## Technical Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph (stateful directed graph with HITL) |
| LLM | Gemini 3.1 Flash Lite (via `langchain-google-genai`) |
| Embeddings | Gemini Embedding 001 |
| Vector Store | ChromaDB |
| Retrieval | Hybrid BM25 + Dense with RRF fusion |
| Re-ranking | Cross-encoder (`sentence-transformers`) |
| Drug Normalization | RxNorm REST API |
| DDI Database | DDInter 2.0 (Parquet, local fallback JSON) |
| Schema Validation | Pydantic v2 |
| Configuration | pydantic-settings with `.env` support |
| UI | Streamlit |
| Package Manager | uv |
| Python | 3.12+ |

## Repository Structure

```
MediAgent/
├── app.py                          # Streamlit dashboard entry point
├── pyproject.toml                  # Project metadata and dependencies
├── src/
│   ├── agents/
│   │   ├── intake_agent.py         # Patient Intake Agent
│   │   ├── ddi_agent.py            # Drug-Drug Interaction Agent
│   │   ├── guidelines_rag_agent.py # Clinical Guidelines RAG Agent
│   │   └── synthesis_agent.py      # Synthesis Agent
│   ├── graph/
│   │   ├── state.py                # MediAgentState (TypedDict schema)
│   │   └── workflow.py             # LangGraph pipeline construction
│   ├── models/
│   │   ├── patient.py              # PatientProfile model
│   │   ├── interactions.py         # DrugInteraction model
│   │   ├── guidelines.py           # GuidelineMatch model
│   │   └── report.py               # ClinicalReport model
│   ├── tools/
│   │   ├── rxnorm_client.py        # RxNorm API client
│   │   ├── ddi_checker.py          # DDI lookup logic
│   │   └── ddinter_loader.py       # DDInter 2.0 data loader
│   ├── rag/
│   │   ├── ingest.py               # Guideline ingestion and chunking
│   │   ├── hybrid_retriever.py     # BM25 + dense hybrid retriever
│   │   └── reranker.py             # Cross-encoder re-ranker
│   └── config/
│       ├── settings.py             # Centralized configuration
│       ├── prompt_loader.py        # Prompt template loader
│       └── prompts/v1/             # Versioned prompt templates
├── data/
│   ├── guidelines/                 # Clinical guideline documents
│   ├── drug_interactions/          # DDInter data (Parquet + JSON fallback)
│   ├── test_cases/                 # Golden and silver evaluation datasets
│   └── vectorstore/                # ChromaDB persistent storage
├── eval/
│   ├── evaluate.py                 # Evaluation harness
│   └── results/                    # Evaluation output (JSON)
└── scripts/
    └── generate_silver_dataset.py  # Silver dataset generation
```

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A Google API key with Gemini access

### Installation

```bash
git clone https://github.com/ahra3/MediAgent.git
cd MediAgent
uv sync
```

### Configuration

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### Running the Application

```bash
uv run streamlit run app.py
```

### Running the Evaluation

```bash
# Against the golden dataset (5 cases)
uv run python -m eval.evaluate

# Against the silver dataset (50 cases)
uv run python -m eval.evaluate --dataset data/test_cases/silver_dataset.json
```

## Citation

If you use MediAgent in your research, please cite:

```bibtex
@inproceedings{marouf2026mediagent,
  title     = {MediAgent: A Multi-Agent Framework for Reliable Clinical Decision Support with Tool-Grounded Verification},
  author    = {Marouf, Zohra},
  booktitle = {Women in Machine Learning Workshop at NeurIPS},
  year      = {2026}
}
```

The evaluation dataset was derived from the [MTSamples](https://mtsamples.com/) corpus. If you use the evaluation data, please also acknowledge MTSamples:

```bibtex
@misc{mtsamples,
  title = {Medical Transcription Samples},
  author = {MTSamples},
  year = {2024},
  howpublished = {\url{https://mtsamples.com/}}
}
```
