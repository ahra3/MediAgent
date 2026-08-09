from pathlib import Path
from pydantic_settings import BaseSettings 



class Settings (BaseSettings):
    # --- LLM Configuration ---
    llm_model: str = "gemini-3.1-flash-lite"
    llm_temperature_deterministic: float = 0.0  # Intake + DDI agents (need consistency)
    llm_temperature_creative: float = 0.3  # Synthesis agent (natural language)
   
   
    # --- Embedding Configuration ---
    embedding_model: str = "gemini-embedding-001"
  
    # --- API Keys ---
    google_api_key: str = ""  # Loaded from .env via GOOGLE_API_KEY
  
  
    # --- Paths ---
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data"
    ddinter_parquet: Path = data_dir / "drug_interactions" / "ddinter_combined.parquet"
    fallback_json: Path = data_dir / "drug_interactions" / "common_interactions.json"
    guidelines_dir: Path = data_dir / "guidelines" / "processed"
    test_cases_path: Path = data_dir / "test_cases" / "golden_dataset.json"
 
 
    # --- RxNorm API ---
    rxnorm_base_url: str = "https://rxnav.nlm.nih.gov/REST"
    rxnorm_timeout: int = 10  # seconds
  
  
    # --- RAG Configuration ---
    # Section-aware chunking (primary split by ## headers)
    chunk_max_tokens: int = 800      # max size before splitting a section further
    chunk_min_tokens: int = 100      # min size — merge small sections together
    chunk_overlap: int = 50          # only used when splitting oversized sections
    retrieval_top_k: int = 10        # initial retrieval count
    rerank_top_k: int = 5            # final count after reranking           
    
    
    model_config = {"env_file": ".env", "extra": "ignore"}

settings = Settings()