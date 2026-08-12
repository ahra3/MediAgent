"""Generate a 50-case Silver Standard Evaluation Dataset.

Pulls 50 real clinical notes from the MTSamples dataset (via HuggingFace)
and uses Gemini 2.5 Flash to act as an annotator (Senior Physician),
extracting the exact medications, conditions, and DDIs expected to form
the "Golden Answer Key" for evaluation.
"""

import json
import time
from pathlib import Path
from pydantic import BaseModel, Field

import polars as pl
from datasets import load_dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import settings

from dotenv import load_dotenv
load_dotenv()


class ExpectedDDI(BaseModel):
    drugs: list[str] = Field(description="List of EXACTLY TWO drug names involved in the interaction.")

class ExpectedCase(BaseModel):
    medications_extracted: list[str] = Field(description="List of medications mentioned in the note.")
    conditions: list[str] = Field(description="List of medical conditions/diagnoses mentioned.")
    expected_ddi: list[ExpectedDDI] = Field(description="List of likely drug-drug interactions expected to be found.")


def generate_silver_dataset():
    print("Loading medical_mtsamples dataset...")
    # Using Polars as requested for faster dataframe operations
    ds = load_dataset("NickyNicky/medical_mtsamples", split="train")
    df = ds.to_polars()
    
    # Filter for valid transcripts and sample 50
    df_filtered = df.filter(pl.col("transcription").is_not_null())
    df_filtered = df_filtered.filter(pl.col("transcription").str.len_chars() > 500)
    
    print("Sampling 50 diverse cases...")
    # Take a diverse slice, e.g., using a random sample (seed for reproducibility)
    sampled = df_filtered.sample(n=50, seed=42)
    
    llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0)
    structured_llm = llm.with_structured_output(ExpectedCase)
    
    silver_dataset = []
    
    print("Generating Silver Standard Annotations (this takes ~2 minutes)...")
    for row in sampled.iter_rows(named=True):
        raw_text = row["transcription"]
        # Basic cleanup
        raw_text = raw_text.strip()
        
        prompt = f"""You are a Senior Clinical Annotator.
Read the following clinical note and extract:
1. All active medications (just the names).
2. All diagnosed medical conditions.
3. Any obvious/likely Drug-Drug Interactions (DDIs) among the medications listed. If none are obvious, return an empty list.

CLINICAL NOTE:
{raw_text}
"""
        try:
            # We add a small sleep to avoid rate limits
            time.sleep(1)
            result = structured_llm.invoke(prompt)
            
            ddis = [{"drugs": ddi.drugs} for ddi in result.expected_ddi if len(ddi.drugs) == 2]
            
            case_data = {
                "case_id": f"SILVER-{len(silver_dataset) + 1:03d}",
                "description": row.get("medical_specialty", "Clinical Note").strip(),
                "raw_text": raw_text,
                "expected_output": {
                    "medications_extracted": result.medications_extracted,
                    "conditions": result.conditions,
                    "expected_ddi": ddis
                }
            }
            silver_dataset.append(case_data)
            print(f"Processed SILVER-{len(silver_dataset):03d}")
            
        except Exception as e:
            print(f"Error on note: {e}")
            
    
    # Save the dataset
    out_dir = Path("data/test_cases")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "silver_dataset.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(silver_dataset, f, indent=4)
        
    print(f"\nSuccessfully generated {len(silver_dataset)} cases at {out_path}")

if __name__ == "__main__":
    generate_silver_dataset()
