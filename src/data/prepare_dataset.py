"""
Data Preparation Pipeline for University-FineTuned-Assistant.

Loads raw CSV data, performs validation and cleaning, converts examples
into standard chat template format (messages), splits into train/val/test splits,
and exports JSONL files while reporting key dataset metrics.
"""

import os
import sys
import json
import yaml
import re
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "model": {
            "system_prompt": "You are the official University Information Assistant for Apex National University (ANU). Provide clear, accurate, professional, and friendly answers regarding university faculties, programs, student services, policies, and campus life."
        },
        "data": {
            "raw_csv_path": "data/raw/university_data.csv",
            "processed_dir": "data/processed",
            "train_jsonl": "data/processed/train.jsonl",
            "validation_jsonl": "data/processed/validation.jsonl",
            "test_jsonl": "data/processed/test.jsonl",
            "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15}
        }
    }


def clean_text(text: str) -> str:
    """Normalize whitespace and remove unwanted control characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_data(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    raw_path = config["data"]["raw_csv_path"]
    processed_dir = config["data"]["processed_dir"]
    system_prompt = config["model"]["system_prompt"]
    split_ratios = config["data"]["split_ratios"]

    # If raw CSV is missing, generate it
    if not os.path.exists(raw_path):
        print(f"[!] Raw dataset not found at '{raw_path}'. Generating synthetic dataset now...")
        from src.data.generate_dataset import save_raw_dataset
        save_raw_dataset(raw_path)

    # 1. Load raw data
    print(f"[*] Loading raw dataset from '{raw_path}'...")
    df = pd.read_csv(raw_path)
    initial_count = len(df)
    print(f"[*] Raw records loaded: {initial_count}")

    # 2. Validate required columns
    required_cols = ["category", "subcategory", "question", "answer", "source"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in raw dataset.")

    # 3. Clean and validate records
    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)
    df["category"] = df["category"].apply(clean_text)
    df["subcategory"] = df["subcategory"].apply(clean_text)
    df["source"] = df["source"].apply(clean_text)

    # Remove empty or whitespace-only records
    df = df[(df["question"].str.len() > 3) & (df["answer"].str.len() > 3)]

    # Deduplicate based on exact question string
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)
    cleaned_count = len(df)
    print(f"[*] Cleaned and deduplicated records: {cleaned_count} (removed {initial_count - cleaned_count} duplicates/invalid)")

    # 4. Format into instruction chat format (messages)
    chat_records: List[Dict] = []
    for _, row in df.iterrows():
        record = {
            "category": row["category"],
            "subcategory": row["subcategory"],
            "source": row["source"],
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": row["question"]
                },
                {
                    "role": "assistant",
                    "content": row["answer"]
                }
            ]
        }
        chat_records.append(record)

    # 5. Split dataset into Train, Validation, Test
    train_ratio = split_ratios.get("train", 0.70)
    val_ratio = split_ratios.get("validation", 0.15)
    test_ratio = split_ratios.get("test", 0.15)

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_data, temp_data = train_test_split(
        chat_records,
        test_size=temp_ratio,
        random_state=42,
        shuffle=True
    )

    # Second split: Val vs Test
    val_relative_ratio = val_ratio / temp_ratio
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1.0 - val_relative_ratio),
        random_state=42,
        shuffle=True
    )

    # 6. Save JSONL files
    os.makedirs(processed_dir, exist_ok=True)
    train_path = config["data"]["train_jsonl"]
    val_path = config["data"]["validation_jsonl"]
    test_path = config["data"]["test_jsonl"]

    def write_jsonl(file_path: str, data: List[Dict]):
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(train_path, train_data)
    write_jsonl(val_path, val_data)
    write_jsonl(test_path, test_data)

    # 7. Calculate and display dataset statistics
    all_q_lens = [len(r["messages"][1]["content"].split()) for r in chat_records]
    all_a_lens = [len(r["messages"][2]["content"].split()) for r in chat_records]
    cat_counts = df["category"].value_counts().to_dict()

    print("\n" + "="*50)
    print("        DATASET PREPARATION SUMMARY")
    print("="*50)
    print(f"Total examples:         {len(chat_records)}")
    print(f"Training set:           {len(train_data)} ({len(train_data)/len(chat_records):.1%}) -> {train_path}")
    print(f"Validation set:         {len(val_data)} ({len(val_data)/len(chat_records):.1%}) -> {val_path}")
    print(f"Testing set:            {len(test_data)} ({len(test_data)/len(chat_records):.1%}) -> {test_path}")
    print("-" * 50)
    print("Categories distribution:")
    for cat, cnt in cat_counts.items():
        print(f"  - {cat:<20}: {cnt:>4} examples")
    print("-" * 50)
    print(f"Average question length: {sum(all_q_lens)/len(all_q_lens):.1f} words")
    print(f"Average answer length:   {sum(all_a_lens)/len(all_a_lens):.1f} words")
    print("="*50 + "\n")

    return {
        "total": len(chat_records),
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path
    }


if __name__ == "__main__":
    prepare_data()
