"""
Baseline Model Evaluation Script for University-FineTuned-Assistant.
Evaluates the unmodified Base Model before LoRA fine-tuning to establish
the baseline performance and capture initial responses.
"""

import os
import sys
import json
import time
from typing import List, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.model_loader import UniversityAssistant
from src.evaluation.metrics import compute_rouge_scores, compute_token_f1, compute_domain_terminology_match


# Key benchmark questions to probe baseline model knowledge
BENCHMARK_QUESTIONS = [
    {
        "question": "What faculties does Apex National University have?",
        "reference": "Apex National University comprises five faculties: Computing & IT, Engineering, Business & Management, Science & Health, and Arts & Humanities."
    },
    {
        "question": "Which department offers the Software Engineering degree at ANU?",
        "reference": "Software Engineering is offered by the Department of Software Engineering within the Faculty of Computing & IT."
    },
    {
        "question": "What is the purpose and operating hours of the Ada Lovelace Memorial Library?",
        "reference": "The Ada Lovelace Memorial Library is open Monday to Friday from 7:00 AM to 11:00 PM, and weekends from 9:00 AM to 9:00 PM, with a 24/7 study hall."
    },
    {
        "question": "What student support services are available at ANU?",
        "reference": "ANU provides Career Guidance & Placement, Counseling & Mental Health, Financial Aid, International Student Services, and Disability & Accessibility Support."
    },
    {
        "question": "What are the core courses offered in the Computer Science curriculum at ANU?",
        "reference": "Core courses include CS101 Introduction to Programming, CS201 Data Structures and Algorithms, SE204 Object-Oriented Design, and DB301 Database Systems."
    },
    {
        "question": "What are the on-campus residence halls at ANU?",
        "reference": "ANU has three on-campus residence halls: Oak Hall (first-year co-ed), Maple Hall (upperclassmen suites), and Trinity Tower (postgraduate/international)."
    },
    {
        "question": "What is the minimum lecture attendance policy at ANU?",
        "reference": "Students must maintain a minimum of 80% attendance in lectures, tutorials, and laboratory sessions to be eligible for final examinations."
    }
]


def run_baseline_evaluation(config_path: str = "configs/config.yaml"):
    print("=" * 65)
    print("          BASELINE MODEL EVALUATION (PRE-FINE-TUNING)")
    print("=" * 65)
    print("Evaluating how the raw Base Model responds to university-specific queries...")

    os.makedirs("outputs/predictions", exist_ok=True)
    os.makedirs("outputs/evaluation", exist_ok=True)

    # Initialize Base Model
    assistant = UniversityAssistant(model_type="base", config_path=config_path)

    results: List[Dict] = []
    predictions: List[str] = []
    references: List[str] = []

    for i, item in enumerate(BENCHMARK_QUESTIONS, 1):
        q = item["question"]
        ref = item["reference"]
        print(f"\n[{i}/{len(BENCHMARK_QUESTIONS)}] Query: {q}")
        
        pred = assistant.generate_response(
            user_message=q,
            max_new_tokens=150,
            temperature=0.1
        )

        f1 = compute_token_f1(pred, ref)
        print(f"-> Base Model Output: {pred}")
        print(f"-> Reference Truth:   {ref}")
        print(f"-> Token F1 Score:     {f1:.4f}")

        results.append({
            "id": i,
            "question": q,
            "reference": ref,
            "baseline_prediction": pred,
            "token_f1": f1
        })
        predictions.append(pred)
        references.append(ref)

    # Compute aggregate baseline metrics
    rouge = compute_rouge_scores(predictions, references)
    term_acc = compute_domain_terminology_match(predictions, references)

    summary = {
        "model": assistant.base_model_name,
        "evaluation_type": "baseline_unfinetuned",
        "sample_count": len(BENCHMARK_QUESTIONS),
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "domain_terminology_accuracy": term_acc,
        "predictions": results
    }

    out_file = "outputs/predictions/baseline_predictions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("                BASELINE RESULTS SUMMARY")
    print("=" * 65)
    print(f"Model:                         {assistant.base_model_name}")
    print(f"ROUGE-1:                       {rouge['rouge1']:.4f}")
    print(f"ROUGE-2:                       {rouge['rouge2']:.4f}")
    print(f"ROUGE-L:                       {rouge['rougeL']:.4f}")
    print(f"Domain Terminology Match:      {term_acc:.1%}")
    print(f"[+] Saved detailed outputs to: {out_file}")
    print("=" * 65 + "\n")

    return summary


if __name__ == "__main__":
    run_baseline_evaluation()
