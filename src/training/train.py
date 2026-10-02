"""
Fine-Tuning Pipeline for University-FineTuned-Assistant.
Uses Hugging Face Transformers, PEFT (LoRA/QLoRA), and TRL SFTTrainer.
Configurable via configs/config.yaml.
"""

import os
import sys
import json
import time
import yaml
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer, SFTConfig

from src.training.utils import (
    get_hardware_info,
    print_hardware_summary,
    print_trainable_parameters
)
from src.data.prepare_dataset import prepare_data


def load_yaml_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_chat_prompt(example, tokenizer):
    """Format standard messages into model-specific chat template."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}


def train(config_path: str = "configs/config.yaml"):
    start_time = time.time()
    config = load_yaml_config(config_path)

    print("=" * 60)
    print("   UNIVERSITY DOMAIN FINE-TUNING PIPELINE (LoRA/QLoRA)")
    print("=" * 60)

    # 1. Ensure dataset exists
    train_path = config["data"]["train_jsonl"]
    val_path = config["data"]["validation_jsonl"]
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("[!] Processed datasets not found. Running data preparation...")
        prepare_data(config_path)

    # 2. Hardware Detection
    hw_info = get_hardware_info()
    print_hardware_summary(hw_info)

    model_id = config["model"]["base_model_name_or_path"]
    output_dir = config["model"]["output_adapter_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("outputs/training_logs", exist_ok=True)

    # 3. Quantization Configuration (QLoRA)
    bnb_config = None
    use_4bit = config["training"].get("use_4bit_quantization", False) and hw_info["supports_4bit"]
    
    if use_4bit:
        print("[*] Configuring 4-bit NormalFloat (NF4) Quantization for QLoRA...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    else:
        print(f"[*] Running without 4-bit quantization (Device: {hw_info['device']}).")

    # 4. Load Tokenizer
    print(f"[*] Loading Tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. Load Base Model
    print(f"[*] Loading Base Model '{model_id}'...")
    device_map = "auto" if hw_info["cuda_available"] else None
    torch_dtype = hw_info["recommended_dtype"]

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = device_map
    elif hw_info["cuda_available"]:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs
    )

    if not hw_info["cuda_available"] and hw_info["device"] != "cpu":
        model = model.to(hw_info["device"])

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # 6. LoRA Configuration
    lora_cfg = config["lora"]
    print(f"[*] Initializing LoRA configuration (r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']})...")
    
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"])
    )

    model = get_peft_model(model, peft_config)
    trainable_params, all_param, pct_trainable = print_trainable_parameters(model)

    # 7. Load Datasets
    print("[*] Loading and formatting datasets...")
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    val_dataset = load_dataset("json", data_files=val_path, split="train")

    train_dataset = train_dataset.map(lambda x: format_chat_prompt(x, tokenizer))
    val_dataset = val_dataset.map(lambda x: format_chat_prompt(x, tokenizer))

    # 8. Training Arguments
    train_cfg = config["training"]
    training_args = SFTConfig(
        output_dir="outputs/training_logs/checkpoints",
        dataset_text_field="text",
        max_length=config["model"].get("max_seq_length", 128),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", 20),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        warmup_steps=train_cfg.get("warmup_steps", 2),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        logging_steps=train_cfg.get("logging_steps", 5),
        save_strategy="no",
        fp16=train_cfg.get("fp16", False) and hw_info["cuda_available"],
        bf16=train_cfg.get("bf16", False) and hw_info["cuda_available"],
        optim=train_cfg.get("optimizer", "adamw_torch"),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        report_to="none"
    )

    # 9. Initialize SFTTrainer
    print("[*] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer
    )

    # 10. Execute Training
    print("\n" + "=" * 60)
    print("                STARTING FINE-TUNING")
    print("=" * 60)
    train_result = trainer.train()
    elapsed_time = time.time() - start_time

    print(f"\n[+] Fine-tuning completed in {elapsed_time/60:.2f} minutes!")

    # 11. Save Adapter, Tokenizer, and Metrics
    print(f"[*] Saving fine-tuned LoRA adapter to '{output_dir}'...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training summary log
    train_metrics = train_result.metrics
    train_metrics["training_time_seconds"] = round(elapsed_time, 2)
    train_metrics["trainable_parameters"] = trainable_params
    train_metrics["total_parameters"] = all_param
    train_metrics["trainable_percentage"] = round(pct_trainable, 4)
    train_metrics["hardware"] = hw_info["device_name"]
    train_metrics["base_model"] = model_id

    summary_file = "outputs/training_logs/training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)

    print(f"[+] Training log saved to '{summary_file}'")
    print("=" * 60)
    return train_result


if __name__ == "__main__":
    train()
