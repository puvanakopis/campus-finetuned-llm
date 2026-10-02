"""
Model Loader and Inference Engine for University-FineTuned-Assistant.
Supports loading Base Model and LoRA Adapter on CPU, CUDA, and Apple Silicon.
"""

import os
import yaml
import torch
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class UniversityAssistant:
    def __init__(
        self,
        model_type: str = "lora",  # "lora" or "base"
        config_path: str = "configs/config.yaml",
        device: Optional[str] = None
    ):
        self.config = load_config(config_path)
        self.model_type = model_type
        self.base_model_name = self.config["model"]["base_model_name_or_path"]
        self.adapter_dir = self.config["model"]["output_adapter_dir"]
        self.system_prompt = self.config["model"].get(
            "system_prompt",
            "You are the official University Information Assistant."
        )

        # Determine target device
        if device is None or device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.torch_dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32

        print(f"[*] Initializing Assistant on [{self.device.upper()}] (Model Type: {self.model_type.upper()})...")
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        # 1. Load Tokenizer
        tokenizer_source = self.adapter_dir if (self.model_type == "lora" and os.path.exists(self.adapter_dir)) else self.base_model_name
        print(f"[*] Loading tokenizer from '{tokenizer_source}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Base Model
        print(f"[*] Loading base model '{self.base_model_name}'...")
        device_map = "auto" if self.device == "cuda" else None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        # 3. Load LoRA Adapter if requested
        if self.model_type == "lora":
            if os.path.exists(self.adapter_dir) and os.path.exists(os.path.join(self.adapter_dir, "adapter_config.json")):
                print(f"[*] Loading LoRA adapter weights from '{self.adapter_dir}'...")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_dir)
            else:
                print(f"[!] Warning: Adapter directory '{self.adapter_dir}' not found. Falling back to base model.")
                self.model_type = "base"

        self.model.eval()
        print(f"[+] Model loaded successfully ({self.model_type.upper()}).\n")

    def generate_response(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.3,
        top_p: float = 0.9
    ) -> str:
        """Generate response given a user query and conversation history."""
        sys_prompt = system_prompt or self.system_prompt
        messages = [{"role": "system", "content": sys_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})

        # Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode generated text only (remove prompt input tokens)
        generated_tokens = output_tokens[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response
