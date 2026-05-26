"""
QLoRA Trainer: fine-tunes LLaMA-3.2-1B-Instruct on Grok-4 annotated failure traces.
Uses 4-bit NF4 quantization + LoRA adapters (~2% trainable params).

Usage:
    python -m distillation.qlora_trainer \
        --data data/train.jsonl \
        --output models/student_lora
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with input. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


@dataclass
class TrainingConfig:
    data_path: str = "data/train.jsonl"
    output_dir: str = "models/student_lora"
    base_model: str = BASE_MODEL
    # QLoRA
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    # LoRA
    lora_r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    save_steps: int = 50
    logging_steps: int = 10
    fp16: bool = True


def load_dataset(data_path: str):
    """Load JSONL training data and format with prompt template."""
    samples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            text = PROMPT_TEMPLATE.format(
                instruction=sample.get("instruction", ""),
                input=sample.get("input", ""),
                output=sample.get("output", ""),
            )
            samples.append({"text": text})
    print(f"Loaded {len(samples)} training samples from {data_path}")
    return samples


def train(config: Optional[TrainingConfig] = None):
    """Run QLoRA fine-tuning."""
    if config is None:
        config = TrainingConfig()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. "
            "Install with: pip install torch transformers peft trl bitsandbytes datasets"
        )

    # --- 4-bit quantization config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {config.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    raw_samples = load_dataset(config.data_path)
    dataset = Dataset.from_list(raw_samples)

    # --- Training args ---
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()

    # Save adapter + training manifest
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    manifest = {
        "base_model": config.base_model,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "target_modules": LORA_TARGET_MODULES,
        "num_epochs": config.num_epochs,
        "data_path": config.data_path,
    }
    with open(output_dir / "training_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Adapter saved to {output_dir}")
    return str(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for failure analysis student")
    parser.add_argument("--data", required=True, help="Path to training JSONL")
    parser.add_argument("--output", required=True, help="Output directory for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    cfg = TrainingConfig(
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )
    train(cfg)
