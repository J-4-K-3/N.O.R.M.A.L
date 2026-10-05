"""Production-oriented LoRA fine-tuning template for text models.

This script is intended to be used with domain-specific JSONL datasets and
supports checkpointing, evaluation metrics, and integration with optional
experiment tracking backends such as Weights & Biases or MLflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a local text model with PEFT/LoRA.")
    p.add_argument("--base_model", required=True, help="Base Hugging Face model id or local path")
    p.add_argument("--train_file", required=True, help="JSONL training file with a 'text' field")
    p.add_argument("--output_dir", required=True, help="Directory for the LoRA adapter output")
    p.add_argument("--run_name", default="normal-lora-run")
    p.add_argument("--dataset_version", default="v1")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--tracking_backend", choices=["local", "wandb", "mlflow"], default="local")
    p.add_argument("--checkpoint_frequency", type=int, default=100)
    return p.parse_args()


def compute_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def init_tracking(backend: str, run_name: str, config: Dict[str, Any]):
    if backend == "wandb":
        try:
            import wandb

            wandb.init(project=os.getenv("WANDB_PROJECT", "normal-engine"), entity=os.getenv("WANDB_ENTITY", ""), name=run_name, config=config)
            return wandb
        except Exception:
            return None
    if backend == "mlflow":
        try:
            import mlflow

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.start_run(run_name=run_name)
            mlflow.log_params(config)
            return mlflow
        except Exception:
            return None
    return None


def log_tracking_metrics(tracker: Any, metrics: Dict[str, Any]):
    if tracker is None:
        return
    if hasattr(tracker, "log"):
        tracker.log(metrics)
    elif hasattr(tracker, "log_metrics"):
        tracker.log_metrics(metrics)


def save_training_metadata(output_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "training_metadata.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def main():
    args = parse_args()
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError("PEFT is required for this fine-tuning workflow. Install 'peft' and try again.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")

    dataset_hash = compute_sha256(args.train_file)
    tracker = init_tracking(
        args.tracking_backend,
        args.run_name,
        {
            "base_model": args.base_model,
            "dataset_version": args.dataset_version,
            "dataset_sha256": dataset_hash,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto" if os.getenv("CUDA_VISIBLE_DEVICES") else None,
    )

    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"] if "gpt" in args.base_model.lower() else None,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files={"train": args.train_file})
    records = dataset["train"]
    if "text" not in records.column_names:
        raise ValueError("The training file must contain a 'text' column.")

    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, max_length=args.max_length)

    tokenized = records.map(tokenize_fn, batched=False)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=args.checkpoint_frequency,
        save_total_limit=3,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    eval_metrics = trainer.evaluate() if hasattr(trainer, "evaluate") else {}
    trainer.save_model(str(output_dir))

    report = {
        "run_name": args.run_name,
        "base_model": args.base_model,
        "dataset_version": args.dataset_version,
        "dataset_sha256": dataset_hash,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "metrics": eval_metrics,
        "tracking_backend": args.tracking_backend,
        "status": "completed",
    }

    save_training_metadata(str(output_dir), report)
    if tracker is not None:
        log_tracking_metrics(tracker, eval_metrics)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
