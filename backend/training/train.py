import pandas as pd
import numpy as np
import torch
import os
import sys
import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EvalPrediction,
    TrainerCallback
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle


class ProgressCallback(TrainerCallback):
    """Prints training progress to stdout (visible in Docker logs where tqdm is not)."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step > 0:
            print(f"   Step {state.global_step}/{state.max_steps} — loss: {logs.get('loss', 'N/A')}", flush=True)


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_bert_model():
    # Get the directory where this script lives (works in Docker and locally)
    training_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Detect if running in Docker/Codespace (limited resources)
    is_docker = os.environ.get("DOCKER_ENV", "").lower() == "true"
    if is_docker:
        print("🐳 Docker/Codespace detected — using lightweight training settings", flush=True)
    
    print("Loading dataset...", flush=True)
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
    df = pd.DataFrame(dataset['train'])
    
    # Filter for top 10 categories for better quality
    top_queues = df['queue'].value_counts().nlargest(10).index.tolist()
    print(f"Fine-tuning on Top 10 queues: {top_queues}", flush=True)
    df = df[df['queue'].isin(top_queues)]

    # ENSURE TEXT IS STRING AND NOT NULL
    df = df.dropna(subset=['body'])
    df['body'] = df['body'].astype(str)
    
    # In Docker/Codespace, use a tiny subset — CPU-only training must be fast
    if is_docker and len(df) > 500:
        print(f"   Sampling 500 from {len(df)} rows for fast CPU training...", flush=True)
        df = df.sample(n=500, random_state=42)
    
    # Label Encoding
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['queue'])
    
    # Save Label Encoder
    os.makedirs(training_dir, exist_ok=True)
    with open(os.path.join(training_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    
    # Prepare HF Dataset
    hf_dataset = Dataset.from_pandas(df[['body', 'label']])
    hf_dataset = hf_dataset.rename_column("body", "text")
    
    # Split
    hf_dataset = hf_dataset.train_test_split(test_size=0.1)
    
    # Tokenization
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    print("Tokenizing data...", flush=True)
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    
    # Model
    num_labels = len(top_queues)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Training Arguments — ultra-light for Docker/Codespace
    if is_docker:
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            eval_strategy="no",
            save_strategy="no",
            fp16=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            disable_tqdm=True,
        )
    else:
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
    
    total_steps = (len(tokenized_datasets["train"]) // training_args.per_device_train_batch_size) * int(training_args.num_train_epochs)
    print(f"Training: {int(training_args.num_train_epochs)} epoch(s), ~{total_steps} steps, batch_size={training_args.per_device_train_batch_size}", flush=True)
    
    # Trainer — add progress callback for Docker (tqdm doesn't show in Docker logs)
    callbacks = [ProgressCallback()] if is_docker else []
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    print("Starting fine-tuning...", flush=True)
    trainer.train()
    
    # Save model and tokenizer
    save_path = os.path.join(training_dir, 'models', 'fine_tuned_bert')
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"✅ Training complete. Model saved to {save_path}", flush=True)

if __name__ == "__main__":
    train_bert_model()

