import pandas as pd
import numpy as np
import torch
import os
import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EvalPrediction
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle

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
    
    print("Loading dataset...")
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
    df = pd.DataFrame(dataset['train'])
    
    # Filter for top 10 categories for better quality
    top_queues = df['queue'].value_counts().nlargest(10).index.tolist()
    print(f"Fine-tuning on Top 10 queues: {top_queues}")
    df = df[df['queue'].isin(top_queues)]

    # ENSURE TEXT IS STRING AND NOT NULL
    df = df.dropna(subset=['body'])
    df['body'] = df['body'].astype(str)
    
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
    
    print("Tokenizing data...")
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    
    # Model
    num_labels = len(top_queues)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Training Arguments
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
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save model and tokenizer
    save_path = os.path.join(training_dir, 'models', 'fine_tuned_bert')
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    train_bert_model()
