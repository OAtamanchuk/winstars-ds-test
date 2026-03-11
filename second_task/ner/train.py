import argparse
import json
import os

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# Configuration constants
MODEL_NAME = "distilbert-base-cased"  # Pretrained transformer model for token classification
MODEL_DIR = "models/ner_model"        # Directory to save the trained model and tokenizer

# Label vocabulary for NER tags 
LABELS = ["O", "B-ANIMAL"]

# Bidirectional mappings for labels to integers and back, required for model training and inference
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


def load_json_dataset(path):
    """
    Load a dataset stored as a JSON list of examples.
    """
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    # Convert list of dicts into a HuggingFace Dataset object
    return Dataset.from_list(data)


def tokenize_and_align(example, tokenizer):
    """
    Tokenize a single example and align its word-level labels with subword tokens.
    """

    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    word_ids = tokenized.word_ids()

    labels = []

    for word_id in word_ids:

        if word_id is None:
            # special tokens or padding
            labels.append(-100)

        else:
            label = example["labels"][word_id]
            labels.append(label2id[label])

    tokenized["labels"] = labels

    return tokenized


def compute_metrics(eval_pred):
    """
    Compute simple token-level accuracy for evaluation.
    """

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=2)

    true_labels = []
    true_preds = []

    # Flatten predictions/labels, ignoring ignored token markers (-100)
    for pred, lab in zip(predictions, labels):
        for p, l in zip(pred, lab):
            if l != -100:
                true_labels.append(l)
                true_preds.append(p)

    accuracy = np.mean(np.array(true_labels) == np.array(true_preds))

    return {"token_accuracy": accuracy}


def main():

    parser = argparse.ArgumentParser(
        description="Train a token-classification model to extract animal names from text"
    )

    parser.add_argument("--train_path", default="data/ner/train.json",
                        help="Path to training dataset JSON file")
    parser.add_argument("--val_path", default="data/ner/val.json",
                        help="Path to validation dataset JSON file")

    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training/evaluation batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate for optimizer")

    args = parser.parse_args()

    print("Loading datasets...")

    # Read json-formatted train/validation files
    train_data = load_json_dataset(args.train_path)
    val_data = load_json_dataset(args.val_path)

    # Load tokenizer for the chosen pretrained model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing datasets...")

    # Apply tokenization and label alignment to all examples
    train_data = train_data.map(
        lambda x: tokenize_and_align(x, tokenizer),
        batched=False
    )

    val_data = val_data.map(
        lambda x: tokenize_and_align(x, tokenizer),
        batched=False
    )

    print("Loading model...")

    # Instantiate a token classification model with custom label mappings
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Configure training arguments
    training_args = TrainingArguments(

        output_dir="ner_training",  # Directory for checkpoints and logs

        learning_rate=args.lr,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        num_train_epochs=args.epochs,

        evaluation_strategy="epoch",  # Evaluate at end of each epoch
        save_strategy="epoch",        # Save model at end of each epoch

        logging_steps=10,             # Log every 10 steps

        report_to="none"              # Disable external logging 
    )

    # Initialize the Trainer with model, data, and configuration
    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_data,
        eval_dataset=val_data,

        tokenizer=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics
    )

    print("Starting training...")

    trainer.train()

    # Create output directory if necessary and save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print("Model saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()