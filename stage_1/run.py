#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from pathlib import Path
from typing import List, Tuple
import yaml
from tqdm import tqdm
import argparse

VULN_DIR = Path(__file__).parent.parent / "vuln"


class VulnerabilityDataset(Dataset):
    def __init__(self, data: List[Tuple[Path, bool]], tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        report_path, label = self.data[idx]

        # Read the report text
        try:
            text = report_path.read_text(encoding="utf-8", errors="ignore")
        except:
            text = ""

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def prepare_data() -> Tuple[List[Tuple[Path, bool]], List[Tuple[Path, bool]]]:
    """Prepare training and test data from vulnerability reports"""
    train_data = []
    test_data = []

    for vuln_dir in VULN_DIR.iterdir():
        if not vuln_dir.is_dir():
            continue

        config_yaml = vuln_dir / "config.yaml"
        report_txt = vuln_dir / "report.txt"

        if config_yaml.exists() and report_txt.exists():
            try:
                config_data = yaml.safe_load(config_yaml.read_text())
                hunk_count = config_data["hunk_count"]
                covered_count = config_data["covered_count"]

                # Label is True if both hunk_count and covered_count are exactly 1
                label = (hunk_count == 1) and (covered_count == 1)
                data = (report_txt, label)

                year = int(config_data["datetime"][:4])
                if year < 2024:
                    train_data.append(data)
                else:
                    test_data.append(data)
            except Exception as e:
                print(f"Error processing {vuln_dir}: {e}")
                continue

    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")

    # Print label distribution
    train_labels = [label for _, label in train_data]
    test_labels = [label for _, label in test_data]
    print(
        f"Train positive samples: {sum(train_labels)}/{len(train_labels)} ({sum(train_labels)/len(train_labels):.2%})"
    )
    if test_data:
        print(
            f"Test positive samples: {sum(test_labels)}/{len(test_labels)} ({sum(test_labels)/len(test_labels):.2%})"
        )

    return train_data, test_data


def train_model(
    train_data,
    val_data,
    model_name="roberta-base",
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    output_dir="./roberta_vuln_model",
):
    """Train RoBERTa model for vulnerability classification"""

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=2, output_attentions=False, output_hidden_states=False
    )

    # Create datasets
    train_dataset = VulnerabilityDataset(train_data, tokenizer)
    val_dataset = VulnerabilityDataset(val_data, tokenizer) if val_data else None

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset
        else None
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Training loop
    model.train()
    best_val_acc = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation
        if val_loader:
            val_acc, val_loss = evaluate_model(model, val_loader, device)
            print(
                f"Validation accuracy: {val_acc:.4f}, Validation loss: {val_loss:.4f}"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")

    # Save final model if no validation set
    if not val_loader:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Final model saved.")

    return model, tokenizer


def evaluate_model(model, data_loader, device):
    """Evaluate model on given data"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)

    model.train()
    return accuracy, avg_loss


def inference(model_path, text_input, tokenizer_path=None):
    """Run inference on a single text input"""
    if tokenizer_path is None:
        tokenizer_path = model_path

    # Load model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Tokenize input
    encoding = tokenizer(
        text_input,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()

    return {
        "prediction": prediction,
        "confidence": probabilities[0][prediction].item(),
        "probabilities": {
            "negative": probabilities[0][0].item(),
            "positive": probabilities[0][1].item(),
        },
    }


def test_model(model_path, test_data):
    """Test the trained model on test data"""
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataset = VulnerabilityDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Evaluate
    accuracy, avg_loss = evaluate_model(model, test_loader, device)

    # Get detailed metrics
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    # Print detailed results
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average="weighted"
    )
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nDetailed Classification Report:")
    print(
        classification_report(
            true_labels, predictions, target_names=["Negative", "Positive"]
        )
    )

    return accuracy, predictions, true_labels


def main():
    parser = argparse.ArgumentParser(description="RoBERTa Vulnerability Classification")
    parser.add_argument(
        "--mode",
        choices=["train", "test", "inference"],
        required=True,
        help="Mode: train, test, or inference",
    )
    parser.add_argument(
        "--model-path",
        default=Path(__file__).parent / "roberta_vuln_model",
        help="Path to save/load model",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--text", type=str, help="Text for inference mode")
    parser.add_argument("--file", type=str, help="File path for inference mode")

    args = parser.parse_args()

    if args.mode == "train":
        print("Preparing data...")
        train_data, test_data = prepare_data()

        if not train_data:
            print("No training data found!")
            return

        val_data = test_data
        print("Starting training...")
        model, tokenizer = train_model(
            train_data,
            val_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.model_path,
        )
        print("Training completed!")

    elif args.mode == "test":
        print("Preparing test data...")
        _, test_data = prepare_data()

        if not test_data:
            print("No test data found!")
            return

        print("Testing model...")
        test_model(args.model_path, test_data)

    elif args.mode == "inference":
        if args.text:
            text_input = args.text
        elif args.file:
            text_input = Path(args.file).read_text()
        else:
            print("Please provide either --text or --file for inference")
            return

        print("Running inference...")
        result = inference(args.model_path, text_input)

        print(f"Prediction: {'Positive' if result['prediction'] == 1 else 'Negative'}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(
            f"Probabilities: Negative={result['probabilities']['negative']:.4f}, "
            f"Positive={result['probabilities']['positive']:.4f}"
        )


if __name__ == "__main__":
    main()
