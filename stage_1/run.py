#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import yaml
from tqdm import tqdm
import argparse
import os
import json
import time

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
        text = report_path.read_text()

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


def save_checkpoint(
    model: nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    step: int,
    best_val_acc: float,
    train_losses: List[float],
    val_accuracies: List[float],
    val_losses: List[float],
    checkpoint_dir: Path,
    is_best: bool = False
) -> None:
    """Save checkpoint with all training state"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_losses': val_losses,
        'timestamp': time.time()
    }
    
    # Save latest checkpoint
    checkpoint_path = checkpoint_dir / 'checkpoint_latest.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save best checkpoint if this is the best model so far
    if is_best:
        best_checkpoint_path = checkpoint_dir / 'checkpoint_best.pt'
        torch.save(checkpoint, best_checkpoint_path)
        # Also save the model and tokenizer separately for easy loading
        model.save_pretrained(checkpoint_dir / 'best_model')
        tokenizer.save_pretrained(checkpoint_dir / 'best_model')
        print(f"Best checkpoint saved to {best_checkpoint_path}")
    
    # Save training history as JSON for easy analysis
    history = {
        'epoch': epoch,
        'step': step,
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_losses': val_losses,
        'timestamp': time.time()
    }
    
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device
) -> Tuple[int, int, float, List[float], List[float], List[float]]:
    """Load checkpoint and restore training state"""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return 0, 0, 0.0, [], [], []
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    best_val_acc = checkpoint['best_val_acc']
    train_losses = checkpoint.get('train_losses', [])
    val_accuracies = checkpoint.get('val_accuracies', [])
    val_losses = checkpoint.get('val_losses', [])
    
    print(f"Resumed from epoch {epoch}, step {step}, best val acc: {best_val_acc:.4f}")
    return epoch, step, best_val_acc, train_losses, val_accuracies, val_losses


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
            config_data = yaml.safe_load(config_yaml.read_text())
            hunk_count = config_data["hunk_count"]
            covered_count = config_data["covered_count"]

            label = (hunk_count == 1) and (covered_count == 1)
            data = (report_txt, label)

            year = int(config_data["datetime"][:4])
            if year < 2024:
                train_data.append(data)
            else:
                test_data.append(data)

    train_labels = [label for _, label in train_data]
    test_labels = [label for _, label in test_data]

    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    print(f"Train positive samples: {sum(train_labels)/len(train_labels):.2%})")
    print(f"Test positive samples: {sum(test_labels)/len(test_labels):.2%})")

    return train_data, test_data


def train_model(
    train_data,
    val_data,
    model_name="roberta-base",
    epochs=3,
    batch_size=64,
    learning_rate=2e-5,
    output_dir="./roberta_vuln_model",
    num_workers=8,
    resume_from_checkpoint=None,
    save_every_n_epochs=1,
):
    """Train RoBERTa model with checkpoint support"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=2, output_attentions=False, output_hidden_states=False
    )

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")

    # Create datasets
    train_dataset = VulnerabilityDataset(train_data, tokenizer)
    val_dataset = VulnerabilityDataset(val_data, tokenizer) if val_data else None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = (
        DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )
        if val_dataset
        else None
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=learning_rate, 
        eps=1e-8,
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Enable mixed precision training
    scaler = torch.amp.GradScaler()

    # Initialize training state
    start_epoch = 0
    global_step = 0
    best_val_acc = 0
    train_losses = []
    val_accuracies = []
    val_losses = []

    # Load checkpoint if specified
    if resume_from_checkpoint:
        start_epoch, global_step, best_val_acc, train_losses, val_accuracies, val_losses = load_checkpoint(
            resume_from_checkpoint, model, tokenizer, optimizer, scheduler, scaler, device
        )

    # Training loop
    model.train()

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = outputs.loss

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            
            progress_bar.set_postfix({
                "loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "step": global_step
            })

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation
        val_acc, val_loss = 0.0, 0.0
        if val_loader:
            val_acc, val_loss = evaluate_model(model, val_loader, device)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            print(f"Validation accuracy: {val_acc:.4f}, Validation loss: {val_loss:.4f}")

        # Check if this is the best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {val_acc:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every_n_epochs == 0 or is_best or epoch == epochs - 1:
            save_checkpoint(
                model, tokenizer, optimizer, scheduler, scaler,
                epoch + 1, global_step, best_val_acc,
                train_losses, val_accuracies, val_losses,
                output_dir / 'checkpoints', is_best
            )

    # Save final model
    final_model_dir = output_dir / 'final_model'
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"Final model saved to {final_model_dir}")

    return model, tokenizer


def evaluate_model(model, data_loader, device):
    """Evaluate model on given data with mixed precision"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
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


def main():
    parser = argparse.ArgumentParser(description="RoBERTa Vulnerability Classification with Checkpointing")
    parser.add_argument(
        "--model-path",
        default=Path(__file__).parent / "roberta_vuln_model",
        help="Path to save/load model",
    )
    parser.add_argument(
        "--epochs", type=int, default=10000, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument(
        "--resume-from-checkpoint", 
        type=str, 
        help="Path to checkpoint to resume from (e.g., ./roberta_vuln_model/checkpoints/checkpoint_latest.pt)"
    )
    parser.add_argument(
        "--save-every-n-epochs", 
        type=int, 
        default=10, 
        help="Save checkpoint every N epochs"
    )
    parser.add_argument("--text", type=str, help="Text for inference mode")
    parser.add_argument("--file", type=str, help="File path for inference mode")

    args = parser.parse_args()

    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
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
        num_workers=args.num_workers,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_every_n_epochs=args.save_every_n_epochs,
    )
    print("Training completed!")


if __name__ == "__main__":
    main()