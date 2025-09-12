#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    LongformerTokenizer,
    LongformerForSequenceClassification,
    LongformerConfig,
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
    def __init__(self, data: List[Tuple[Path, bool]], tokenizer, max_length=4096):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        report_path, label = self.data[idx]
        text = report_path.read_text(encoding="utf-8")

        # Add special tokens and ensure proper formatting
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,  # Ensure special tokens are added
        )

        # Ensure input_ids don't exceed vocab size
        input_ids = encoding["input_ids"].flatten()
        vocab_size = self.tokenizer.vocab_size

        # Clamp any out-of-bounds token IDs
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_extended_longformer_model(base_model_name="allenai/longformer-base-4096", max_pos=4096):
    """Create a Longformer model extended to support longer sequences"""

    # Load the tokenizer first to check vocab size
    tokenizer = LongformerTokenizer.from_pretrained(base_model_name)

    # Load the base configuration and modify it
    config = LongformerConfig.from_pretrained(base_model_name)
    original_max_pos = config.max_position_embeddings

    # Only extend if we need more positions than the base model supports
    if max_pos <= original_max_pos:
        print(f"Using base model max positions: {original_max_pos}")
        max_pos = original_max_pos

    config.max_position_embeddings = max_pos
    config.num_labels = 2
    config.output_attentions = False
    config.output_hidden_states = False

    # Set tokenizer max length
    tokenizer.model_max_length = max_pos

    print(f"Loading base model with config: max_pos={max_pos}, vocab_size={config.vocab_size}")

    # Load the base model with updated config
    model = LongformerForSequenceClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)

    # Only extend position embeddings if needed
    if max_pos > original_max_pos:
        print(f"Extending position embeddings from {original_max_pos} to {max_pos}")

        old_pos_emb = model.longformer.embeddings.position_embeddings
        old_max_pos = old_pos_emb.weight.size(0)

        # Create new position embeddings with proper initialization
        new_pos_emb = nn.Embedding(max_pos, config.hidden_size)

        with torch.no_grad():
            # Copy existing embeddings
            new_pos_emb.weight[:old_max_pos] = old_pos_emb.weight

            # Initialize new positions using interpolation instead of repetition
            if max_pos > old_max_pos:
                # Use the last position embedding for new positions initially
                last_pos_emb = old_pos_emb.weight[-1:].clone()
                for i in range(old_max_pos, max_pos):
                    new_pos_emb.weight[i] = last_pos_emb.squeeze(0)

        # Replace the embedding layer
        model.longformer.embeddings.position_embeddings = new_pos_emb

        # Update the model's position_ids buffer to handle the new max length
        if hasattr(model.longformer.embeddings, "position_ids"):
            model.longformer.embeddings.register_buffer("position_ids", torch.arange(max_pos).expand((1, -1)), persistent=False)

    return model, tokenizer


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
    is_best: bool = False,
) -> None:
    """Save checkpoint with all training state"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "val_losses": val_losses,
        "timestamp": time.time(),
    }

    # Save latest checkpoint
    checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Save best checkpoint if this is the best model so far
    if is_best:
        best_checkpoint_path = checkpoint_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_checkpoint_path)
        # Also save the model and tokenizer separately for easy loading
        model.save_pretrained(checkpoint_dir / "best_model")
        tokenizer.save_pretrained(checkpoint_dir / "best_model")
        print(f"Best checkpoint saved to {best_checkpoint_path}")

    # Save training history as JSON for easy analysis
    history = {
        "epoch": epoch,
        "step": step,
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "val_losses": val_losses,
        "timestamp": time.time(),
    }

    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[int, int, float, List[float], List[float], List[float]]:
    """Load checkpoint and restore training state"""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return 0, 0, 0.0, [], [], []

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dicts with error handling
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    except Exception as e:
        print(f"Warning: Could not load some checkpoint components: {e}")
        # Continue with partial loading

    epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    best_val_acc = checkpoint["best_val_acc"]
    train_losses = checkpoint.get("train_losses", [])
    val_accuracies = checkpoint.get("val_accuracies", [])
    val_losses = checkpoint.get("val_losses", [])

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
            try:
                config_data = yaml.safe_load(config_yaml.read_text())

                # Validate required keys exist
                if not all(key in config_data for key in ["hunk_count", "covered_count", "datetime"]):
                    print(f"Skipping {vuln_dir}: missing required keys in config.yaml")
                    continue

                hunk_count = config_data["hunk_count"]
                covered_count = config_data["covered_count"]

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

    train_labels = [label for _, label in train_data]
    test_labels = [label for _, label in test_data]

    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    print(f"Train positive samples: {sum(train_labels)/len(train_labels):.2%}")  # Fixed syntax
    print(f"Test positive samples: {sum(test_labels)/len(test_labels):.2%}")  # Fixed syntax

    return train_data, test_data


def train_model(
    train_data,
    val_data,
    base_model_name="allenai/longformer-base-4096",
    epochs=3,
    batch_size=2,  # Reduced further for 4096 tokens
    learning_rate=2e-5,
    output_dir="./longformer_vuln_model",
    num_workers=4,  # Reduced workers
    resume_from_checkpoint=None,
    save_every_n_epochs=1,
    max_length=4096,
):
    """Train extended Longformer model with checkpoint support"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating extended Longformer model with {max_length} max tokens...")
    # Initialize extended model and tokenizer
    model, tokenizer = create_extended_longformer_model(base_model_name=base_model_name, max_pos=max_length)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    print(f"Base model: {base_model_name}")
    print(f"Extended max sequence length: {max_length}")
    print(f"Batch size: {batch_size}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Create datasets with the specified max_length
    train_dataset = VulnerabilityDataset(train_data, tokenizer, max_length=max_length)
    val_dataset = VulnerabilityDataset(val_data, tokenizer, max_length=max_length) if val_data else None

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
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=0.01)

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
        (
            start_epoch,
            global_step,
            best_val_acc,
            train_losses,
            val_accuracies,
            val_losses,
        ) = load_checkpoint(
            resume_from_checkpoint,
            model,
            tokenizer,
            optimizer,
            scheduler,
            scaler,
            device,
        )

    # Training loop
    model.train()

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                # Debug: Check for out-of-bounds indices
                if input_ids.max() >= tokenizer.vocab_size:
                    print(f"Warning: Found token ID {input_ids.max()} >= vocab_size {tokenizer.vocab_size}")
                    input_ids = torch.clamp(input_ids, 0, tokenizer.vocab_size - 1)

                # Mixed precision forward pass
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Invalid loss detected: {loss}")
                    continue

                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                progress_bar.set_postfix(
                    {
                        "loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "step": global_step,
                    }
                )

            except Exception as e:
                print(f"Error in training step {batch_idx}: {e}")
                continue

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
                model,
                tokenizer,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                global_step,
                best_val_acc,
                train_losses,
                val_accuracies,
                val_losses,
                output_dir / "checkpoints",
                is_best,
            )

    # Save final model
    final_model_dir = output_dir / "final_model"
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
            try:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    logits = outputs.logits

                total_loss += loss.item()

                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())

            except Exception as e:
                print(f"Error in evaluation: {e}")
                continue

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = accuracy_score(true_labels, predictions) if predictions else 0

    model.train()
    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser(description="Extended Longformer Vulnerability Classification with 4096 tokens")
    parser.add_argument(
        "--model-path",
        default=Path(__file__).parent / "longformer_vuln_model",
        help="Path to save/load model",
    )
    parser.add_argument(
        "--base-model",
        default="allenai/longformer-base-4096",
        help="Base Longformer model to extend",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training (small for 4096 tokens)",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--resume-from-checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
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

    val_data = test_data  # Using test as validation - consider splitting train data instead
    print("Starting training...")
    model, tokenizer = train_model(
        train_data,
        val_data,
        base_model_name=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.model_path,
        num_workers=args.num_workers,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_every_n_epochs=args.save_every_n_epochs,
        max_length=args.max_length,
    )
    print("Training completed!")


if __name__ == "__main__":
    main()
