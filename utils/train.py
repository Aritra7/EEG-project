"""
Training Loop
==============
Supports both BaselineMLP and EEGCNNTransformer with:
- Selective backpropagation (only subject-specific head gradients)
- Logging, checkpointing, early stopping
- Validation monitoring
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import numpy as np


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
        elif val_metric < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_metric
            self.counter = 0


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    log_interval: int = 50,
) -> Dict[str, float]:
    """Train for one epoch. Returns dict of metrics."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        eeg = batch["eeg"].to(device)
        labels = batch["label"].to(device)
        subject_idx = batch["subject_idx"].to(device)

        optimizer.zero_grad()
        logits = model(eeg, subject_idx)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * eeg.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += eeg.size(0)

        if (batch_idx + 1) % log_interval == 0:
            running_acc = correct / total * 100
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {running_acc:.2f}%")

    return {
        "loss": total_loss / total,
        "accuracy": correct / total * 100,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_subjects: int = 13,
) -> Dict[str, float]:
    """Evaluate model. Returns metrics including per-subject accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Per-subject tracking
    subject_correct = np.zeros(num_subjects)
    subject_total = np.zeros(num_subjects)

    # For confusion matrix
    all_preds = []
    all_labels = []

    for batch in loader:
        eeg = batch["eeg"].to(device)
        labels = batch["label"].to(device)
        subject_idx = batch["subject_idx"].to(device)

        logits = model(eeg, subject_idx)
        loss = criterion(logits, labels)

        total_loss += loss.item() * eeg.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += eeg.size(0)

        # Per-subject
        for sid in range(num_subjects):
            mask = subject_idx == sid
            if mask.any():
                subject_correct[sid] += (preds[mask] == labels[mask]).sum().item()
                subject_total[sid] += mask.sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    per_subject_acc = {}
    for sid in range(num_subjects):
        if subject_total[sid] > 0:
            per_subject_acc[sid] = subject_correct[sid] / subject_total[sid] * 100

    return {
        "loss": total_loss / total,
        "accuracy": correct / total * 100,
        "per_subject_acc": per_subject_acc,
        "all_preds": np.array(all_preds),
        "all_labels": np.array(all_labels),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    save_dir: str = "./checkpoints",
    patience: int = 10,
    scheduler_type: str = "cosine",
    log_interval: int = 50,
    model_name: str = "model",
):
    """Full training loop with validation, scheduling, and early stopping."""

    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    early_stopping = EarlyStopping(patience=patience)
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"Training {model_name} for {num_epochs} epochs")
    print(f"Device: {device} | LR: {lr} | Batch size: {train_loader.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, log_interval)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        if scheduler_type == "plateau":
            scheduler.step(val_metrics["accuracy"])
        else:
            scheduler.step()

        # Logging
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{num_epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.2f}% | "
              f"LR: {current_lr:.2e}")

        # Save history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "history": history,
            }, os.path.join(save_dir, f"{model_name}_best.pt"))
            print(f"  ✓ New best val acc: {best_val_acc:.2f}%")

        # Early stopping
        early_stopping(val_metrics["accuracy"])
        if early_stopping.should_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")
    return history
