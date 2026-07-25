"""
trainer.py -- Shared training loop for all three Cardiotox-Fusion models.

Contains:
  - Trainer class: generic training loop (works for GNN, Transformer, and Fusion)
  - train_epoch(): single epoch of training
  - evaluate(): evaluate on a DataLoader and return metrics
  - EarlyStopping: early stop on val-AUC with patience
  - compute_metrics(): AUC-ROC, AUC-PR, F1, accuracy from logits

Design choices:
  - BCEWithLogitsLoss with class weighting (handles imbalance)
  - AdamW optimizer (better L2 regularization than Adam)
  - Cosine LR schedule with linear warmup
  - Early stopping on val-AUC (not val-loss, which can be misleading with imbalance)
  - Gradient clipping (max_norm=1.0) for training stability
  - Best model saved by val-AUC (not last epoch)
"""

import os
import sys
import time
from typing import Callable, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, confusion_matrix
)

from config import CFG
from scripts.utils import get_logger, compute_class_weight

logger = get_logger("trainer")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute all classification metrics from raw logits.

    Args:
        labels:    Ground-truth binary labels (0/1) -- shape (N,)
        logits:    Raw model logits (before sigmoid) -- shape (N,)
        threshold: Decision threshold for binary predictions (default 0.5)

    Returns:
        dict with keys: auc_roc, auc_pr, f1, accuracy, tp, tn, fp, fn
    """
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= threshold).astype(int)

    metrics = {}
    try:
        metrics["auc_roc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc_roc"] = float("nan")

    try:
        metrics["auc_pr"] = average_precision_score(labels, probs)
    except ValueError:
        metrics["auc_pr"] = float("nan")

    metrics["f1"] = f1_score(labels, preds, zero_division=0)
    metrics["accuracy"] = accuracy_score(labels, preds)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})

    return metrics


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when val-AUC stops improving.

    Args:
        patience:  Epochs to wait after last improvement (default: CFG.EARLY_STOP_PATIENCE)
        min_delta: Minimum improvement to count as improvement (default: 0.001)
        mode:      'max' (for AUC) or 'min' (for loss)
    """

    def __init__(
        self,
        patience: int = CFG.EARLY_STOP_PATIENCE,
        min_delta: float = 0.001,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.

        Returns:
            True if training should stop, False otherwise.
        """
        improved = (
            value > self.best_value + self.min_delta if self.mode == "max"
            else value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Generic training loop for all three Cardiotox-Fusion models.

    Handles:
      - BCEWithLogitsLoss with positive class weighting
      - AdamW + cosine LR schedule with linear warmup
      - Gradient clipping
      - Early stopping on val-AUC
      - Checkpoint saving (best val-AUC model)
      - Full training history logging

    Args:
        model:            PyTorch model with forward() returning logits or dict
        checkpoint_path:  Path to save best model checkpoint
        device:           'cuda' or 'cpu' (auto-detected if None)
        pos_weight:       Positive class weight for BCEWithLogitsLoss
                          (auto-computed from training labels if None)
        forward_fn:       Optional custom forward function. If None, uses
                          standard forward(batch) returning logits or dict['logits'].
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: str,
        device: Optional[str] = None,
        pos_weight: Optional[float] = None,
        forward_fn: Optional[Callable] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.checkpoint_path = checkpoint_path
        self.pos_weight = pos_weight
        self.forward_fn = forward_fn
        self.history = {"train_loss": [], "val_auc": [], "val_loss": [], "lr": []}

        logger.info(f"Trainer initialized. Device: {self.device}")
        if hasattr(model, "count_parameters"):
            params = model.count_parameters()
            logger.info(f"Model parameters: {params}")
        else:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Trainable parameters: {n_params:,}")

    def _build_optimizer_and_scheduler(self, n_train_steps: int):
        """Build AdamW optimizer + cosine schedule with linear warmup."""
        optimizer = AdamW(
            self.model.parameters(),
            lr=CFG.LEARNING_RATE,
            weight_decay=CFG.WEIGHT_DECAY,
        )

        warmup_steps = CFG.LR_WARMUP_EPOCHS * n_train_steps
        cosine_steps = (CFG.MAX_EPOCHS - CFG.LR_WARMUP_EPOCHS) * n_train_steps

        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        return optimizer, scheduler

    def _run_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a single batch through the model.

        Returns:
            (logits, labels) -- both shape (N,) on the current device
        """
        if self.forward_fn is not None:
            logits, labels = self.forward_fn(self.model, batch, self.device)
        else:
            # Generic forward for simple models
            batch = batch.to(self.device)
            out = self.model(batch)
            logits = out["logits"] if isinstance(out, dict) else out
            labels = batch.y

        return logits.squeeze(-1), labels.squeeze(-1)

    def train(
        self,
        train_loader,
        val_loader,
        train_labels: np.ndarray,
    ) -> dict:
        """
        Full training loop.

        Args:
            train_loader:  DataLoader for training set
            val_loader:    DataLoader for validation set
            train_labels:  All training labels (for computing pos_weight if not set)

        Returns:
            dict: Training history (losses, AUCs per epoch)
        """
        # Auto-compute positive class weight
        pos_weight_val = self.pos_weight or compute_class_weight(train_labels)
        pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        logger.info(f"BCEWithLogitsLoss pos_weight: {pos_weight_val:.3f}")

        optimizer, scheduler = self._build_optimizer_and_scheduler(len(train_loader))
        early_stopper = EarlyStopping()

        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        best_val_auc = -1.0
        start_time = time.time()

        for epoch in range(1, CFG.MAX_EPOCHS + 1):
            # --- Training epoch ---
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()
                logits, labels = self._run_batch(batch)
                loss = criterion(logits, labels.to(self.device))
                loss.backward()

                # Gradient clipping (prevents exploding gradients)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # --- Validation ---
            val_metrics, val_loss = self.evaluate(val_loader, criterion)
            val_auc = val_metrics.get("auc_roc", 0.0)
            current_lr = scheduler.get_last_lr()[0]

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_auc"].append(val_auc)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_auc": val_auc,
                    "val_metrics": val_metrics,
                    "history": self.history,
                }, self.checkpoint_path)

            # Log every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {epoch:3d}/{CFG.MAX_EPOCHS} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val AUC: {val_auc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Best AUC: {best_val_auc:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.0f}s"
                )

            # Early stopping check
            if early_stopper(val_auc):
                logger.info(f"Early stopping at epoch {epoch} (best val AUC: {best_val_auc:.4f})")
                break

        logger.info(f"Training complete. Best val AUC: {best_val_auc:.4f}")
        logger.info(f"Best checkpoint: {self.checkpoint_path}")
        return self.history

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        criterion: Optional[nn.Module] = None,
    ) -> tuple[dict, float]:
        """
        Evaluate model on a DataLoader.

        Args:
            loader:    DataLoader to evaluate on
            criterion: Optional loss function (for val_loss computation)

        Returns:
            (metrics_dict, avg_loss)
        """
        self.model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            logits, labels = self._run_batch(batch)
            labels_device = labels.to(self.device)

            if criterion is not None:
                loss = criterion(logits, labels_device)
                total_loss += loss.item()

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            n_batches += 1

        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        avg_loss = total_loss / max(n_batches, 1)
        metrics = compute_metrics(all_labels, all_logits)
        return metrics, avg_loss
