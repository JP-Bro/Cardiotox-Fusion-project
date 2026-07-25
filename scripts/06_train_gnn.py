"""
06_train_gnn.py -- Train the structure-only GNN baseline model.

This is Baseline 1: uses molecular structure (SMILES => graph) only.
No LINCS gene expression data.
Trains on the FULL 1,211-compound labeled set (not just LINCS-covered).

IMPORTANT for fair comparison (Design Decision #6):
  This model trains on all 1,211 compounds but MUST be evaluated on
  the same LINCS-covered held-out test set as the other two models.
  Comparing across different test sets would not be an honest comparison.

Expected performance (from Seal et al. 2023): ~0.84 AUC-ROC.
This is the hardest baseline to beat -- structure is the stronger signal.

Run with:
  python scripts/06_train_gnn.py
  python scripts/06_train_gnn.py --test-only   # evaluate existing checkpoint
"""

import os
import sys
import argparse
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from config import CFG
from scripts.utils import get_logger, ensure_dirs, set_seed
from models.gnn_model import GNNClassifier
from models.dataset import CardiotoxGraphDataset, build_dataloaders, graph_collate_fn
from models.trainer import Trainer, compute_metrics

LOG_FILE = os.path.join(CFG.RESULTS_DIR, "logs", "06_train_gnn.log")
logger = get_logger("06_train_gnn", LOG_FILE)


def gnn_forward_fn(model, batch, device):
    """Custom forward function for GNN model with PyG Data batch."""
    batch = batch.to(device)
    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    return logits, batch.y


def main():
    parser = argparse.ArgumentParser(description="Train GNN structure-only baseline")
    parser.add_argument("--test-only", action="store_true",
                        help="Skip training, evaluate existing checkpoint on test set")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Override max epochs (default: {CFG.MAX_EPOCHS})")
    args = parser.parse_args()

    if args.epochs:
        CFG.MAX_EPOCHS = args.epochs

    logger.info("=" * 60)
    logger.info("SCRIPT: 06_train_gnn.py -- Structure-Only GNN Baseline")
    logger.info("=" * 60)

    set_seed(CFG.RANDOM_SEED)
    ensure_dirs(CFG.CHECKPOINTS_DIR, CFG.RESULTS_DIR, os.path.join(CFG.RESULTS_DIR, "logs"))

    # Load graph manifest
    manifest_path = os.path.join(CFG.PROCESSED_DIR, "graph_manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Graph manifest not found at {manifest_path}. Run 04_build_graphs.py first."
        )
    manifest = pd.read_csv(manifest_path)
    logger.info(f"Graph manifest: {len(manifest)} entries, {(manifest['status'] != 'failed').sum()} valid")

    # Build dataset
    dataset = CardiotoxGraphDataset(manifest)
    labels = dataset.labels

    logger.info(f"Dataset: {len(dataset)} compounds")
    logger.info(f"Label balance: {labels.sum()} positive ({labels.mean():.1%})")

    # Build data loaders (stratified split, PyG collate)
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset, labels,
        batch_size=CFG.BATCH_SIZE,
        use_weighted_sampler=True,
        collate_fn=graph_collate_fn,
    )

    # Build model
    model = GNNClassifier(
        node_feat_dim=CFG.NODE_FEATURE_DIM,
        edge_feat_dim=CFG.EDGE_FEATURE_DIM,
        hidden_dim=CFG.GNN_HIDDEN_DIM,
        embed_dim=CFG.EMBED_DIM,
        num_layers=CFG.GNN_NUM_LAYERS,
        dropout=CFG.GNN_DROPOUT,
    )
    logger.info(f"GNNClassifier parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        checkpoint_path=CFG.GNN_CHECKPOINT,
        forward_fn=gnn_forward_fn,
        pos_weight=1.0,  # Avoid double-balancing with WeightedRandomSampler
    )

    if not args.test_only:
        # Train
        logger.info("Starting GNN training...")
        train_labels = train_loader.dataset.labels
        history = trainer.train(train_loader, val_loader, train_labels)

        # Save history
        history_path = os.path.join(CFG.RESULTS_DIR, "gnn_training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to: {history_path}")

    # Load best checkpoint and evaluate on test set
    if os.path.isfile(CFG.GNN_CHECKPOINT):
        ckpt = torch.load(CFG.GNN_CHECKPOINT, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint (epoch {ckpt['epoch']}, val AUC={ckpt['val_auc']:.4f})")

    trainer_eval = Trainer(model=model, checkpoint_path=CFG.GNN_CHECKPOINT, forward_fn=gnn_forward_fn)
    test_metrics, test_loss = trainer_eval.evaluate(test_loader)

    logger.info("=" * 60)
    logger.info("GNN BASELINE -- TEST SET RESULTS")
    logger.info(f"  AUC-ROC  : {test_metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR   : {test_metrics['auc_pr']:.4f}")
    logger.info(f"  F1 Score : {test_metrics['f1']:.4f}")
    logger.info(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test Loss: {test_loss:.4f}")
    logger.info(f"  TP/TN/FP/FN: {test_metrics.get('tp')}/{test_metrics.get('tn')}/{test_metrics.get('fp')}/{test_metrics.get('fn')}")
    logger.info("=" * 60)

    # Save results
    results = {
        "model": "GNN (structure-only)",
        "dataset": "full_labeled_set",
        "test_split": "stratified_0.15",
        **test_metrics,
    }
    results_path = os.path.join(CFG.RESULTS_DIR, "gnn_test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Test results saved to: {results_path}")
    logger.info("06_train_gnn.py COMPLETE")


if __name__ == "__main__":
    main()
