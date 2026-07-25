"""
07_train_transformer.py -- Train the biology-only Transformer baseline model.

This is Baseline 2: uses LINCS L1000 gene expression profiles only.
No molecular structure data.
Trains ONLY on the LINCS-covered subset (~423 compounds after matching).

Expected performance (from Seal et al. 2023): ~0.76 AUC-ROC for biology-only.
This is the weaker baseline -- structure consistently outperforms in prior work.

Run with:
  python scripts/07_train_transformer.py
  python scripts/07_train_transformer.py --test-only
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
from torch.utils.data import DataLoader

from config import CFG
from scripts.utils import get_logger, ensure_dirs, set_seed
from models.transformer_model import TransformerClassifier
from models.dataset import CardiotoxFusionDataset, build_dataloaders, fusion_collate_fn
from models.trainer import Trainer, compute_metrics

LOG_FILE = os.path.join(CFG.RESULTS_DIR, "logs", "07_train_transformer.log")
logger = get_logger("07_train_transformer", LOG_FILE)


def transformer_forward_fn(model, batch, device):
    """Custom forward function for Transformer model with fusion collate batch."""
    gene_expr = batch["gene_expr"].to(device)
    labels = batch["label"].to(device)
    logits = model(gene_expr)
    return logits, labels


def load_expression_data():
    """
    Load LINCS expression matrix and matched compound manifest.

    Returns:
        (manifest_df, expr_df, name_to_sigid)
    """
    if not os.path.isfile(CFG.LINCS_MATCHED_CSV):
        raise FileNotFoundError(
            f"Matched compounds CSV not found at {CFG.LINCS_MATCHED_CSV}. "
            "Run 03_match_lincs.py first."
        )
    if not os.path.isfile(CFG.EXPRESSION_CSV):
        raise FileNotFoundError(
            f"Expression matrix not found at {CFG.EXPRESSION_CSV}. "
            "Run 03_match_lincs.py --extract first (requires GCTX download)."
        )
    if not os.path.isfile(os.path.join(CFG.PROCESSED_DIR, "graph_manifest.csv")):
        raise FileNotFoundError(
            "Graph manifest not found. Run 04_build_graphs.py first."
        )

    matched_df = pd.read_csv(CFG.LINCS_MATCHED_CSV)
    expr_df = pd.read_csv(CFG.EXPRESSION_CSV, index_col=0)
    manifest = pd.read_csv(os.path.join(CFG.PROCESSED_DIR, "graph_manifest.csv"))

    logger.info(f"Matched compounds: {len(matched_df)}, LINCS-matched: {matched_df['lincs_match'].sum()}")
    logger.info(f"Expression matrix: {expr_df.shape}")

    # Build name->sig_id mapping
    matched_only = matched_df[matched_df["lincs_match"] & matched_df["sig_id"].ne("")]
    name_to_sigid = dict(zip(matched_only["query_name"], matched_only["sig_id"]))
    logger.info(f"Compounds with expression data: {len(name_to_sigid)}")

    return manifest, expr_df, name_to_sigid


def main():
    parser = argparse.ArgumentParser(description="Train Transformer biology-only baseline")
    parser.add_argument("--test-only", action="store_true",
                        help="Skip training, evaluate existing checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.epochs:
        CFG.MAX_EPOCHS = args.epochs

    logger.info("=" * 60)
    logger.info("SCRIPT: 07_train_transformer.py -- Biology-Only Transformer Baseline")
    logger.info("=" * 60)

    set_seed(CFG.RANDOM_SEED)
    ensure_dirs(CFG.CHECKPOINTS_DIR, CFG.RESULTS_DIR, os.path.join(CFG.RESULTS_DIR, "logs"))

    manifest, expr_df, name_to_sigid = load_expression_data()

    # Transformer-only: no graph needed (include_graph=False)
    dataset = CardiotoxFusionDataset(
        manifest_df=manifest,
        expr_df=expr_df,
        name_to_sigid=name_to_sigid,
        include_graph=False,
    )
    labels = dataset.labels
    logger.info(f"Dataset: {len(dataset)} compounds with expression data")
    logger.info(f"Label balance: {labels.sum()} positive ({labels.mean():.1%})")

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset, labels,
        batch_size=CFG.BATCH_SIZE,
        use_weighted_sampler=True,
        collate_fn=fusion_collate_fn,
    )

    model = TransformerClassifier(
        n_genes=CFG.NUM_GENES,
        d_model=CFG.TRANS_D_MODEL,
        nhead=CFG.TRANS_NHEAD,
        num_layers=CFG.TRANS_NUM_LAYERS,
        dim_feedforward=CFG.TRANS_DIM_FF,
        dropout=CFG.TRANS_DROPOUT,
        embed_dim=CFG.EMBED_DIM,
    )
    logger.info(f"TransformerClassifier parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        checkpoint_path=CFG.TRANSFORMER_CKPT,
        forward_fn=transformer_forward_fn,
        pos_weight=1.0,  # Avoid double-balancing with WeightedRandomSampler
    )

    if not args.test_only:
        logger.info("Starting Transformer training...")
        train_labels = train_loader.dataset.labels
        history = trainer.train(train_loader, val_loader, train_labels)

        history_path = os.path.join(CFG.RESULTS_DIR, "transformer_training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to: {history_path}")

    if os.path.isfile(CFG.TRANSFORMER_CKPT):
        ckpt = torch.load(CFG.TRANSFORMER_CKPT, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint (epoch {ckpt['epoch']}, val AUC={ckpt['val_auc']:.4f})")

    trainer_eval = Trainer(model=model, checkpoint_path=CFG.TRANSFORMER_CKPT,
                           forward_fn=transformer_forward_fn)
    test_metrics, test_loss = trainer_eval.evaluate(test_loader)

    logger.info("=" * 60)
    logger.info("TRANSFORMER BASELINE -- TEST SET RESULTS")
    logger.info(f"  AUC-ROC  : {test_metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR   : {test_metrics['auc_pr']:.4f}")
    logger.info(f"  F1 Score : {test_metrics['f1']:.4f}")
    logger.info(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test Loss: {test_loss:.4f}")
    logger.info(f"  NOTE: Biology-only ceiling from Seal et al. 2023 = ~0.76 AUC-ROC")
    logger.info(f"        None of the LINCS cell lines are cardiomyocytes -- see PROJECT_DOCUMENTATION.md Risk R1")
    logger.info("=" * 60)

    results = {
        "model": "Transformer (biology-only)",
        "dataset": "lincs_covered_subset",
        "test_split": "stratified_0.15",
        "seal_2023_biology_baseline": 0.76,
        **test_metrics,
    }
    results_path = os.path.join(CFG.RESULTS_DIR, "transformer_test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Test results saved to: {results_path}")
    logger.info("07_train_transformer.py COMPLETE")


if __name__ == "__main__":
    main()
