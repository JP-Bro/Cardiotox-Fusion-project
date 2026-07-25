"""
08_train_fusion.py -- Train the GNN + Transformer + Cross-Attention Fusion model.

This is the main model of the project. Fuses both structural and biological
information via bidirectional cross-attention.

Dataset: LINCS-covered subset (~423 compounds) -- same as Transformer baseline.
The GNN also uses this subset to ensure a fair apples-to-apples comparison
(Design Decision #6).

Expected performance: hopefully > 0.84 AUC-ROC (GNN-only baseline).
Even a modest improvement is a publishable finding.
If fusion does NOT outperform structure-only, that is also a valid result
-- to be reported honestly, not suppressed.

Run with:
  python scripts/08_train_fusion.py
  python scripts/08_train_fusion.py --test-only
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

from config import CFG
from scripts.utils import get_logger, ensure_dirs, set_seed
from models.fusion_model import FusionClassifier
from models.dataset import CardiotoxFusionDataset, build_dataloaders, fusion_collate_fn
from models.trainer import Trainer, compute_metrics

LOG_FILE = os.path.join(CFG.RESULTS_DIR, "logs", "08_train_fusion.log")
logger = get_logger("08_train_fusion", LOG_FILE)


def fusion_forward_fn(model, batch, device):
    """
    Custom forward function for the Fusion model.

    Extracts graph components + gene expression from the collated batch
    and calls the FusionClassifier's forward() method.
    """
    graph = batch["graph"].to(device)
    gene_expr = batch["gene_expr"].to(device)
    labels = batch["label"].to(device)

    out = model(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        batch=graph.batch,
        gene_expr=gene_expr,
    )
    return out["logits"], labels


def load_fusion_data():
    """Load all required data for fusion training."""
    for path, desc in [
        (CFG.LINCS_MATCHED_CSV, "Matched compounds CSV (run 03_match_lincs.py)"),
        (CFG.EXPRESSION_CSV, "Expression matrix (run 03_match_lincs.py --extract)"),
        (os.path.join(CFG.PROCESSED_DIR, "graph_manifest.csv"), "Graph manifest (run 04_build_graphs.py)"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}\nRun: {desc}")

    matched_df = pd.read_csv(CFG.LINCS_MATCHED_CSV)
    expr_df = pd.read_csv(CFG.EXPRESSION_CSV, index_col=0)
    manifest = pd.read_csv(os.path.join(CFG.PROCESSED_DIR, "graph_manifest.csv"))

    matched_only = matched_df[matched_df["lincs_match"] & matched_df["sig_id"].ne("")]
    name_to_sigid = dict(zip(matched_only["query_name"], matched_only["sig_id"]))

    logger.info(f"Fusion dataset: {len(name_to_sigid)} compounds with graph + expression")
    return manifest, expr_df, name_to_sigid


def main():
    parser = argparse.ArgumentParser(description="Train GNN + Transformer + Fusion model")
    parser.add_argument("--test-only", action="store_true",
                        help="Skip training, evaluate existing checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--freeze-encoders", action="store_true",
                        help="Freeze pretrained encoder weights (train only fusion + head)")
    parser.add_argument("--gnn-ckpt", type=str, default=None,
                        help="Path to pretrained GNN encoder checkpoint (for weight init)")
    parser.add_argument("--transformer-ckpt", type=str, default=None,
                        help="Path to pretrained Transformer encoder checkpoint (for weight init)")
    args = parser.parse_args()

    if args.epochs:
        CFG.MAX_EPOCHS = args.epochs

    logger.info("=" * 60)
    logger.info("SCRIPT: 08_train_fusion.py -- GNN + Transformer + Cross-Attention Fusion")
    logger.info("=" * 60)

    set_seed(CFG.RANDOM_SEED)
    ensure_dirs(CFG.CHECKPOINTS_DIR, CFG.RESULTS_DIR, os.path.join(CFG.RESULTS_DIR, "logs"))

    manifest, expr_df, name_to_sigid = load_fusion_data()

    # Fusion dataset: includes BOTH graph and expression
    dataset = CardiotoxFusionDataset(
        manifest_df=manifest,
        expr_df=expr_df,
        name_to_sigid=name_to_sigid,
        include_graph=True,
    )
    labels = dataset.labels
    logger.info(f"Fusion dataset: {len(dataset)} compounds")
    logger.info(f"Label balance: {labels.sum()} positive ({labels.mean():.1%})")

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset, labels,
        batch_size=CFG.BATCH_SIZE,
        use_weighted_sampler=True,
        collate_fn=fusion_collate_fn,
    )

    # Build fusion model
    model = FusionClassifier(
        gnn_kwargs={
            "node_feat_dim": CFG.NODE_FEATURE_DIM,
            "edge_feat_dim": CFG.EDGE_FEATURE_DIM,
            "hidden_dim": CFG.GNN_HIDDEN_DIM,
            "embed_dim": CFG.EMBED_DIM,
            "num_layers": CFG.GNN_NUM_LAYERS,
            "dropout": CFG.GNN_DROPOUT,
        },
        transformer_kwargs={
            "n_genes": CFG.NUM_GENES,
            "d_model": CFG.TRANS_D_MODEL,
            "nhead": CFG.TRANS_NHEAD,
            "num_layers": CFG.TRANS_NUM_LAYERS,
            "dim_feedforward": CFG.TRANS_DIM_FF,
            "dropout": CFG.TRANS_DROPOUT,
            "embed_dim": CFG.EMBED_DIM,
        },
        fusion_embed_dim=CFG.EMBED_DIM,
        nhead=CFG.FUSION_NHEAD,
        dropout=CFG.FUSION_DROPOUT,
    )

    param_counts = model.count_parameters()
    logger.info("FusionClassifier parameter counts:")
    for k, v in param_counts.items():
        logger.info(f"  {k}: {v:,}")

    # Optional: initialize encoders from pretrained checkpoints
    if args.gnn_ckpt and os.path.isfile(args.gnn_ckpt):
        ckpt = torch.load(args.gnn_ckpt, map_location="cpu")
        # Load only encoder weights (not classifier head)
        enc_state = {k.replace("encoder.", ""): v
                     for k, v in ckpt["model_state_dict"].items()
                     if k.startswith("encoder.")}
        model.gnn_encoder.load_state_dict(enc_state, strict=False)
        logger.info(f"Initialized GNN encoder from: {args.gnn_ckpt}")

    if args.transformer_ckpt and os.path.isfile(args.transformer_ckpt):
        ckpt = torch.load(args.transformer_ckpt, map_location="cpu")
        enc_state = {k.replace("encoder.", ""): v
                     for k, v in ckpt["model_state_dict"].items()
                     if k.startswith("encoder.")}
        model.transformer_encoder.load_state_dict(enc_state, strict=False)
        logger.info(f"Initialized Transformer encoder from: {args.transformer_ckpt}")

    # Optional: freeze encoders (train only fusion + head)
    if args.freeze_encoders:
        for param in model.gnn_encoder.parameters():
            param.requires_grad = False
        for param in model.transformer_encoder.parameters():
            param.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Encoders frozen. Trainable parameters: {n_trainable:,}")

    trainer = Trainer(
        model=model,
        checkpoint_path=CFG.FUSION_CHECKPOINT,
        forward_fn=fusion_forward_fn,
        pos_weight=1.0,  # Avoid double-balancing with WeightedRandomSampler
    )

    if not args.test_only:
        logger.info("Starting Fusion model training...")
        train_labels = train_loader.dataset.labels
        history = trainer.train(train_loader, val_loader, train_labels)

        history_path = os.path.join(CFG.RESULTS_DIR, "fusion_training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to: {history_path}")

    if os.path.isfile(CFG.FUSION_CHECKPOINT):
        ckpt = torch.load(CFG.FUSION_CHECKPOINT, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint (epoch {ckpt['epoch']}, val AUC={ckpt['val_auc']:.4f})")

    trainer_eval = Trainer(model=model, checkpoint_path=CFG.FUSION_CHECKPOINT,
                           forward_fn=fusion_forward_fn)
    test_metrics, test_loss = trainer_eval.evaluate(test_loader)

    logger.info("=" * 60)
    logger.info("FUSION MODEL -- TEST SET RESULTS")
    logger.info(f"  AUC-ROC  : {test_metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR   : {test_metrics['auc_pr']:.4f}")
    logger.info(f"  F1 Score : {test_metrics['f1']:.4f}")
    logger.info(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test Loss: {test_loss:.4f}")
    logger.info("=" * 60)

    results = {
        "model": "Fusion (GNN + Transformer + CrossAttention)",
        "dataset": "lincs_covered_subset",
        "test_split": "stratified_0.15",
        "seal_2023_structure_baseline": 0.84,
        "seal_2023_biology_baseline": 0.76,
        **test_metrics,
    }
    results_path = os.path.join(CFG.RESULTS_DIR, "fusion_test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Test results saved to: {results_path}")
    logger.info("08_train_fusion.py COMPLETE")


if __name__ == "__main__":
    main()
