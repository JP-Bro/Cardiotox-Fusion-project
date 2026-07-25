"""
09_evaluate.py -- Evaluate all three models on the SAME held-out test set.

CRITICAL: All three models MUST be evaluated on the same LINCS-covered test set
for the comparison to be fair and honest (Design Decision #6 from trace-through).

Why this matters:
  - GNN-only can train on all 1,211 compounds
  - Transformer-only and Fusion can only train on ~423 LINCS-covered compounds
  - If GNN-only is also evaluated on all 1,211 but others on ~423, the comparison
    is misleading -- the GNN has a larger test set and potentially different difficulty
  - ALL three must be evaluated on the LINCS-covered subset's test split

What this script does:
  1. Loads all three model checkpoints
  2. Loads the SAME LINCS-covered test split used for Transformer and Fusion
  3. Evaluates GNN, Transformer, and Fusion on this shared test set
  4. Generates side-by-side comparison tables with confidence intervals
  5. Outputs results/baseline_comparison.md (ready for paper write-up)

Run with:
  python scripts/09_evaluate.py
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/headless runs
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.utils import resample

from config import CFG
from scripts.utils import get_logger, ensure_dirs, set_seed
from models.gnn_model import GNNClassifier
from models.transformer_model import TransformerClassifier
from models.fusion_model import FusionClassifier
from models.dataset import CardiotoxFusionDataset, CardiotoxGraphDataset, stratified_split, fusion_collate_fn, graph_collate_fn
from models.trainer import Trainer, compute_metrics

LOG_FILE = os.path.join(CFG.RESULTS_DIR, "logs", "09_evaluate.log")
logger = get_logger("09_evaluate", LOG_FILE)


def bootstrap_auc(labels: np.ndarray, probs: np.ndarray,
                  n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """
    Compute bootstrap confidence interval for AUC-ROC.

    Args:
        labels:      True binary labels
        probs:       Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        ci:          Confidence interval level (default 0.95)

    Returns:
        (mean_auc, lower_bound, upper_bound)
    """
    from sklearn.metrics import roc_auc_score
    boot_aucs = []
    rng = np.random.RandomState(CFG.RANDOM_SEED)

    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(labels), len(labels))
        boot_labels = labels[idx]
        boot_probs = probs[idx]
        if len(np.unique(boot_labels)) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(boot_labels, boot_probs))
        except Exception:
            continue

    if not boot_aucs:
        return float("nan"), float("nan"), float("nan")

    alpha = 1 - ci
    lower = np.percentile(boot_aucs, 100 * alpha / 2)
    upper = np.percentile(boot_aucs, 100 * (1 - alpha / 2))
    return np.mean(boot_aucs), lower, upper


def plot_roc_curves(all_results: dict, output_path: str):
    """Plot ROC curves for all three models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"GNN (structure-only)": "#e74c3c",
               "Transformer (biology-only)": "#3498db",
               "Fusion (GNN+Transformer)": "#2ecc71"}

    ax1, ax2 = axes

    for model_name, result in all_results.items():
        labels = result["labels"]
        probs = result["probs"]
        color = colors.get(model_name, "gray")

        # ROC curve
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=color, lw=2,
                 label=f"{model_name} (AUC={roc_auc:.3f})")

        # PR curve
        precision, recall, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, color=color, lw=2,
                 label=f"{model_name} (AUC={pr_auc:.3f})")

    # ROC plot formatting
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("ROC Curves -- All Models\n(same LINCS-covered test set)", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # PR plot formatting
    ax2.axhline(y=sum(all_results[list(all_results.keys())[0]]["labels"]) /
                len(all_results[list(all_results.keys())[0]]["labels"]),
                color="k", linestyle="--", lw=1, label="Random baseline")
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("Precision-Recall Curves -- All Models", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC/PR curves saved to: {output_path}")


def generate_comparison_report(all_results: dict, all_metrics: dict) -> str:
    """Generate the baseline_comparison.md markdown report."""
    lines = [
        "# Cardiotox-Fusion: Baseline Comparison Report",
        "",
        "> **CRITICAL NOTE**: All three models evaluated on the **same LINCS-covered test set**.",
        "> (Design Decision #6 from trace-through -- fair comparison requirement)",
        "",
        "## Test Set Information",
        "",
    ]

    # Get test set info from first result
    first = list(all_results.values())[0]
    n_test = len(first["labels"])
    n_pos = int(first["labels"].sum())
    lines += [
        f"| Metric | Value |",
        f"|---|---|",
        f"| Test set size | {n_test} compounds |",
        f"| Positive (cardiotoxic) | {n_pos} ({n_pos/n_test:.1%}) |",
        f"| Negative (no concern) | {n_test - n_pos} ({(n_test-n_pos)/n_test:.1%}) |",
        f"| Split | Stratified 70/15/15 (seed={CFG.RANDOM_SEED}) |",
        "",
        "## Model Performance",
        "",
        "| Model | AUC-ROC (95% CI) | AUC-PR | F1 | Accuracy |",
        "|---|---|---|---|---|",
    ]

    for model_name, metrics in all_metrics.items():
        auc_roc = metrics.get("auc_roc", float("nan"))
        auc_pr = metrics.get("auc_pr", float("nan"))
        f1 = metrics.get("f1", float("nan"))
        acc = metrics.get("accuracy", float("nan"))
        ci_low = metrics.get("ci_low", float("nan"))
        ci_high = metrics.get("ci_high", float("nan"))
        ci_str = f"({ci_low:.3f}-{ci_high:.3f})" if not (np.isnan(ci_low) or np.isnan(ci_high)) else ""
        lines.append(f"| {model_name} | {auc_roc:.4f} {ci_str} | {auc_pr:.4f} | {f1:.4f} | {acc:.4f} |")

    lines += [
        "",
        "## Prior Work Reference",
        "",
        "| Model | AUC-ROC | Source |",
        "|---|---|---|",
        "| Structure-only (chemical fingerprints) | 0.84 | Seal et al. 2023 |",
        "| Biology-only (LINCS => GO features) | 0.76 | Seal et al. 2023 |",
        "| Random baseline | 0.50 | -- |",
        "",
        "## Key Limitations (Must Be Stated in Paper)",
        "",
        "1. **LINCS cell lines are not cardiomyocytes** -- the biology branch learns",
        "   a general cross-tissue transcriptional toxicity signature, not cardiac-specific.",
        "   This is consistent with Seal et al.'s approach and their ~0.76 AUC-ROC ceiling.",
        "",
        "2. **Dataset size** -- the fused model trains on ~423-450 compounds (LINCS-covered",
        "   subset), vs. 1,211 for the structure-only baseline trained independently.",
        "   Both are evaluated on the same test set here for fairness.",
        "",
        "3. **Coverage imbalance** -- LINCS covers 37.7% of positive-label compounds",
        "   vs. 28.0% of negative-label compounds. Stratified splits used throughout.",
        "",
        "4. **Attention ≠ explanation** -- GNN and cross-attention weights are not",
        "   validated explanations until cross-checked against known toxicophores.",
        "   See results/interpretability_validation.md.",
    ]

    return "\n".join(lines)


def main():
    logger.info("=" * 60)
    logger.info("SCRIPT: 09_evaluate.py -- Fair Comparison on Same Test Set")
    logger.info("=" * 60)

    set_seed(CFG.RANDOM_SEED)
    ensure_dirs(CFG.RESULTS_DIR, os.path.join(CFG.RESULTS_DIR, "logs"))

    # Load shared dataset (LINCS-covered only)
    if not os.path.isfile(CFG.LINCS_MATCHED_CSV):
        raise FileNotFoundError("Run 03_match_lincs.py first.")
    if not os.path.isfile(CFG.EXPRESSION_CSV):
        raise FileNotFoundError("Run 03_match_lincs.py --extract first.")
    if not os.path.isfile(os.path.join(CFG.PROCESSED_DIR, "graph_manifest.csv")):
        raise FileNotFoundError("Run 04_build_graphs.py first.")

    matched_df = pd.read_csv(CFG.LINCS_MATCHED_CSV)
    expr_df = pd.read_csv(CFG.EXPRESSION_CSV, index_col=0)
    manifest = pd.read_csv(os.path.join(CFG.PROCESSED_DIR, "graph_manifest.csv"))
    matched_only = matched_df[matched_df["lincs_match"] & matched_df["sig_id"].ne("")]
    name_to_sigid = dict(zip(matched_only["query_name"], matched_only["sig_id"]))

    # Build the SAME fusion dataset as used in training (same split)
    fusion_dataset = CardiotoxFusionDataset(
        manifest_df=manifest, expr_df=expr_df,
        name_to_sigid=name_to_sigid, include_graph=True,
    )
    labels = fusion_dataset.labels
    _, _, test_dataset = stratified_split(fusion_dataset, labels)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
        collate_fn=fusion_collate_fn,
    )

    all_results = {}
    all_metrics = {}

    # --- Evaluate each model ---
    model_configs = [
        {
            "name": "GNN (structure-only)",
            "ckpt": CFG.GNN_CHECKPOINT,
            "model": GNNClassifier(),
            "forward_fn": lambda m, b, d: (
                m(b["graph"].to(d).x,
                  b["graph"].to(d).edge_index,
                  b["graph"].to(d).edge_attr,
                  b["graph"].to(d).batch),
                b["label"].to(d),
            ),
        },
        {
            "name": "Transformer (biology-only)",
            "ckpt": CFG.TRANSFORMER_CKPT,
            "model": TransformerClassifier(),
            "forward_fn": lambda m, b, d: (
                m(b["gene_expr"].to(d)),
                b["label"].to(d),
            ),
        },
        {
            "name": "Fusion (GNN+Transformer)",
            "ckpt": CFG.FUSION_CHECKPOINT,
            "model": FusionClassifier(),
            "forward_fn": lambda m, b, d: (
                m(x=b["graph"].to(d).x,
                  edge_index=b["graph"].to(d).edge_index,
                  edge_attr=b["graph"].to(d).edge_attr,
                  batch=b["graph"].to(d).batch,
                  gene_expr=b["gene_expr"].to(d))["logits"],
                b["label"].to(d),
            ),
        },
    ]

    for config in model_configs:
        if not os.path.isfile(config["ckpt"]):
            logger.warning(f"Checkpoint not found for {config['name']}: {config['ckpt']} -- skipping")
            continue

        model = config["model"]
        ckpt = torch.load(config["ckpt"], map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        trainer = Trainer(model=model, checkpoint_path=config["ckpt"],
                          forward_fn=config["forward_fn"])
        test_metrics, _ = trainer.evaluate(test_loader)

        # Collect raw logits for CI computation and plotting
        all_logits, all_labels = [], []
        device = next(model.parameters()).device
        with torch.no_grad():
            for batch in test_loader:
                logits, lbl = config["forward_fn"](model, batch, device)
                # Squeeze handling for single-item batches
                l_np = logits.squeeze().cpu().numpy()
                lbl_np = lbl.squeeze().cpu().numpy()
                if l_np.ndim == 0:
                    l_np = np.expand_dims(l_np, 0)
                if lbl_np.ndim == 0:
                    lbl_np = np.expand_dims(lbl_np, 0)
                all_logits.append(l_np)
                all_labels.append(lbl_np)

        logits_arr = np.concatenate(all_logits)
        labels_arr = np.concatenate(all_labels)
        probs_arr = 1 / (1 + np.exp(-logits_arr))

        mean_auc, ci_low, ci_high = bootstrap_auc(labels_arr, probs_arr)
        test_metrics["ci_low"] = ci_low
        test_metrics["ci_high"] = ci_high

        all_results[config["name"]] = {"labels": labels_arr, "probs": probs_arr}
        all_metrics[config["name"]] = test_metrics

        logger.info(f"{config['name']}: AUC-ROC={test_metrics['auc_roc']:.4f} "
                    f"(95% CI: {ci_low:.4f}-{ci_high:.4f})")

    # Plot ROC/PR curves
    if all_results:
        plot_roc_curves(all_results, os.path.join(CFG.RESULTS_DIR, "roc_pr_curves.png"))

    # Generate comparison report
    report_md = generate_comparison_report(all_results, all_metrics)
    with open(CFG.BASELINE_REPORT, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Comparison report: {CFG.BASELINE_REPORT}")

    # Save raw metrics JSON
    metrics_json = {k: {mk: float(mv) if isinstance(mv, (float, np.floating)) else mv
                        for mk, mv in v.items()}
                    for k, v in all_metrics.items()}
    with open(CFG.RESULTS_CSV.replace(".csv", ".json"), "w") as f:
        json.dump(metrics_json, f, indent=2)

    logger.info("09_evaluate.py COMPLETE")


if __name__ == "__main__":
    main()
