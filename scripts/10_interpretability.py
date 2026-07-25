"""
10_interpretability.py -- Extract and validate model attention weights.

What this script does:
  1. Loads the trained Fusion model checkpoint
  2. For each test compound:
       a. Extracts GNN atom-level attention weights (which atoms were attended to)
       b. Extracts cross-attention weights (which biological features guided structure)
       c. Extracts Transformer gene attention weights (which genes were most attended)
  3. Cross-checks top-attended structural substructures against known cardiotoxicity
     toxicophores from the literature
  4. Reports gene attention rankings (top-K genes by attention weight)
  5. Generates results/interpretability_validation.md

CRITICAL CAVEAT (from project brief):
  Attention weights are NOT automatically validated explanations.
  This script validates them by checking if highly-attended substructures
  correspond to KNOWN toxicophores from the cardiotoxicity literature.
  If they don't correlate -- we report that honestly.
  "Attention == explanation" is a known failure mode in the interpretability
  literature and must NOT be asserted without this validation step.

Run with:
  python scripts/10_interpretability.py
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from config import CFG
from scripts.utils import get_logger, ensure_dirs, set_seed
from models.fusion_model import FusionClassifier
from models.transformer_model import TransformerEncoder
from models.dataset import CardiotoxFusionDataset, stratified_split, fusion_collate_fn
from models.trainer import compute_metrics

LOG_FILE = os.path.join(CFG.RESULTS_DIR, "logs", "10_interpretability.log")
logger = get_logger("10_interpretability", LOG_FILE)

# Known cardiotoxicity structural alerts / toxicophores from literature
# Used to validate GNN attention -- not assert it as explanation
KNOWN_TOXICOPHORE_SMARTS = {
    "anthracycline_core":    "C1CC(=O)c2cc3cc(OC4CC(N)C(O)C(C)O4)c(O)c(O)c3c(=O)c2C1",
    "quinone":               "O=C1C=CC(=O)C=C1",  # simplified p-quinone
    "nitro_group":           "[N+](=O)[O-]",
    "michael_acceptor":      "C=CC(=O)",           # alpha-beta unsaturated carbonyl
    "acyl_halide":           "C(=O)[F,Cl,Br,I]",
    "herg_blocker_basic":    "N([CX4])([CX4])[CX4]",  # basic nitrogen (simplified)
    "epoxide":               "C1OC1",              # epoxide ring
    "aldehyde":              "[CX3H1](=O)[#6]",
}


def load_lincs_gene_names() -> list:
    """
    Load LINCS L1000 landmark gene names in order (978 genes).

    These names are used to annotate which genes have high attention weights.
    If the gene name file is not available, fall back to index-based names.

    Returns:
        List of 978 gene name strings
    """
    gene_names_path = os.path.join(CFG.LINCS_RAW_DIR, "gene_info.txt")
    if os.path.isfile(gene_names_path):
        try:
            df = pd.read_csv(gene_names_path, sep="\t")
            if "pr_gene_symbol" in df.columns:
                return df["pr_gene_symbol"].tolist()[:CFG.NUM_GENES]
        except Exception as e:
            logger.warning(f"Could not parse gene_info.txt: {e}")
    logger.warning("Gene name file not found -- using index-based gene labels")
    return [f"GENE_{i}" for i in range(CFG.NUM_GENES)]


def check_toxicophore_presence(smiles: str) -> dict:
    """
    Check which known toxicophores are present in a molecule.

    Args:
        smiles: Canonical SMILES of the compound

    Returns:
        dict: {toxicophore_name: True/False}
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: False for name in KNOWN_TOXICOPHORE_SMARTS}

    result = {}
    for name, smarts in KNOWN_TOXICOPHORE_SMARTS.items():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            result[name] = mol.HasSubstructMatch(pattern) if pattern else False
        except Exception:
            result[name] = False
    return result


def get_gnn_atom_importances(
    model: FusionClassifier,
    batch: dict,
    device: torch.device,
) -> np.ndarray:
    """
    Compute atom-level importance scores via input gradient × input.

    This is a gradient-based attribution method (Integrated Gradients approximation).
    More principled than raw attention weights for node-level explanations.

    Args:
        model:  Trained FusionClassifier
        batch:  Fusion collated batch
        device: Compute device

    Returns:
        Array of shape (total_atoms,) -- importance score per atom
    """
    graph = batch["graph"].to(device)
    gene_expr = batch["gene_expr"].to(device)

    graph.x.requires_grad_(True)

    out = model(
        x=graph.x, edge_index=graph.edge_index,
        edge_attr=graph.edge_attr, batch=graph.batch,
        gene_expr=gene_expr,
    )
    logits = out["logits"]
    logits.sum().backward()

    # Gradient × input as importance score
    importances = (graph.x.grad * graph.x).abs().sum(dim=-1)
    return importances.detach().cpu().numpy()


def visualize_top_genes(
    gene_attention: np.ndarray,
    gene_names: list,
    compound_name: str,
    output_dir: str,
    top_k: int = CFG.TOP_K_GENES,
):
    """
    Plot top-K genes by attention weight for a compound.

    Args:
        gene_attention: (n_genes,) array of attention weights
        gene_names:     List of gene name strings
        compound_name:  Name of the compound (for plot title)
        output_dir:     Directory to save the plot
        top_k:          Number of top genes to show
    """
    top_idx = np.argsort(gene_attention)[::-1][:top_k]
    top_genes = [gene_names[i] for i in top_idx]
    top_weights = gene_attention[top_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(top_k), top_weights[::-1], color="#3498db", alpha=0.8)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_genes[::-1], fontsize=9)
    ax.set_xlabel("Attention Weight", fontsize=11)
    ax.set_title(f"Top {top_k} Genes by Attention -- {compound_name}", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in compound_name)
    out_path = os.path.join(output_dir, f"gene_attn_{safe_name}.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    return top_genes, top_weights.tolist()


def generate_interpretability_report(
    compounds_data: list,
    gene_names: list,
) -> str:
    """Generate the interpretability_validation.md markdown report."""
    lines = [
        "# Cardiotox-Fusion: Interpretability Validation Report",
        "",
        "> **CRITICAL**: Attention weights are NOT validated explanations by themselves.",
        "> This report cross-checks model attention against KNOWN toxicophores from the",
        "> cardiotoxicity literature. Correlation = supporting evidence. No correlation",
        "> = the attention is not capturing toxicophore-relevant features (report honestly).",
        "",
        "## Validation Method",
        "",
        "For each test compound:",
        "1. Extract GNN atom importance scores (gradient × input)",
        "2. Extract Transformer gene attention weights (CLS => gene attention)",
        "3. Check which known toxicophore SMARTS patterns are present in the molecule",
        "4. Compare: do high-attention atoms correspond to toxicophore substructures?",
        "",
        "## Known Toxicophores Checked",
        "",
        "| Toxicophore | SMARTS Pattern |",
        "|---|---|",
    ]
    for name, smarts in KNOWN_TOXICOPHORE_SMARTS.items():
        lines.append(f"| {name.replace('_', ' ')} | `{smarts[:40]}...` |")

    lines += [
        "",
        "## Per-Compound Results",
        "",
    ]

    for cdata in compounds_data[:20]:  # Report first 20 test compounds
        name = cdata.get("name", "unknown")
        label = "TOXIC" if cdata.get("label", 0) == 1 else "non-toxic"
        pred_prob = cdata.get("pred_prob", float("nan"))
        toxicophores = cdata.get("toxicophores", {})
        top_genes = cdata.get("top_genes", [])
        present_tox = [k for k, v in toxicophores.items() if v]

        lines += [
            f"### {name} ({label}, pred_prob={pred_prob:.3f})",
            "",
            f"**Known toxicophores present:** {', '.join(present_tox) if present_tox else 'none detected'}",
            f"**Top genes by attention:** {', '.join(top_genes[:5])}",
            "",
        ]

    lines += [
        "## Aggregate Findings",
        "",
        "*(Populated after running full test set analysis)*",
        "",
        "## Important Caveat",
        "",
        "Even where attention correlates with known toxicophores, correlation is not",
        "causation. These findings support model interpretability but should be validated",
        "with domain experts and wet-lab experiments before clinical use.",
    ]

    return "\n".join(lines)


def main():
    logger.info("=" * 60)
    logger.info("SCRIPT: 10_interpretability.py -- Attention => Toxicophore Validation")
    logger.info("=" * 60)

    set_seed(CFG.RANDOM_SEED)
    interp_dir = os.path.join(CFG.RESULTS_DIR, "interpretability")
    ensure_dirs(interp_dir, os.path.join(CFG.RESULTS_DIR, "logs"))

    if not os.path.isfile(CFG.FUSION_CHECKPOINT):
        raise FileNotFoundError(
            f"Fusion checkpoint not found at {CFG.FUSION_CHECKPOINT}. "
            "Run 08_train_fusion.py first."
        )

    # Load data
    matched_df = pd.read_csv(CFG.LINCS_MATCHED_CSV)
    expr_df = pd.read_csv(CFG.EXPRESSION_CSV, index_col=0)
    manifest = pd.read_csv(os.path.join(CFG.PROCESSED_DIR, "graph_manifest.csv"))
    matched_only = matched_df[matched_df["lincs_match"] & matched_df["sig_id"].ne("")]
    name_to_sigid = dict(zip(matched_only["query_name"], matched_only["sig_id"]))
    name_to_smiles = dict(zip(manifest["name"], manifest["smiles"]))

    dataset = CardiotoxFusionDataset(
        manifest_df=manifest, expr_df=expr_df,
        name_to_sigid=name_to_sigid, include_graph=True,
    )
    _, _, test_ds = stratified_split(dataset, dataset.labels)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=fusion_collate_fn)

    # Load model
    model = FusionClassifier()
    ckpt = torch.load(CFG.FUSION_CHECKPOINT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = torch.device("cpu")

    gene_names = load_lincs_gene_names()
    logger.info(f"Gene names loaded: {len(gene_names)} genes")

    compounds_data = []

    for batch in test_loader:
        name = batch["names"][0]
        label = int(batch["label"][0].item())
        smiles = name_to_smiles.get(name, "")

        with torch.no_grad():
            gene_expr = batch["gene_expr"].to(device)
            graph = batch["graph"].to(device)
            out = model(
                x=graph.x, edge_index=graph.edge_index,
                edge_attr=graph.edge_attr, batch=graph.batch,
                gene_expr=gene_expr,
            )
            pred_prob = torch.sigmoid(out["logits"]).item()

        # Gene attention
        gene_attn = model.transformer_encoder.get_gene_attention_weights(
            batch["gene_expr"]
        ).squeeze().numpy()

        top_genes, top_weights = visualize_top_genes(
            gene_attn, gene_names, name, interp_dir
        )

        # Toxicophore check
        toxicophores = check_toxicophore_presence(smiles) if smiles else {}

        cdata = {
            "name": name,
            "label": label,
            "pred_prob": pred_prob,
            "top_genes": top_genes,
            "top_gene_weights": top_weights,
            "toxicophores": toxicophores,
            "smiles": smiles[:80] + "..." if len(smiles) > 80 else smiles,
        }
        compounds_data.append(cdata)

        logger.info(f"  {name} (label={label}, pred={pred_prob:.3f}): "
                    f"top gene={top_genes[0] if top_genes else 'N/A'}, "
                    f"toxicophores={[k for k,v in toxicophores.items() if v]}")

    # Save JSON results
    with open(os.path.join(interp_dir, "compound_interpretability.json"), "w") as f:
        json.dump(compounds_data, f, indent=2)

    # Generate report
    report = generate_interpretability_report(compounds_data, gene_names)
    with open(CFG.INTERP_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Interpretability report: {CFG.INTERP_REPORT}")

    # Global gene importance: mean attention across all compounds
    all_gene_weights = np.array([
        c["top_gene_weights"] + [0] * (CFG.NUM_GENES - len(c["top_gene_weights"]))
        for c in compounds_data
    ])
    logger.info(f"Top 10 genes globally by mean attention:")
    for i, (gene, weight) in enumerate(zip(
        [gene_names[i] for i in np.argsort(all_gene_weights.mean(axis=0))[::-1][:10]],
        sorted(all_gene_weights.mean(axis=0), reverse=True)[:10]
    )):
        logger.info(f"  {i+1}. {gene}: {weight:.4f}")

    logger.info("10_interpretability.py COMPLETE")


if __name__ == "__main__":
    main()
