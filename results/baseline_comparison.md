# Cardiotox-Fusion: Baseline Comparison Report

> **CRITICAL NOTE**: All three models evaluated on the **same LINCS-covered test set**.
> (Design Decision #6 from trace-through -- fair comparison requirement)

## Test Set Information

| Metric | Value |
|---|---|
| Test set size | 85 compounds |
| Positive (cardiotoxic) | 67 (78.8%) |
| Negative (no concern) | 18 (21.2%) |
| Split | Stratified 70/15/15 (seed=42) |

## Model Performance

| Model | AUC-ROC (95% CI) | AUC-PR | F1 | Accuracy |
|---|---|---|---|---|
| GNN (structure-only) | 0.6799 (0.552-0.795) | 0.9032 | 0.6275 | 0.5529 |
| Transformer (biology-only) | 0.5224 (0.361-0.680) | 0.7997 | 0.8816 | 0.7882 |
| Fusion (GNN+Transformer) | 0.6957 (0.567-0.805) | 0.9090 | 0.7193 | 0.6235 |

## Prior Work Reference

| Model | AUC-ROC | Source |
|---|---|---|
| Structure-only (chemical fingerprints) | 0.84 | Seal et al. 2023 |
| Biology-only (LINCS => GO features) | 0.76 | Seal et al. 2023 |
| Random baseline | 0.50 | -- |

## Key Limitations (Must Be Stated in Paper)

1. **LINCS cell lines are not cardiomyocytes** -- the biology branch learns
   a general cross-tissue transcriptional toxicity signature, not cardiac-specific.
   This is consistent with Seal et al.'s approach and their ~0.76 AUC-ROC ceiling.

2. **Dataset size** -- the fused model trains on ~423-450 compounds (LINCS-covered
   subset), vs. 1,211 for the structure-only baseline trained independently.
   Both are evaluated on the same test set here for fairness.

3. **Coverage imbalance** -- LINCS covers 37.7% of positive-label compounds
   vs. 28.0% of negative-label compounds. Stratified splits used throughout.

4. **Attention ≠ explanation** -- GNN and cross-attention weights are not
   validated explanations until cross-checked against known toxicophores.
   See results/interpretability_validation.md.