# Cardiotox-Fusion

**Dual-modality cardiotoxicity prediction using molecular structure (GNN) and L1000 gene expression (Transformer).**

---

## Repository Structure

```
cardiotox-fusion/
├── data/
│   ├── raw/
│   │   ├── dictrank/          # FDA DICTrank Excel file (download separately)
│   │   └── lincs/             # LINCS L1000 raw files (download separately — see below)
│   ├── processed/
│   │   ├── labeled_compounds.csv
│   │   └── compounds_with_smiles.csv
│   └── splits/
│       ├── drug_split.csv         # Stratified 70/15/15 split (InChIKey skeleton grouped)
│       └── scaffold_split.csv     # Bemis-Murcko scaffold split (70/15/15 by compound count)
├── scripts/
│   ├── 01_fetch_labels.py         # DICTrank Excel → labeled_compounds.csv
│   ├── 02_resolve_smiles.py       # SMILES resolution + charge neutralisation
│   ├── 03_rebuild_pipeline.py     # LINCS matching, landmark gene extraction, splits
│   └── 04_baselines.py            # RF + Morgan FP, LR + L1000 expression baselines
├── results/
│   ├── baseline_results.csv
│   └── pipeline_rebuild_audit.md
├── config.py
└── WEEK1_SUBMISSION.md
```

---

## Downloading Raw Data (Required Before Running)

The following large files must be downloaded manually and placed in `data/raw/lincs/`:

| File | Size | Source |
|---|---|---|
| `GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx` | 5.8 GB | [GEO GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138) |
| `GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz` | 217 KB | [GEO GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138) |
| `sig_info.txt.gz` | — | [GEO GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138) |
| `pert_info.txt.gz` | — | [GEO GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138) |

The FDA DICTrank Excel file (`DICT_2023.xlsx` or similar) should be placed in `data/raw/dictrank/` and its path set in `config.py`.

---

## Reproducing the Pipeline

After downloading the raw files, run the four scripts in order:

```bash
python scripts/01_fetch_labels.py       # Process DICTrank labels
python scripts/02_resolve_smiles.py     # Resolve SMILES & neutralise charges
python scripts/03_rebuild_pipeline.py   # Build expression matrix & splits
python scripts/04_baselines.py          # Run baseline models
```

All scripts use `config.py` for paths and `RANDOM_SEED = 42`.

---

## Dataset Summary

- **Source:** FDA DICTrank (1,211 non-ambiguous drugs after dropping 107 ambiguous)
- **Cohort:** Strict HA1E / 10.0 µM / 24 h LINCS condition — **487 unique chemical skeletons**
- **Labels:** Binary (Most + Less Concern = 1 [Toxic], No Concern = 0 [Safe])
- **Class balance:** 384 toxic / 103 safe (79.3% / 20.7%)
- **Grouping:** Deduplicated by **14-character InChIKey connectivity block** (charge-neutralised)

---

## Baseline Results (Scaffold Split Test Set, N = 74: 64 Toxic, 10 Safe)

> **Metric Reporting Standard:** In an imbalanced test set with 86.5% positive prevalence, a random classifier achieves an AUC-PR floor of **0.8649**. Thus, **AUC-ROC** and **Balanced Accuracy** serve as the primary discriminative metrics, with **AUC-PR** reported alongside its 0.865 prevalence baseline.

| Modality & Model | AUC-ROC (Primary) | 95% Bootstrap CI | Balanced Acc | AUC-PR (Floor = 0.865) | 95% Bootstrap CI |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Guess (Baseline)** | **0.5000** | — | **0.5000** | **0.8649** | — |
| **Structure: Random Forest** (200 trees, Morgan FP) | **0.7227** | [0.523, 0.906] | **0.6375** | **0.9395** | [0.879, 0.990] |
| **Structure: Support Vector** (RBF, Morgan FP) | **0.7078** | [0.489, 0.895] | **0.6000** | **0.9326** | [0.862, 0.989] |
| **Biology: Sparse Logistic Reg** (L1 / Lasso, L1000) | **0.5922** | [0.377, 0.801] | **0.5672** | **0.9061** | [0.821, 0.972] |
| **Biology: Support Vector** (Linear, L1000) | **0.5141** | [0.286, 0.731] | **0.5000** | **0.8549** | [0.752, 0.966] |
| **Biology: Logistic Regression** (L2, L1000) | **0.4875** | [0.260, 0.721] | **0.5062** | **0.8503** | [0.741, 0.954] |

### Key Scientific Takeaway
1. **Structure Modality:** 2D Morgan fingerprints with Random Forest achieve real, moderate predictive signal (**AUC-ROC = 0.723**, 95% CI strictly above 0.50).
2. **Biology Modality:** Classical linear models on raw 978 landmark gene expression exhibit no detectable signal above a random coin flip (all 95% CIs cross 0.500). This aligns with established literature and justifies the use of a deep self-attention Transformer to capture complex multi-gene biological perturbations.
3. **Statistical Power Limitation:** The 74-compound scaffold test set contains 10 negatives. The wide bootstrap confidence intervals honestly reflect this biological data availability constraint.
