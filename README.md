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

## Baseline Results (Scaffold Split Test Set, N = 74)

| Model | AUC-PR | 95% CI | AUC-ROC |
|---|---|---|---|
| Random Classifier (prevalence) | 0.8649 | — | 0.500 |
| LR on L1000 expression | 0.8600 | [0.7535, 0.9561] | 0.4828 |
| RF on Morgan fingerprints | **0.9344** | [0.8681, 0.9893] | **0.7125** |

> **Note:** Wide CIs reflect the small scaffold test set (10 negatives). All deep model comparisons will be evaluated against this same 487-drug scaffold split and bootstrapped CIs will be reported.
