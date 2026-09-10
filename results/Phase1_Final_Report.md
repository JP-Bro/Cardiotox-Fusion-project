# Cardiotox-Fusion — Phase 1 Final Report

**Project:** Dual-Modality Cardiotoxicity Prediction (GNN + Transformer)  
**Team:** Jimit Patel, Mallu Vineela Reddy, Aditya Gaurang Patel, Nehal Solanki, Pranjal Chhipa  
**Repository:** https://github.com/JP-Bro/Cardiotox-Fusion-project  
**Date:** September 2026

---

## 1. Project Overview

We are building two independent deep learning models to predict FDA DICTrank cardiotoxicity:

| Branch | Input | Model |
|---|---|---|
| **Structure** | 2D molecular graph (atoms + bonds) | 3-Layer Graph Isomorphism Network (GIN) |
| **Expression** | 978-gene LINCS L1000 z-score vector | 2-Layer Transformer Encoder with learned gene-identity embeddings |

Both models are evaluated on the **same** 487-drug scaffold split so that results are directly comparable. This is the only setting where a fair structure vs. biology comparison is possible.

---

## 2. Dataset Construction

### 2.1 Source
- **FDA DICTrank** (1,318 raw entries → 1,211 after dropping 107 ambiguous)
- **LINCS L1000 Level-5 COMPZ** (GSE70138, 118,050 signatures × 12,328 genes)

### 2.2 Binary Labelling
`Most Concern` + `Less Concern` = **1 (Cardiotoxic)**  
`No Concern` = **0 (Safe)**  
Ambiguous class dropped entirely (follows Seal et al. 2024).

### 2.3 LINCS Matching (Strict Cohort)
Filter: **HA1E cell line / 10.0 µM dose / 24 h treatment / `trt_cp` type**  
Matched signatures: **1,837** across **582** raw DICTrank entries.

### 2.4 Landmark Gene Verification
Downloaded `GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz` and cross-referenced `pr_is_lm == 1`.

**Critical finding:** The 978 true landmark genes are **scattered** across GCTX row indices 0–12,321, not in the first 978 rows. Slicing the first 978 rows yields only 194 landmark genes. All 978 row indices are mapped explicitly and extracted.

**STEP 1 console output (verify_landmark_genes):**
```
Verified total landmark genes in gene_info.txt.gz: 978
Total row IDs in GCTX file: 12328
Found 978 landmark gene rows in GCTX file.
Landmark row indices range: min=0, max=12321
NOTICE: Landmark genes are scattered across GCTX rows. Using verified landmark index mapping.
```

### 2.5 Deduplication & Label Conflict Resolution
- Formal charges neutralised with RDKit `Uncharger`.
- Compounds grouped by **14-character InChIKey connectivity block** (removes salt forms, ionic variants, and near-duplicates).
- Label conflicts resolved: if any salt/form of a skeleton carries cardiotoxicity risk (label 1), the skeleton is assigned label 1.
  - `ACYCLOVIR` (0) vs `ACYCLOVIR SODIUM` (1) → **Resolved to 1**
  - `OMEPRAZOLE` (0) vs `ESOMEPRAZOLE MAGNESIUM` (1) → **Resolved to 1**

### 2.6 Final Dataset
| Metric | Value |
|---|---|
| Unique chemical skeletons (14-char InChIKey blocks) | **487** |
| Cardiotoxic (label 1) | 384 (78.8%) |
| Safe (label 0) | 103 (21.2%) |
| Expression matrix shape | 487 × 978 |

---

## 3. Data Splits

### 3.1 Drug-Level Split (Stratified)
Groups all entries by InChIKey connectivity block before splitting. Cross-split structural leakage: **0%**.

| Partition | Compounds |
|---|---|
| Train | 340 |
| Validation | 73 |
| Test | 74 |

### 3.2 Scaffold Split (Bemis-Murcko)
- Scaffolds computed with `includeChirality=False` (diastereomers like Quinine/Quinidine share one scaffold).
- Acyclic compounds each receive a unique pseudo-scaffold ID to prevent false clustering.
- Scaffold clusters allocated by **compound count** (greedy packing, not scaffold count).

| Partition | Compounds |
|---|---|
| Train | 340 |
| Validation | 73 |
| Test | 74 |

---

## 4. Statistical Power & Evaluation Strategy

The scaffold test set has **10 negative (safe) compounds** out of 74. The random classifier 95% CI for AUC-PR is **[0.785, 0.915]** — wide but honestly reported.

### Evaluation approach
- **Primary:** Both GNN and Transformer evaluated on the **same 487-drug scaffold split**. Bootstrap 95% CIs on all metrics.
- **Cohort choice:** We retain the **strict HA1E / 10.0 µM / 24 h** cohort for scientific rigour. The LINCS coverage limitation is stated plainly.
- **Secondary (structure only):** GNN additionally evaluated on the full 1,211-drug DICTrank set as a clearly labelled secondary analysis — never compared against the Transformer headline.

---

## 5. Baseline Results & Critical Metric Interpretation

Evaluated on the strict Bemis-Murcko scaffold test set (N = 74: 64 Toxic, 10 Safe; 86.5% positive prevalence).

### Metric Reporting Hierarchy
In an imbalanced evaluation set with 86.5% positive prevalence, the random classifier achieves an AUC-PR floor of **0.8649**. A trivial model that labels all compounds as toxic achieves high AUC-PR while learning nothing. Therefore, **AUC-ROC** and **Balanced Accuracy** serve as the primary discriminative metrics, with **AUC-PR** reported alongside its 0.865 prevalence floor.

| Modality & Model | AUC-ROC (Primary) | 95% Bootstrap CI | Balanced Acc | AUC-PR (Floor = 0.865) | 95% Bootstrap CI |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Guess (Baseline)** | **0.5000** | — | **0.5000** | **0.8649** | — |
| **Structure: Random Forest** (200 trees, Morgan FP) | **0.7227** | [0.523, 0.906] | **0.6375** | **0.9395** | [0.879, 0.990] |
| **Structure: Support Vector** (RBF, Morgan FP) | **0.7078** | [0.489, 0.895] | **0.6000** | **0.9326** | [0.862, 0.989] |
| **Biology: Sparse Logistic Reg** (L1 / Lasso, L1000) | **0.5922** | [0.377, 0.801] | **0.5672** | **0.9061** | [0.821, 0.972] |
| **Biology: Support Vector** (Linear, L1000) | **0.5141** | [0.286, 0.731] | **0.5000** | **0.8549** | [0.752, 0.966] |
| **Biology: Logistic Regression** (L2, L1000) | **0.4875** | [0.260, 0.721] | **0.5062** | **0.8503** | [0.741, 0.954] |

### Rigorous Scientific Interpretation
1. **Structure Modality Signal:** Molecular fingerprints coupled with Random Forest demonstrate genuine predictive power (**AUC-ROC = 0.723**, 95% CI strictly above 0.50). Any deep GNN trained in Phase 2 must outperform this floor to justify graph convolution complexity.
2. **Biology Modality Signal:** Classical linear and shallow models on raw 978 landmark gene expression exhibit no detectable signal above a random coin flip (all 95% bootstrap CIs span 0.500; L1 Lasso CI is `[0.377, 0.801]`, which encompasses random chance). This demonstrates that linear projections cannot isolate cardiotoxic transcriptional perturbations from 978 genes, establishing the necessity for deep non-linear attention architectures (Transformer Encoder).
3. **Statistical Power Limitation:** The 74-compound scaffold test set contains 10 negatives. The wide bootstrap confidence intervals honestly reflect this biological data constraint and are reported transparently.

---

## 6. Pipeline Reproducibility

All four scripts run sequentially from raw data on a clean clone (after downloading the LINCS files — see README):

```bash
python scripts/01_fetch_labels.py       # DICTrank → labeled_compounds.csv
python scripts/02_resolve_smiles.py     # SMILES resolution + charge neutralisation
python scripts/03_rebuild_pipeline.py   # Landmark extraction + splits
python scripts/04_baselines.py          # Baseline models
```

> **Note:** Raw LINCS files (5.8 GB GCTX, sig_info, pert_info, pert_info, gene_info) must be downloaded from GEO (GSE70138) and placed in `data/raw/lincs/`. The download step is documented in `README.md`.

---

## 7. Next Steps (Phase 2)

1. **GNN baseline** — 3-layer GIN trained on the 340-drug scaffold train set, evaluated on the 74-drug test set with bootstrapped CIs.
2. **Transformer baseline** — 2-layer, 4-head Transformer Encoder with learned gene-identity embeddings, same split.
3. **Report comparison** — Present all four models (RF, LR, GNN, Transformer) in one table with bootstrapped 95% CIs, under the same scaffold test set.
