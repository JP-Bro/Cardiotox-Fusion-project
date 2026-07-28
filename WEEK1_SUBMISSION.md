# Cardiotox-Fusion — Week 1 Project Deliverables & Submission Guide

This document lists the exact files, outputs, and verification commands completed and uploaded to the GitHub repository for the **Week 1** milestone.

---

## 1. Repository Information
* **Repository Name:** `Cardiotox-Fusion-project`
* **Owner:** `JP-Bro`
* **URL:** [https://github.com/JP-Bro/Cardiotox-Fusion-project](https://github.com/JP-Bro/Cardiotox-Fusion-project)
* **Status:** All Week 1 files have been successfully pushed and are up-to-date with `origin/main`.

---

## 2. Directory Structure of Uploaded Deliverables
Below is the directory manifest showing where each required Week 1 deliverable is located in the repository:

```text
cardiotox-fusion/
├── data/
│   └── splits/
│       ├── drug_split.csv             # Stratified Drug-level split (Train/Val/Test)
│       └── scaffold_split.csv         # Bemis-Murcko Scaffold-level split
├── notebooks/
│   └── phase1_usability_and_splits.ipynb  # Executable Jupyter Notebook (cell-by-cell)
├── results/
│   ├── phase1_audit_report.md         # Markdown summary report of audit results
│   ├── Cardiotox_Fusion_Phase1_Report.docx  # Compiled Word report (primary)
│   ├── phase1_usable_cohort.png       # Plot 1: Usable cohort counts by class
│   └── phase1_splits_distribution.png # Plot 2: Partition size comparison
└── scripts/
    ├── phase1_audit_splits.py         # Main execution script for audit and splits
    ├── phase1_plot.py                 # Visualization script generating the 2 plots
    └── create_docx.py                 # Script compiling the Word report
```

---

## 3. Usability Audit Summary (Ground Truth Counts)
* **Usable Compounds in Dataset:** **562** (filtered for RDKit-valid SMILES and matched LINCS L1000 Level-5 active signatures).
* **Per-Class Distribution:**
  * **Most Concern (High Risk):** 199 compounds (35.4%)
  * **Less Concern (Medium Risk):** 246 compounds (43.8%)
  * **No Concern (Safe):** 117 compounds (20.8%)
* **Imbalance Handling:** Due to the 79.2% positive cardiotoxicity rate, **AUC-PR** is frozen as our primary evaluation metric.

---

## 4. Dataset Partition Summary (70 / 15 / 15)

### A. Drug-Level Splits (Stratified by Cardiotoxicity Label)
* **Train:** 393 compounds (Positive Rate: 79.1%)
* **Validation:** 84 compounds (Positive Rate: 79.8%)
* **Test:** 85 compounds (Positive Rate: 78.8%)

### B. Bemis-Murcko Scaffold-Level Splits (Grouped by Carbon Scaffold)
* **Train:** 393 compounds (Positive Rate: 79.4%)
* **Validation:** 84 compounds (Positive Rate: 75.0%)
* **Test:** 85 compounds (Positive Rate: 82.4%)

---

## 5. Methodological & Rigorous Constraints (Addressing Faculty Feedback)

To ensure scientific rigor, prevent evaluation contamination, and align with Dr. Bharat Manna's feedback, we have established three explicit guidelines:

1.  **Label Framing (DICTrank vs. hERG):** The classification model is trained to predict general cardiotoxicity classes from the FDA DICTrank dataset, not specific hERG channel blockade. While hERG blockade is the primary physical mechanism of cardiotoxicity, DICTrank labels represent clinical cardiotoxicity concern. We target general cardiotoxicity, and move all hERG-specific claims to post-hoc discussion, maintaining clinical honesty.
2.  **L1000 Signature Aggregation Rule:** Replicate Level-5 L1000 signatures (COMPZ z-scores) in the HA1E cell line under the target condition of 10.0 µM dose and 24 h exposure are **mean-aggregated** to construct a single consolidated transcriptomic vector per compound.
3.  **Independent Model Evaluation (No Fusion):** To prevent target leakage and contamination (which happens when structurally-imputed profiles are mixed with experimental ones), **we do not perform any fusion**. The GNN (structure-only) and the Transformer (biology-only) models are evaluated as completely independent parallel architectures:
    *   *GNN (Structure-only):* Operates on molecular graphs derived from chemical SMILES.
    *   *Transformer (Biology-only):* Operates on experimental LINCS L1000 signatures.
    *   *Fair Comparison Across Population Sizes:* The GNN will be evaluated on both (a) the full DICTrank set and (b) the L1000-matched subset, allowing a direct head-to-head comparison on identical drugs.

---

## 6. Academic Novelty & Publishability Strategy

In response to previous literature (e.g., Seal et al. 2024), we highlight our core novelties which make this project publishable as a rigorous benchmarking study:

*   **Bemis-Murcko Scaffold Splits on DICTrank:** Prior studies did not perform scaffold splits on DICTrank. This evaluation assigns whole scaffold clusters to the held-out test set until at least 15% of unique drugs are covered, directly quantifying how much cardiotoxicity prediction is driven by chemical-series memorization versus out-of-distribution generalizability to unseen chemical families.
*   **Testing omics Signal Recovery:** Previous shallow classifiers reported near-random performance (~0.57 AUC) for L1000 transcriptomics. We directly test whether a learned representation (multi-head Transformer encoder) can recover predictive biological signal that simpler methods missed. Both positive and negative outcomes will be reported honestly.

---

## 7. Verification Commands
To re-run the usability audit, re-generate the splits, and re-create the visualization plots locally, execute the following commands from the repository root:

```bash
# 1. Run the usability audit and generate the split files
python scripts/phase1_audit_splits.py

# 2. Generate the two separate visualization plots
python scripts/phase1_plot.py

# 3. Re-compile the Microsoft Word report
python scripts/create_docx.py
```

*All scripts run with a frozen random seed (`42`) set in `config.py` to guarantee identical outputs across all machines.*
