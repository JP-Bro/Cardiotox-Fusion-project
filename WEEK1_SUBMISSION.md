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
│   ├── phase1_report.html             # Printable HTML report formatted for PDF
│   ├── phase1_usable_cohort.png       # Plot 1: Usable cohort counts by class
│   └── phase1_splits_distribution.png # Plot 2: Partition size comparison
└── scripts/
    ├── phase1_audit_splits.py         # Main execution script for audit and splits
    └── phase1_plot.py                 # Visualization script generating the 2 plots
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

## 5. Verification Commands
To re-run the usability audit, re-generate the splits, and re-create the visualization plots locally, execute the following commands from the repository root:

```bash
# 1. Run the usability audit and generate the split files
python scripts/phase1_audit_splits.py

# 2. Generate the two separate visualization plots
python scripts/phase1_plot.py
```

*All scripts run with a frozen random seed (`42`) set in `config.py` to guarantee identical outputs across all machines.*
