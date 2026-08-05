# Cardiotox-Fusion — Week 1 Project Deliverables & Submission Guide

This document lists the exact files, outputs, and verification commands completed and uploaded to the GitHub repository for the **Phase 1 (Week 1)** milestone.

---

## 1. Repository Information
* **Repository Name:** `Cardiotox-Fusion-project`
* **Owner:** `JP-Bro`
* **URL:** [https://github.com/JP-Bro/Cardiotox-Fusion-project](https://github.com/JP-Bro/Cardiotox-Fusion-project)
* **Status:** All Phase 1 pipeline fixes, frozen splits, and audit logs have been committed and pushed.

---

## 2. Directory Structure of Uploaded Deliverables
Below is the directory manifest showing where each required deliverable is located in the repository:

```text
cardiotox-fusion/
├── data/
│   └── splits/
│       ├── drug_split.csv             # Stratified Drug-level split (Parent-grouped, 0% cross-leak)
│       └── scaffold_split.csv         # Bemis-Murcko Scaffold-level split
├── notebooks/
│   └── phase1_usability_and_splits.ipynb  # Executable Jupyter Notebook
├── results/
│   ├── phase1_audit_report.md         # Summary report of audit results
│   ├── pipeline_rebuild_audit.md      # Detailed audit log of pipeline rebuild & gene verification
│   ├── Cardiotox_Fusion_Phase1_Report.docx  # Compiled Word report
│   ├── phase1_usable_cohort.png       # Plot 1: Usable cohort counts by class
│   └── phase1_splits_distribution.png # Plot 2: Partition size comparison
└── scripts/
    ├── 03_rebuild_pipeline.py         # End-to-end LINCS matching, mean-aggregation, & split generator
    ├── phase1_audit_splits.py         # Main execution script for audit and splits
    ├── phase1_plot.py                 # Visualization script
    └── create_docx.py                 # Script compiling Word report
```

---

## 3. Usability Audit Summary & Strict Cohort Counts (Post-Audit Fixes)
* **Ambiguous Class Handling:** Ambiguous concern drugs are removed from classification targets, mapping DICTrank to binary classification (`Most` + `Less` concern = 1 vs `No` concern = 0), matching Seal et al.
* **Parent Structure Deduplication:** Salt-form variants sharing identical parent SMILES (e.g., *Sildenafil* vs *Sildenafil Citrate*) are deduplicated into a single representative parent molecule.
* **Strict Usable Cohort (HA1E, 10.0 µM, 24 h):** **514 unique parent compounds** matching valid SMILES, verified landmark gene IDs (pr_is_lm = 1), and mean-aggregated Level-5 signatures.
* **Leakage Elimination:** Drug-level splits group by `parent_smiles` prior to splitting, guaranteeing **0% cross-split SMILES leakage** between train, val, and test.

---

## 4. Dataset Partition Summary (70 / 15 / 15)

### A. Drug-Level Splits (Grouped by Parent SMILES & Stratified by Label)
* **Train:** 359 compounds (78.8% Toxic)
* **Validation:** 78 compounds (78.2% Toxic)
* **Test:** 78 compounds (78.2% Toxic)

### B. Bemis-Murcko Scaffold-Level Splits (Grouped by Carbon Scaffold)
* **Train:** 344 compounds
* **Validation:** 65 compounds
* **Test:** 106 compounds

---

## 5. Verification Commands (Reproducibility)
To rebuild the entire matching, mean-aggregation, deduplication, and split pipeline from scratch on a fresh clone:

```bash
# 1. Run the end-to-end LINCS matching, aggregation, and split generation script
python scripts/03_rebuild_pipeline.py

# 2. Run verification script to confirm 0 cross-split leakage and class balance
python scripts/phase1_audit_splits.py
```
