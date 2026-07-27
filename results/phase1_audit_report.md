# Phase 1: Usability Audit & Leakage-Free Splits

This report summarizes the results of the **Week 1** audit of DICTrank compounds and matched LINCS L1000 GSE70138 signatures, along with the frozen train/val/test splits.

---

## 1. Usability Audit

A drug is defined as **usable** if:
1. It has a valid SMILES string parseable by RDKit.
2. It has a matched Level-5 signature inside the processed L1000 expression matrix.

### Final Usable Cohort Count: **562** compounds

### Class Distribution (Usable LINCS-Overlapping Cohort)
| DICTrank Concern Category | Binarized Class Label | Usable Compound Count | Percentage |
| :--- | :--- | :--- | :--- |
| **Most Concern** | 1 (Cardiotoxic) | 199 | 35.4% |
| **Less Concern** | 1 (Cardiotoxic) | 246 | 43.8% |
| **No Concern** | 0 (Safe) | 117 | 20.8% |
| **Total Labeled Set** | - | **562** | **100%** |

*Note: Ambiguous concern compounds are excluded from the binarized classification task.*

---

## 2. Leakage-Free Splitting Strategies

To prevent compound leakage across train and test partitions, we split the data strictly by unique drug identity (drug-level split) and by chemical backbone (scaffold-level split).

### A. Drug-Level Split (Stratified 70/15/15)
All LINCS signatures belonging to the same compound are assigned to a single split partition.

| Partition | Compound Count | Positive Label Rate |
| :--- | :--- | :--- |
| **Train** | 393 | 79.13% |
| **Validation** | 84 | 79.76% |
| **Test** | 85 | 78.82% |

### B. Bemis-Murcko Scaffold Split (Grouped 70/15/15)
Compounds are grouped by their Bemis-Murcko scaffold. This ensures that the validation and test sets evaluate generalization performance to completely unseen chemical classes.

| Partition | Compound Count | Positive Label Rate |
| :--- | :--- | :--- |
| **Train** | 393 | 79.39% |
| **Validation** | 84 | 75.00% |
| **Test** | 85 | 82.35% |

---

## 3. environment & Integrity Guarantees
- **Random Seed:** Frozen at `42` for all split functions.
- **Paths:** Outputs saved to `data/splits/drug_split.csv` and `data/splits/scaffold_split.csv`.
- **Reproducibility:** Splits can be verified by running `python scripts/phase1_audit_splits.py`.

*Report generated on July 27, 2026.*
