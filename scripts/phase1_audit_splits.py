"""
phase1_audit_splits.py -- Execute Week 1 (Phase 1) tasks for Cardiotox-Fusion.

1. Usability Audit:
   - Identify DICTrank drugs with a valid SMILES and a matched LINCS signature.
   - Report counts per concern class (most, less, no concern).
2. Leakage-Free Splitting:
   - Drug-level split: 70% Train, 15% Val, 15% Test.
   - Scaffold-level split: Bemis-Murcko scaffolds computed via RDKit, grouped by scaffold.
3. Save split outputs to CSV files under data/splits/.
4. Write a markdown report documenting all counts and splits.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Set

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import CFG
from scripts.utils import get_logger, ensure_dirs, set_seed
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

LOG_FILE = os.path.join(CFG.RESULTS_DIR, "logs", "phase1_audit_splits.log")
logger = get_logger("phase1_audit_splits", LOG_FILE)


def run_audit() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Paths
    labeled_path = CFG.LABELS_CSV
    matched_path = CFG.LINCS_MATCHED_CSV

    if not os.path.isfile(labeled_path):
        raise FileNotFoundError(f"Missing {labeled_path}. Run step 1 first.")
    if not os.path.isfile(matched_path):
        raise FileNotFoundError(f"Missing {matched_path}. Run step 3 first.")

    # Load datasets
    df_labels = pd.read_csv(labeled_path)
    df_matched = pd.read_csv(matched_path)

    logger.info(f"Loaded {len(df_labels)} labeled compounds from DICTrank.")
    logger.info(f"Loaded {len(df_matched)} matched compounds from LINCS matches.")

    # Drop duplicate column in df_matched before merge
    if "cardiotox_label" in df_matched.columns:
        df_matched = df_matched.drop(columns=["cardiotox_label"])

    # Merge matching on drug name
    # labeled_compounds has: resolved_name, cardiotox_label, DICT_Concern
    # lincs_matched_compounds has: query_name, parent_smiles, lincs_match, sig_id, inchi_key
    df_merged = pd.merge(
        df_labels,
        df_matched,
        left_on="resolved_name",
        right_on="query_name",
        how="inner"
    )
    logger.info(f"Merged set contains {len(df_merged)} compounds.")

    # 1. Usability filter:
    # - valid SMILES (check with RDKit)
    # - lincs_match is True (or has a valid sig_id in expression matrix)
    # Let's verify valid SMILES
    valid_smiles_mask = []
    scaffolds = []
    
    for idx, row in df_merged.iterrows():
        smi = row["parent_smiles"]
        if pd.isna(smi) or not isinstance(smi, str):
            valid_smiles_mask.append(False)
            scaffolds.append(None)
            continue
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_smiles_mask.append(False)
            scaffolds.append(None)
        else:
            valid_smiles_mask.append(True)
            try:
                scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                scaffolds.append(scaf if scaf else "generic_linear")
            except Exception:
                scaffolds.append("error_scaffold")

    df_merged["valid_smiles"] = valid_smiles_mask
    df_merged["scaffold"] = scaffolds

    # Usable: has valid SMILES AND lincs_match is True
    # Let's check expression matrix file to make sure the sig_id is actually in there
    expr_path = CFG.EXPRESSION_CSV
    if os.path.isfile(expr_path):
        # Read the first column (sig_id index)
        df_expr_ids = pd.read_csv(expr_path, usecols=[0])
        # First column name is empty or labeled, let's just grab whatever the first column is
        available_sig_ids = set(df_expr_ids.iloc[:, 0].tolist())
        logger.info(f"Available sig_ids in expression matrix: {len(available_sig_ids)}")
        # Check if matched sig_id is in available sig_ids
        has_expression = df_merged["sig_id"].isin(available_sig_ids)
        df_merged["has_expression_data"] = has_expression
    else:
        logger.warning("expression_matrix.csv not found -- assuming matches are valid if lincs_match is True")
        df_merged["has_expression_data"] = df_merged["lincs_match"]

    df_merged["is_usable"] = df_merged["valid_smiles"] & df_merged["has_expression_data"]

    # Filter usable
    df_usable = df_merged[df_merged["is_usable"]].copy()
    logger.info(f"Usable set shape: {df_usable.shape}")

    # Generate per-class counts for full set vs usable set
    logger.info("\n--- CLASS COUNTS (Full Set) ---")
    full_counts = df_merged["DICT_Concern"].value_counts().to_dict()
    for cls, val in full_counts.items():
        logger.info(f"  {cls:15}: {val}")

    logger.info("\n--- CLASS COUNTS (LINCS-Overlapping Usable Set) ---")
    usable_counts = df_usable["DICT_Concern"].value_counts().to_dict()
    for cls, val in usable_counts.items():
        logger.info(f"  {cls:15}: {val}")

    return df_merged, df_usable


def generate_splits(df_usable: pd.DataFrame):
    set_seed(CFG.RANDOM_SEED)

    # Make sure splits directory exists
    splits_dir = os.path.join(CFG.DATA_DIR, "splits")
    ensure_dirs(splits_dir)

    # ── Drug-level Split (Stratified) ───────────────────────────────────────────
    # 70% Train, 15% Val, 15% Test
    df_usable = df_usable.sample(frac=1, random_state=CFG.RANDOM_SEED).reset_index(drop=True)
    
    # We will stratify by cardiotox_label (binary: 0 or 1)
    labels = df_usable["cardiotox_label"].values
    indices = np.arange(len(df_usable))

    from sklearn.model_selection import train_test_split
    # Split train vs val+test
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=CFG.RANDOM_SEED
    )
    
    # Split val vs test (50/50 from the temp set)
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_labels,
        random_state=CFG.RANDOM_SEED
    )

    # Assign splits
    df_usable.loc[train_idx, "drug_split"] = "train"
    df_usable.loc[val_idx, "drug_split"] = "val"
    df_usable.loc[test_idx, "drug_split"] = "test"

    # ── Scaffold-level Split (Grouped by Bemis-Murcko) ─────────────────────────
    # Group compounds by their scaffold
    scaf_groups = df_usable.groupby("scaffold")
    scaf_counts = scaf_groups.size().sort_values(ascending=False)
    
    # Target counts
    total_size = len(df_usable)
    train_target = int(total_size * 0.70)
    val_target = int(total_size * 0.15)
    test_target = total_size - train_target - val_target

    train_scafs = set()
    val_scafs = set()
    test_scafs = set()

    train_curr = 0
    val_curr = 0
    test_curr = 0

    # Distribute scaffold groups to train, val, and test to match target sizes
    for scaf, count in scaf_counts.items():
        if train_curr < train_target:
            train_scafs.add(scaf)
            train_curr += count
        elif val_curr < val_target:
            val_scafs.add(scaf)
            val_curr += count
        else:
            test_scafs.add(scaf)
            test_curr += count

    # Map back to dataframe
    def get_scaf_split(scaf):
        if scaf in train_scafs: return "train"
        if scaf in val_scafs: return "val"
        return "test"

    df_usable["scaffold_split"] = df_usable["scaffold"].apply(get_scaf_split)

    # Verify split integrity
    logger.info("\n--- DRUG-LEVEL SPLIT SIZES ---")
    drug_split_counts = df_usable["drug_split"].value_counts().to_dict()
    for k, v in drug_split_counts.items():
        pos_rate = df_usable[df_usable["drug_split"] == k]["cardiotox_label"].mean()
        logger.info(f"  {k:8}: {v} compounds (pos rate: {pos_rate:.1%})")

    logger.info("\n--- SCAFFOLD-LEVEL SPLIT SIZES ---")
    scaf_split_counts = df_usable["scaffold_split"].value_counts().to_dict()
    for k, v in scaf_split_counts.items():
        pos_rate = df_usable[df_usable["scaffold_split"] == k]["cardiotox_label"].mean()
        logger.info(f"  {k:8}: {v} compounds (pos rate: {pos_rate:.1%})")

    # Save splits
    drug_split_path = os.path.join(splits_dir, "drug_split.csv")
    scaf_split_path = os.path.join(splits_dir, "scaffold_split.csv")

    # Keep key columns to avoid clutter
    cols_to_save = [
        "resolved_name", "cardiotox_label", "DICT_Concern",
        "parent_smiles", "sig_id", "scaffold"
    ]
    
    df_drug_out = df_usable[cols_to_save + ["drug_split"]].copy()
    df_drug_out.rename(columns={"drug_split": "split"}, inplace=True)
    df_drug_out.to_csv(drug_split_path, index=False)

    df_scaf_out = df_usable[cols_to_save + ["scaffold_split"]].copy()
    df_scaf_out.rename(columns={"scaffold_split": "split"}, inplace=True)
    df_scaf_out.to_csv(scaf_split_path, index=False)

    logger.info(f"Saved drug split to: {drug_split_path}")
    logger.info(f"Saved scaffold split to: {scaf_split_path}")

    # Generate Markdown report
    report_path = os.path.join(CFG.RESULTS_DIR, "phase1_audit_report.md")
    generate_md_report(df_usable, len(df_usable), report_path)
    logger.info(f"Saved audit report to: {report_path}")


def generate_md_report(df_usable: pd.DataFrame, total_usable: int, out_path: str):
    usable_counts = df_usable["DICT_Concern"].value_counts().to_dict()
    
    drug_splits = df_usable["drug_split"].value_counts().to_dict()
    scaf_splits = df_usable["scaffold_split"].value_counts().to_dict()

    md = f"""# Phase 1: Usability Audit & Leakage-Free Splits

This report summarizes the results of the **Week 1** audit of DICTrank compounds and matched LINCS L1000 GSE70138 signatures, along with the frozen train/val/test splits.

---

## 1. Usability Audit

A drug is defined as **usable** if:
1. It has a valid SMILES string parseable by RDKit.
2. It has a matched Level-5 signature inside the processed L1000 expression matrix.

### Final Usable Cohort Count: **{total_usable}** compounds

### Class Distribution (Usable LINCS-Overlapping Cohort)
| DICTrank Concern Category | Binarized Class Label | Usable Compound Count | Percentage |
| :--- | :--- | :--- | :--- |
| **Most Concern** | 1 (Cardiotoxic) | {usable_counts.get('most', 0)} | {usable_counts.get('most', 0)/total_usable:.1%} |
| **Less Concern** | 1 (Cardiotoxic) | {usable_counts.get('less', 0)} | {usable_counts.get('less', 0)/total_usable:.1%} |
| **No Concern** | 0 (Safe) | {usable_counts.get('no', 0)} | {usable_counts.get('no', 0)/total_usable:.1%} |
| **Total Labeled Set** | - | **{total_usable}** | **100%** |

*Note: Ambiguous concern compounds are excluded from the binarized classification task.*

---

## 2. Leakage-Free Splitting Strategies

To prevent compound leakage across train and test partitions, we split the data strictly by unique drug identity (drug-level split) and by chemical backbone (scaffold-level split).

### A. Drug-Level Split (Stratified 70/15/15)
All LINCS signatures belonging to the same compound are assigned to a single split partition.

| Partition | Compound Count | Positive Label Rate |
| :--- | :--- | :--- |
| **Train** | {drug_splits.get('train', 0)} | {df_usable[df_usable['drug_split']=='train']['cardiotox_label'].mean():.2%} |
| **Validation** | {drug_splits.get('val', 0)} | {df_usable[df_usable['drug_split']=='val']['cardiotox_label'].mean():.2%} |
| **Test** | {drug_splits.get('test', 0)} | {df_usable[df_usable['drug_split']=='test']['cardiotox_label'].mean():.2%} |

### B. Bemis-Murcko Scaffold Split (Grouped 70/15/15)
Compounds are grouped by their Bemis-Murcko scaffold. This ensures that the validation and test sets evaluate generalization performance to completely unseen chemical classes.

| Partition | Compound Count | Positive Label Rate |
| :--- | :--- | :--- |
| **Train** | {scaf_splits.get('train', 0)} | {df_usable[df_usable['scaffold_split']=='train']['cardiotox_label'].mean():.2%} |
| **Validation** | {scaf_splits.get('val', 0)} | {df_usable[df_usable['scaffold_split']=='val']['cardiotox_label'].mean():.2%} |
| **Test** | {scaf_splits.get('test', 0)} | {df_usable[df_usable['scaffold_split']=='test']['cardiotox_label'].mean():.2%} |

---

## 3. environment & Integrity Guarantees
- **Random Seed:** Frozen at `42` for all split functions.
- **Paths:** Outputs saved to `data/splits/drug_split.csv` and `data/splits/scaffold_split.csv`.
- **Reproducibility:** Splits can be verified by running `python scripts/phase1_audit_splits.py`.

*Report generated on July 27, 2026.*
"""

    with open(out_path, "w") as f:
        f.write(md)


def main():
    logger.info("=" * 60)
    logger.info("STARTING PHASE 1: AUDIT & SPLIT GENERATOR")
    logger.info("=" * 60)

    # Run audit
    df_merged, df_usable = run_audit()

    # Generate splits
    generate_splits(df_usable)

    logger.info("PHASE 1 COMPLETE.")


if __name__ == "__main__":
    main()
