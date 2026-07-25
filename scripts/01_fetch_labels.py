"""
01_fetch_labels.py -- Step 1: Load, clean, and export DICTrank cardiotoxicity labels.

What this script does:
  1. Loads data/raw/dictrank_dataset_508.xlsx (already downloaded from FDA)
  2. Cleans column names and normalizes the DICT_Concern label column
  3. Binarizes: no->0, less/most->1, ambiguous->dropped
  4. Reports label statistics and data quality checks
  5. Saves clean labeled set to data/processed/labeled_compounds.csv

Known data quality issues encountered during trace-through (all handled here):
  - Column 'Label Section ' has trailing whitespace in raw file
  - Column 'DICT _ Concern' has inconsistent spacing
  - DICT_Concern values have mixed casing: 'less', 'Less', 'less '
  - 27/1,211 rows are missing 'Active Ingredient(s)' -- fallback to 'Generic/Proper Name(s)'
  - 1-row discrepancy vs. FDA's stated counts (less=528, ours=527) -- logged, non-blocking

Run with:
  python scripts/01_fetch_labels.py
"""

import os
import sys

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from config import CFG
from scripts.utils import get_logger, ensure_dirs, get_drug_name

LOG_FILE = os.path.join(CFG.RESULTS_DIR, "logs", "01_fetch_labels.log")
logger = get_logger("01_fetch_labels", LOG_FILE)


def load_and_clean_dictrank(path: str) -> pd.DataFrame:
    """
    Load the FDA DICTrank Excel file and apply all known cleaning steps.

    Problems found during trace-through and handled here:
      1. Trailing whitespace on column names -> str.strip() on all columns
      2. 'DICT _ Concern' inconsistent spacing -> rename to 'DICT_Concern'
      3. Mixed casing in concern values -> str.strip().str.lower()

    Args:
        path: Absolute path to dictrank_dataset_508.xlsx

    Returns:
        pd.DataFrame: Raw cleaned DICTrank with normalized 'DICT_Concern' column
    """
    logger.info(f"Loading DICTrank from: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"DICTrank file not found at {path}. "
            "Download from: https://www.fda.gov/media/178811/download"
        )

    df = pd.read_excel(path)  # requires openpyxl
    logger.info(f"  Raw shape: {df.shape}")

    # Fix 1: Strip all column name whitespace
    df.columns = df.columns.str.strip()

    # Fix 2: Rename inconsistently-spaced concern column
    # Multiple possible raw names observed -- handle both
    for raw_name in ["DICT _ Concern", "DICT_Concern", "DICT Concern"]:
        if raw_name in df.columns:
            df = df.rename(columns={raw_name: "DICT_Concern"})
            break
    assert "DICT_Concern" in df.columns, \
        f"Could not find concern column. Available: {list(df.columns)}"

    # Fix 3: Normalize concern values (mixed casing + whitespace)
    df["DICT_Concern"] = df["DICT_Concern"].str.strip().str.lower()

    logger.info(f"  Concern value counts (post-cleaning): {df['DICT_Concern'].value_counts().to_dict()}")
    return df


def binarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply binarization rule from project brief and drop ambiguous rows.

    Rule:
        'no'            -> cardiotox_label = 0
        'less', 'most'  -> cardiotox_label = 1
        'ambiguous'     -> DROPPED

    Also checks for any unmapped values and raises an error if found.

    Args:
        df: Cleaned DICTrank DataFrame with 'DICT_Concern' column

    Returns:
        pd.DataFrame: 1,211-row labeled set with 'cardiotox_label' column added
    """
    n_before = len(df)
    n_ambiguous = (df["DICT_Concern"] == CFG.DROP_LABEL).sum()

    df_labeled = df[df["DICT_Concern"] != CFG.DROP_LABEL].copy()
    df_labeled["cardiotox_label"] = df_labeled["DICT_Concern"].map(CFG.LABEL_MAP)

    # Check for any unmapped labels (would produce NaN)
    unmapped = df_labeled["cardiotox_label"].isna().sum()
    if unmapped > 0:
        unique_vals = df_labeled.loc[df_labeled["cardiotox_label"].isna(), "DICT_Concern"].unique()
        raise ValueError(f"Unmapped concern values found: {unique_vals}. Update CFG.LABEL_MAP.")

    n_pos = (df_labeled["cardiotox_label"] == 1).sum()
    n_neg = (df_labeled["cardiotox_label"] == 0).sum()
    logger.info(f"  Dropped {n_ambiguous} ambiguous rows ({n_before} -> {len(df_labeled)})")
    logger.info(f"  Label balance: {n_pos} positive (concern), {n_neg} negative (no concern)")
    logger.info(f"  Positive rate: {n_pos / len(df_labeled):.1%}")
    return df_labeled


def check_name_coverage(df_labeled: pd.DataFrame) -> pd.DataFrame:
    """
    Check 'Active Ingredient(s)' coverage and apply name fallback.

    Design Decision #1: Use 'Active Ingredient(s)', fall back to
    'Generic/Proper Name(s)'. Confirmed safe across full dataset (0 dead-ends).

    Args:
        df_labeled: Binarized DICTrank DataFrame

    Returns:
        Same DataFrame with new 'resolved_name' column
    """
    n_missing_ai = df_labeled["Active Ingredient(s)"].isna().sum()
    n_missing_gp = df_labeled["Generic/Proper Name(s)"].isna().sum()
    logger.info(f"  Missing 'Active Ingredient(s)': {n_missing_ai}/{len(df_labeled)} ({n_missing_ai/len(df_labeled):.1%})")
    logger.info(f"  Missing 'Generic/Proper Name(s)': {n_missing_gp}/{len(df_labeled)} (fallback safety check)")

    if n_missing_gp > 0:
        logger.warning(f"  {n_missing_gp} rows missing BOTH name fields -- investigate before proceeding!")

    df_labeled["resolved_name"] = df_labeled.apply(get_drug_name, axis=1)
    n_empty_name = (df_labeled["resolved_name"] == "").sum()
    if n_empty_name > 0:
        logger.error(f"  {n_empty_name} rows have empty resolved_name -- these will fail SMILES resolution!")

    logger.info(f"  'resolved_name' populated for all {len(df_labeled)} rows")
    return df_labeled


def save_labeled_csv(df_labeled: pd.DataFrame, output_path: str):
    """
    Save the cleaned, binarized, name-resolved labeled set to CSV.

    Columns saved:
        resolved_name, cardiotox_label, DICT_Concern,
        Trade Name, Active Ingredient(s), Generic/Proper Name(s)

    Args:
        df_labeled:  Processed DataFrame
        output_path: Destination CSV path
    """
    cols_to_save = [
        "resolved_name", "cardiotox_label", "DICT_Concern",
        "Trade Name", "Active Ingredient(s)", "Generic/Proper Name(s)",
    ]
    # Only keep columns that actually exist
    cols_to_save = [c for c in cols_to_save if c in df_labeled.columns]
    df_out = df_labeled[cols_to_save].reset_index(drop=True)
    df_out.to_csv(output_path, index=False)
    logger.info(f"  Saved {len(df_out)} rows to: {output_path}")


def data_quality_report(df_labeled: pd.DataFrame):
    """
    Print a concise data quality summary to log.
    Includes: shape, label balance, missing values, unique names.
    """
    logger.info("=" * 60)
    logger.info("DATA QUALITY REPORT")
    logger.info(f"  Total labeled compounds : {len(df_labeled)}")
    logger.info(f"  Positive (cardiotoxic)  : {(df_labeled['cardiotox_label'] == 1).sum()}")
    logger.info(f"  Negative (no concern)   : {(df_labeled['cardiotox_label'] == 0).sum()}")
    logger.info(f"  Unique resolved names   : {df_labeled['resolved_name'].nunique()}")
    logger.info(f"  Duplicate names         : {len(df_labeled) - df_labeled['resolved_name'].nunique()}")
    logger.info("=" * 60)


def main():
    logger.info("=" * 60)
    logger.info("SCRIPT: 01_fetch_labels.py")
    logger.info("PURPOSE: Load, clean, and export DICTrank labels")
    logger.info("=" * 60)

    ensure_dirs(CFG.PROCESSED_DIR, os.path.join(CFG.RESULTS_DIR, "logs"))

    # Step 1: Load raw file
    df = load_and_clean_dictrank(CFG.DICTRANK_PATH)

    # FDA discrepancy sanity check
    counts = df["DICT_Concern"].value_counts().to_dict()
    if counts.get("less", 0) != 527:
        logger.warning(f"Expected less=527, got {counts.get('less', 0)}. "
                       "Data version may differ from trace-through session.")

    # Step 2: Binarize
    df_labeled = binarize_labels(df)

    # Step 3: Name resolution check
    df_labeled = check_name_coverage(df_labeled)

    # Step 4: Data quality report
    data_quality_report(df_labeled)

    # Step 5: Save
    save_labeled_csv(df_labeled, CFG.LABELS_CSV)

    logger.info("01_fetch_labels.py COMPLETE")
    logger.info(f"Output: {CFG.LABELS_CSV}")
    return df_labeled


if __name__ == "__main__":
    main()
