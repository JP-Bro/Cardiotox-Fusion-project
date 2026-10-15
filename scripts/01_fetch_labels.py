"""
01_fetch_labels.py -- Load raw FDA DICTrank Excel, binarize labels, drop ambiguous.

Drug-name column (matches how data/processed/compounds_with_smiles.csv was actually built):
  resolved_name = 'Active Ingredient(s)', falling back to 'Generic/Proper Name(s)' for the
  27 rows where the active ingredient is blank (these are mostly withdrawn "Most-Concern" drugs such as
  terfenadine, astemizole, cisapride, sertindole -- dropping them would silently remove known toxic
  drugs). 'Trade Name' is NOT used: it is brand-specific and blank for those 27 rows.

Columns are selected by EXACT name (no substring guessing) so a changed spreadsheet fails loudly.
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG

ACTIVE_COL = 'Active Ingredient(s)'
GENERIC_COL = 'Generic/Proper Name(s)'
CONCERN_COL = 'DICT _ Concern'


def main():
    print("=" * 70)
    print("STEP 1: FETCH & PROCESS DICTRANK LABELS")
    print("=" * 70)

    if not os.path.exists(CFG.DICTRANK_PATH):
        raise FileNotFoundError(f"DICTrank file not found at {CFG.DICTRANK_PATH}")

    df_raw = pd.read_excel(CFG.DICTRANK_PATH)
    print(f"Raw DICTrank rows: {len(df_raw)}")
    missing = [c for c in (ACTIVE_COL, GENERIC_COL, CONCERN_COL) if c not in df_raw.columns]
    if missing:
        raise KeyError(f"Expected columns not found in DICTrank file: {missing}. Found: {df_raw.columns.tolist()}")

    name = df_raw[ACTIVE_COL].fillna(df_raw[GENERIC_COL]).astype(str).str.strip()
    n_fallback = int(df_raw[ACTIVE_COL].isna().sum())
    print(f"Name column: '{ACTIVE_COL}' with fallback to '{GENERIC_COL}' for {n_fallback} blank rows")
    assert not name.isin(['', 'nan']).any(), "Some compounds still have no name after fallback"

    df_clean = pd.DataFrame({
        'resolved_name': name,
        'DICT_Concern': df_raw[CONCERN_COL].astype(str).str.strip().str.lower(),
    })

    print("\nRaw Concern counts:")
    print(df_clean['DICT_Concern'].value_counts())
    unexpected = set(df_clean['DICT_Concern']) - set(CFG.LABEL_MAP) - {CFG.DROP_LABEL}
    if unexpected:
        raise ValueError(f"Unexpected DICT concern values: {unexpected}")

    df_filtered = df_clean[df_clean['DICT_Concern'] != CFG.DROP_LABEL].copy()
    df_filtered['cardiotox_label'] = df_filtered['DICT_Concern'].map(CFG.LABEL_MAP)

    print(f"\nFiltered non-ambiguous rows: {len(df_filtered)}")
    print("Binarized Label counts (1=Toxic, 0=Safe):")
    print(df_filtered['cardiotox_label'].value_counts())

    os.makedirs(CFG.PROCESSED_DIR, exist_ok=True)
    df_filtered.to_csv(CFG.LABELS_CSV, index=False)
    print(f"\nSaved labeled compounds to {CFG.LABELS_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
