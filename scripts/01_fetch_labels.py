"""
01_fetch_labels.py -- Load raw FDA DICTrank Excel, binarize labels, drop ambiguous.
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG

def main():
    print("=" * 70)
    print("STEP 1: FETCH & PROCESS DICTRANK LABELS")
    print("=" * 70)
    
    excel_path = CFG.DICTRANK_PATH
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"DICTrank file not found at {excel_path}")
        
    df_raw = pd.read_excel(excel_path)
    print(f"Raw DICTrank rows: {len(df_raw)}")
    print(f"Raw columns: {df_raw.columns.tolist()}")
    
    # Identify drug name and concern columns
    name_col = None
    concern_col = None
    for col in df_raw.columns:
        col_lower = str(col).lower()
        if 'trade' in col_lower or 'drug' in col_lower or 'name' in col_lower or 'active' in col_lower:
            if not name_col:
                name_col = col
        if 'concern' in col_lower or 'dict' in col_lower or 'class' in col_lower:
            concern_col = col
            
    if not name_col:
        name_col = df_raw.columns[0]
    if not concern_col:
        concern_col = df_raw.columns[-1]
        
    print(f"Using Name Column: '{name_col}', Concern Column: '{concern_col}'")
    
    df_clean = df_raw[[name_col, concern_col]].copy()
    df_clean.columns = ['resolved_name', 'DICT_Concern']
    df_clean['resolved_name'] = df_clean['resolved_name'].astype(str).str.strip()
    df_clean['DICT_Concern'] = df_clean['DICT_Concern'].astype(str).str.strip().str.lower()
    
    print("\nRaw Concern counts:")
    print(df_clean['DICT_Concern'].value_counts())
    
    # Drop ambiguous
    df_filtered = df_clean[df_clean['DICT_Concern'] != CFG.DROP_LABEL].copy()
    
    # Binarize
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
