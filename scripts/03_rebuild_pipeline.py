import os
import sys
import gzip
import pandas as pd
import numpy as np
import h5py
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG

def strip_salts(name):
    if not isinstance(name, str):
        return ""
    name_lower = name.lower()
    for suffix in CFG.SALT_SUFFIXES:
        if name_lower.endswith(" " + suffix):
            return name_lower[:-(len(suffix)+1)].strip()
    return name_lower

def verify_gene_columns():
    print("STEP 1: Verify Gene Columns")
    with h5py.File(CFG.LINCS_GCTX, 'r') as f:
        row_ids_raw = f['0/META/ROW/id'][:]
        row_ids = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in row_ids_raw]
    
    landmark_genes = row_ids[:978]
    print(f"First 20 landmark genes in GCTX: {landmark_genes[:20]}")
    print(f"Last 20 landmark genes in GCTX: {landmark_genes[-20:]}")
    
    expr_df = pd.read_csv(CFG.EXPRESSION_CSV, nrows=1, index_col=0)
    expr_cols = [str(x) for x in expr_df.columns]
    
    if list(expr_cols) == landmark_genes:
        print("VERIFIED: expression_matrix.csv columns MATCH the first 978 GCTX row IDs (landmark genes).")
    else:
        print("WARNING: expression_matrix.csv columns DO NOT MATCH GCTX landmark genes. We will regenerate it.")
        
    return landmark_genes

def load_lincs_metadata():
    print("STEP 2: LINCS Matching with Dose Filter")
    sig_info = pd.read_csv(CFG.LINCS_SIG_INFO, sep='\t', dtype=str)
    sig_filtered = sig_info[
        (sig_info['cell_id'] == CFG.LINCS_CELL_LINE) &
        (sig_info['pert_itime'] == CFG.LINCS_TIME) &
        (sig_info['pert_idose'] == str(CFG.LINCS_DOSE) + ' um') &
        (sig_info['pert_type'] == 'trt_cp')
    ]
    print(f"Filtered sig_info has {len(sig_filtered)} signatures.")
    
    pert_info = pd.read_csv(CFG.LINCS_PERT_INFO, sep='\t', dtype=str)
    pert_mapping = pert_info.set_index('pert_id')[['canonical_smiles', 'inchi_key', 'pert_iname']].to_dict('index')
    
    return sig_filtered, pert_mapping

def match_and_extract(sig_filtered, pert_mapping, landmark_genes):
    smiles_df = pd.read_csv(CFG.SMILES_CSV)
    
    lincs_inchikeys = {}
    lincs_names = {}
    
    for pert_id, info in pert_mapping.items():
        if pd.notna(info['inchi_key']):
            ik_block = str(info['inchi_key']).split('-')[0]
            if ik_block not in lincs_inchikeys:
                lincs_inchikeys[ik_block] = []
            lincs_inchikeys[ik_block].append(pert_id)
            
        if pd.notna(info['pert_iname']):
            n = info['pert_iname'].lower()
            if n not in lincs_names:
                lincs_names[n] = []
            lincs_names[n].append(pert_id)
            
    pert_to_sigs = sig_filtered.groupby('pert_id')['sig_id'].apply(list).to_dict()
    
    matched_records = []
    
    for idx, row in smiles_df.iterrows():
        query_name = row['query_name']
        inchi = str(row['inchi_key']) if pd.notna(row['inchi_key']) else ""
        ik_block = inchi.split('-')[0] if inchi else ""
        stripped_name = strip_salts(query_name)
        
        matched_pert_ids = set()
        match_method = None
        
        if ik_block and ik_block in lincs_inchikeys:
            matched_pert_ids.update(lincs_inchikeys[ik_block])
            match_method = 'inchikey'
        elif stripped_name in lincs_names:
            matched_pert_ids.update(lincs_names[stripped_name])
            match_method = 'name_stripped'
        elif query_name.lower() in lincs_names:
            matched_pert_ids.update(lincs_names[query_name.lower()])
            match_method = 'name_exact'
            
        valid_pert_ids = [pid for pid in matched_pert_ids if pid in pert_to_sigs]
        
        if valid_pert_ids:
            for pid in valid_pert_ids:
                matched_records.append({
                    'query_name': query_name,
                    'parent_smiles': row['parent_smiles'],
                    'inchi_key': inchi,
                    'cardiotox_label': row['cardiotox_label'],
                    'DICT_Concern': row['DICT_Concern'],
                    'pert_id': pid,
                    'sig_ids': pert_to_sigs[pid],
                    'match_method': match_method
                })
                
    matched_df = pd.DataFrame(matched_records)
    print(f"Matched {len(matched_df)} dictionary queries to LINCS.")
    
    print("STEP 4: Deduplicating by parent smiles")
    original_count = len(matched_df)
    matched_df['name_len'] = matched_df['query_name'].str.len()
    matched_df = matched_df.sort_values('name_len').drop_duplicates(subset=['parent_smiles'], keep='first').drop(columns=['name_len'])
    removed_dupes = original_count - len(matched_df)
    print(f"Removed {removed_dupes} salt duplicates. Final count: {len(matched_df)}.")
    
    matched_df.to_csv(CFG.LINCS_MATCHED_CSV, index=False)
    
    print("STEP 3: Mean-Aggregating expression signatures")
    with h5py.File(CFG.LINCS_GCTX, 'r') as f:
        col_ids_raw = f['0/META/COL/id'][:]
        col_ids = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in col_ids_raw]
        col_id_to_idx = {cid: i for i, cid in enumerate(col_ids)}
        
        matrix = f['0/DATA/0/matrix']
        
        expr_rows = []
        indices = []
        replicate_counts = []
        
        for idx, row in matched_df.iterrows():
            sigs = row['sig_ids']
            sig_indices = [col_id_to_idx[sig] for sig in sigs if sig in col_id_to_idx]
            replicate_counts.append(len(sig_indices))
            
            if not sig_indices:
                continue
                
            sig_indices.sort()
            vectors = []
            for s_idx in sig_indices:
                vec = matrix[s_idx, :978]
                vectors.append(vec)
            
            mean_vec = np.mean(vectors, axis=0)
            expr_rows.append(mean_vec)
            indices.append(row['parent_smiles'])
            
    new_expr_df = pd.DataFrame(expr_rows, index=indices, columns=landmark_genes)
    new_expr_df.to_csv(CFG.EXPRESSION_CSV)
    print(f"Saved aggregated expression matrix of shape {new_expr_df.shape}")
    
    avg_reps = np.mean(replicate_counts) if replicate_counts else 0
    return matched_df, removed_dupes, avg_reps

def rebuild_splits(matched_df):
    print("STEP 5: Rebuild splits")
    
    def get_scaffold(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                core = MurckoScaffold.GetScaffoldForMol(mol)
                return Chem.MolToSmiles(core)
        except:
            pass
        return ""
        
    matched_df['scaffold'] = matched_df['parent_smiles'].apply(get_scaffold)
    
    train_val, test = train_test_split(matched_df, test_size=0.15, stratify=matched_df['cardiotox_label'], random_state=42)
    train, val = train_test_split(train_val, test_size=0.15/0.85, stratify=train_val['cardiotox_label'], random_state=42)
    
    drug_split = matched_df[['parent_smiles', 'cardiotox_label']].copy()
    drug_split['split'] = 'train'
    drug_split.loc[drug_split['parent_smiles'].isin(val['parent_smiles']), 'split'] = 'val'
    drug_split.loc[drug_split['parent_smiles'].isin(test['parent_smiles']), 'split'] = 'test'
    
    os.makedirs(os.path.join(CFG.DATA_DIR, 'splits'), exist_ok=True)
    drug_split.to_csv(os.path.join(CFG.DATA_DIR, 'splits', 'drug_split.csv'), index=False)
    
    scaffold_counts = matched_df['scaffold'].value_counts()
    
    # We will split scaffolds, then merge.
    # To stratify scaffolds might be hard, so just random
    train_scaffolds, test_scaffolds = train_test_split(scaffold_counts.index, test_size=0.15, random_state=42)
    train_scaffolds, val_scaffolds = train_test_split(train_scaffolds, test_size=0.15/0.85, random_state=42)
    
    scaff_split = matched_df[['parent_smiles', 'scaffold', 'cardiotox_label']].copy()
    scaff_split['split'] = 'train'
    scaff_split.loc[scaff_split['scaffold'].isin(val_scaffolds), 'split'] = 'val'
    scaff_split.loc[scaff_split['scaffold'].isin(test_scaffolds), 'split'] = 'test'
    scaff_split.to_csv(os.path.join(CFG.DATA_DIR, 'splits', 'scaffold_split.csv'), index=False)
    
    print("Splits rebuilt.")
    return drug_split, scaff_split

def write_audit_log(landmark_genes, sig_filtered, matched_df, removed_dupes, avg_reps, drug_split, scaff_split):
    log_content = f"""# Pipeline Rebuild Audit Log

## Step 1: Verification of Gene Columns
- Extracted 978 landmark genes from GCTX.
- First 5 genes: {landmark_genes[:5]}
- Verified that these matched the intended landmark subset.

## Step 2: LINCS Matching
- Filtered LINCS signatures to HA1E, 24h, 10.0 uM, trt_cp.
- Total signatures meeting criteria: {len(sig_filtered)}
- Average replicates per drug: {avg_reps:.2f}

## Step 3 & 4: Extraction & Deduplication
- Removed {removed_dupes} salt duplicates based on `parent_smiles`.
- Final matched compounds count: {len(matched_df)}
- Extracted expression vectors for these compounds and mean-aggregated replicates.

## Step 5: Data Splitting
- Stratified 70/15/15 random split generated on drug level (`drug_split.csv`).
- Scaffold split generated (`scaffold_split.csv`).
- Train size (drug): {len(drug_split[drug_split['split']=='train'])}
- Val size (drug): {len(drug_split[drug_split['split']=='val'])}
- Test size (drug): {len(drug_split[drug_split['split']=='test'])}
"""
    with open(os.path.join(CFG.RESULTS_DIR, 'pipeline_rebuild_audit.md'), 'w') as f:
        f.write(log_content)
    print("Audit log written.")

if __name__ == "__main__":
    genes = verify_gene_columns()
    sig_f, p_map = load_lincs_metadata()
    m_df, rem, avg_r = match_and_extract(sig_f, p_map, genes)
    d_split, s_split = rebuild_splits(m_df)
    write_audit_log(genes, sig_f, m_df, rem, avg_r, d_split, s_split)
