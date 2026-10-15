"""
03_rebuild_pipeline.py -- LINCS matching, landmark gene extraction, skeleton deduplication, 70/15/15 splits.

Rules enforced here (each is re-checked independently by scripts/verify_pipeline.py):
  * Strict cohort: HA1E, 24 h, 10.0 uM, trt_cp; replicate Level-5 z-vectors are MEAN-aggregated.
  * Landmark genes are taken from gene_info (pr_is_lm == 1), mapped to their real GCTX row indices.
  * One compound per 14-char InChIKey connectivity block (charge-neutralised).
  * One compound per LINCS pert_id: a LINCS perturbagen's expression may be attached to only ONE cohort
    compound (InChIKey matches win over name matches). Otherwise two different cohort entries (e.g. a drug
    and its phosphate prodrug) would share an identical expression vector, possibly across train/test and
    with opposite labels.
  * Scaffold split: Murcko without chirality, acyclic compounds get unique pseudo-scaffolds, packing by
    compound count.
"""
import os
import sys
import gzip
import json
import pandas as pd
import numpy as np
import h5py
from rdkit import Chem
from rdkit.Chem import MolStandardize, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from scripts.utils import random_classifier_aucpr_interval

uncharger = MolStandardize.rdMolStandardize.Uncharger()
FORMULA_REJECTED = []   # (query_name, pert_id, formula_dictrank, formula_lincs) name matches refused for a formula mismatch


def _formula(smi):
    """Molecular formula of the largest fragment after neutralisation (None if unparseable)."""
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    frag = max(Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumHeavyAtoms())
    return rdMolDescriptors.CalcMolFormula(uncharger.uncharge(frag))


def get_inchikey_block(smi):
    if not isinstance(smi, str) or not smi:
        return ""
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return ""
        neutral_mol = uncharger.uncharge(mol)
        ik = Chem.MolToInchiKey(neutral_mol)
        return ik.split('-')[0] if ik else ""
    except Exception:
        return ""

def strip_salts(name):
    if not isinstance(name, str):
        return ""
    name_lower = name.lower().strip()
    for suffix in CFG.SALT_SUFFIXES:
        if name_lower.endswith(" " + suffix):
            return name_lower[:-(len(suffix)+1)].strip()
    return name_lower

def verify_landmark_genes():
    print("=" * 70)
    print("STEP 1: LANDMARK GENE VERIFICATION (pr_is_lm == 1)")
    print("=" * 70)
    
    gene_info_path = os.path.join(CFG.LINCS_RAW_DIR, "gene_info.txt.gz")
    if not os.path.exists(gene_info_path):
        raise FileNotFoundError(f"Missing LINCS gene_info at {gene_info_path}")
        
    with gzip.open(gene_info_path, 'rt') as f:
        df_gene = pd.read_csv(f, sep='\t', dtype=str)
        
    lm_genes_info = set(df_gene[df_gene['pr_is_lm'] == '1']['pr_gene_id'].tolist())
    print(f"Verified total landmark genes in gene_info.txt.gz: {len(lm_genes_info)}")
    
    # Read GCTX row IDs
    with h5py.File(CFG.LINCS_GCTX, 'r') as f:
        row_ids_raw = f['0/META/ROW/id'][:]
        gctx_row_ids = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in row_ids_raw]
        
    print(f"Total row IDs in GCTX file: {len(gctx_row_ids)}")
    
    # Find exact indices in GCTX corresponding to pr_is_lm == 1
    landmark_indices = [i for i, gid in enumerate(gctx_row_ids) if gid in lm_genes_info]
    landmark_gene_ids = [gctx_row_ids[i] for i in landmark_indices]
    
    print(f"Found {len(landmark_indices)} landmark gene rows in GCTX file.")
    print(f"Landmark row indices range: min={min(landmark_indices)}, max={max(landmark_indices)}")
    
    if landmark_indices == list(range(978)):
        print("CONFIRMED: Landmark genes occupy contiguous indices 0..977 in GCTX.")
    else:
        print("NOTICE: Landmark genes are scattered across GCTX rows. Using verified landmark index mapping.")
        
    return landmark_indices, landmark_gene_ids

def load_lincs_signatures():
    print("\n" + "=" * 70)
    print("STEP 2: STRICT LINCS METADATA MATCHING (HA1E, 24 h, 10.0 µM)")
    print("=" * 70)
    
    sig_info = pd.read_csv(CFG.LINCS_SIG_INFO, sep='\t', dtype=str)
    sig_filtered = sig_info[
        (sig_info['cell_id'] == CFG.LINCS_CELL_LINE) &
        (sig_info['pert_itime'] == CFG.LINCS_TIME) &
        (sig_info['pert_idose'] == str(CFG.LINCS_DOSE) + ' um') &
        (sig_info['pert_type'] == 'trt_cp')
    ]
    print(f"Filtered sig_info has {len(sig_filtered)} signatures matching strict condition.")
    sig_any = sig_info[sig_info['pert_type'] == 'trt_cp']
    
    pert_info = pd.read_csv(CFG.LINCS_PERT_INFO, sep='\t', dtype=str)
    pert_mapping = pert_info.set_index('pert_id')[['canonical_smiles', 'inchi_key', 'pert_iname']].to_dict('index')
    
    return sig_filtered, sig_any, pert_mapping

def match_compounds(sig_filtered, pert_mapping, verbose=True):
    smiles_df = pd.read_csv(CFG.SMILES_CSV)
    # Filter out empty parent_smiles
    smiles_df = smiles_df[smiles_df['parent_smiles'].notna() & (smiles_df['parent_smiles'] != '')].copy()
    if verbose:
        print(f"DICTrank compounds with valid SMILES: {len(smiles_df)}")
    
    lincs_inchikeys = {}
    lincs_names = {}
    
    for pert_id, info in pert_mapping.items():
        if pd.notna(info['inchi_key']):
            ik_block = str(info['inchi_key']).split('-')[0]
            if ik_block not in lincs_inchikeys:
                lincs_inchikeys[ik_block] = []
            lincs_inchikeys[ik_block].append(pert_id)
            
        if pd.notna(info['pert_iname']):
            n = info['pert_iname'].lower().strip()
            if n not in lincs_names:
                lincs_names[n] = []
            lincs_names[n].append(pert_id)
            
    pert_to_sigs = sig_filtered.groupby('pert_id')['sig_id'].apply(list).to_dict()
    
    matched_records = []
    for idx, row in smiles_df.iterrows():
        query_name = row['query_name']
        parent_smi = row['parent_smiles']
        ik_block = get_inchikey_block(parent_smi)
        stripped_name = strip_salts(query_name)
        
        matched_pert_ids = set()
        match_method = None
        
        if ik_block and ik_block in lincs_inchikeys:
            matched_pert_ids.update(lincs_inchikeys[ik_block])
            match_method = 'inchikey'
        elif stripped_name in lincs_names:
            matched_pert_ids.update(lincs_names[stripped_name])
            match_method = 'name_stripped'
        elif query_name.lower().strip() in lincs_names:
            matched_pert_ids.update(lincs_names[query_name.lower().strip()])
            match_method = 'name_exact'
            
        # Deterministic choice: a compound can have several LINCS pert_ids (batches). Order by number of strict-cohort
        # signatures (most first), then pert_id, so the result never depends on set/hash order.
        valid_pert_ids = sorted((pid for pid in matched_pert_ids if pid in pert_to_sigs),
                                key=lambda pid: (-len(pert_to_sigs[pid]), pid))
        # Name-based matches are accepted only when LINCS's structure for the perturbagen has the SAME molecular
        # formula (tautomer / salt / stereo differences are fine). A prodrug, ester or mixture component is a
        # different molecule, so its expression must not stand in for the DICTrank structure.
        if match_method != 'inchikey' and valid_pert_ids:
            f_d, keep = _formula(parent_smi), []
            for pid in valid_pert_ids:
                f_l = _formula(pert_mapping[pid]['canonical_smiles'])
                if f_d is not None and f_l == f_d:
                    keep.append(pid)
                elif verbose:
                    FORMULA_REJECTED.append((query_name, pid, f_d, f_l))
            valid_pert_ids = keep
        
        if valid_pert_ids:
            for pid in valid_pert_ids:
                matched_records.append({
                    'query_name': query_name,
                    'parent_smiles': parent_smi,
                    'inchi_key': row.get('inchi_key', ''),
                    'inchikey_block': ik_block,
                    'cardiotox_label': row['cardiotox_label'],
                    'DICT_Concern': row['DICT_Concern'],
                    'pert_id': pid,
                    'sig_ids': pert_to_sigs[pid],
                    'match_method': match_method
                })
                
    df_matched = pd.DataFrame(matched_records)
    if verbose:
        print(f"Matched {len(df_matched)} raw dictionary entries to LINCS signatures.")
    return df_matched

def deduplicate_and_resolve_conflicts(df_matched):
    print("\n" + "=" * 70)
    print("STEP 3: INCHIKEY CONNECTIVITY SKELETON DEDUPLICATION & LABEL CONFLICT RESOLUTION")
    print("=" * 70)
    
    # 1. Resolve label conflicts across identical InChIKey connectivity blocks
    # Rule: If any salt/form of a drug skeleton is flagged toxic (label 1), the skeleton is assigned label 1.
    block_labels = {}
    conflicts = []
    for block, grp in df_matched.groupby('inchikey_block'):
        labels = grp['cardiotox_label'].unique()
        if len(labels) > 1:
            resolved_label = 1  # Toxic salt flags cardiotoxicity risk
            conflicts.append(grp['query_name'].tolist())
            print(f"Resolved label conflict in InChIKey skeleton '{block}' ({grp['query_name'].tolist()}) -> Assigned Label 1")
        else:
            resolved_label = labels[0]
        block_labels[block] = resolved_label
        
    df_matched['cardiotox_label'] = df_matched['inchikey_block'].map(block_labels)
    
    # 2. Keep 1 representative compound per 14-character InChIKey connectivity block
    df_matched['name_len'] = df_matched['query_name'].str.len()
    df_blocks = df_matched.sort_values('name_len', kind='mergesort').drop_duplicates(subset=['inchikey_block'], keep='first').reset_index(drop=True)
    n_blocks = len(df_blocks)

    # 3. One cohort compound per LINCS pert_id (InChIKey match beats name match, then shorter name).
    #    Without this, e.g. 'etoposide phosphate' (name-matched) re-used etoposide's signature and the two
    #    entries landed in different splits with identical expression vectors.
    df_blocks['_priority'] = (df_blocks['match_method'] != 'inchikey').astype(int)
    df_blocks = df_blocks.sort_values(['_priority', 'name_len'], kind='mergesort').reset_index(drop=True)
    dup_mask = df_blocks.duplicated(subset=['pert_id'], keep='first')
    pert_dropped = df_blocks[dup_mask][['query_name', 'pert_id', 'cardiotox_label', 'match_method']].copy()
    kept_for = df_blocks.drop_duplicates('pert_id').set_index('pert_id')['query_name']
    pert_dropped['kept_instead'] = pert_dropped['pert_id'].map(kept_for)
    for _, r in pert_dropped.iterrows():
        print(f"Dropped '{r.query_name}' (label {r.cardiotox_label}, {r.match_method}): LINCS {r.pert_id} already used by '{r.kept_instead}'")
    df_clean = df_blocks[~dup_mask].drop(columns=['name_len', '_priority']).reset_index(drop=True)

    print(f"Final clean dataset: {len(df_clean)} unique chemical skeletons (14-char InChIKey blocks, one LINCS pert_id each).")
    print(f"Label balance: {df_clean['cardiotox_label'].value_counts().to_dict()}")
    info = {'conflicts': conflicts, 'n_blocks': n_blocks, 'pert_dropped': pert_dropped}
    return df_clean, info

def extract_and_aggregate_expression(df_clean, landmark_indices, landmark_gene_ids):
    print("\n" + "=" * 70)
    print("STEP 4: MEAN-AGGREGATING VERIFIED LANDMARK EXPRESSION VECTORS")
    print("=" * 70)
    
    with h5py.File(CFG.LINCS_GCTX, 'r') as f:
        col_ids_raw = f['0/META/COL/id'][:]
        col_ids = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in col_ids_raw]
        col_id_to_idx = {cid: i for i, cid in enumerate(col_ids)}
        
        matrix = f['0/DATA/0/matrix']  # shape: (samples, genes) = (118050, 12328)
        
        expr_rows = []
        indices = []
        
        for idx, row in df_clean.iterrows():
            sigs = row['sig_ids']
            sig_indices = [col_id_to_idx[s] for s in sigs if s in col_id_to_idx]
            
            if not sig_indices:
                continue
                
            sig_indices.sort()
            vectors = []
            for s_idx in sig_indices:
                # Extract ONLY verified landmark gene indices
                row_vec = matrix[s_idx, landmark_indices]
                vectors.append(row_vec)
                
            mean_vec = np.mean(vectors, axis=0)
            expr_rows.append(mean_vec)
            indices.append(row['parent_smiles'])
            
    df_expr = pd.DataFrame(expr_rows, index=indices, columns=landmark_gene_ids)
    df_expr.to_csv(CFG.EXPRESSION_CSV)
    df_clean.to_csv(CFG.LINCS_MATCHED_CSV, index=False)
    print(f"Saved aggregated landmark expression matrix to {CFG.EXPRESSION_CSV} with shape {df_expr.shape}")

def generate_splits(df_clean):
    print("\n" + "=" * 70)
    print("STEP 5: REBUILD LEAK-FREE 70/15/15 DRUG & SCAFFOLD SPLITS")
    print("=" * 70)
    
    # --- A. Drug-Level Split (Stratified 70/15/15 on unique InChIKey blocks) ---
    train_val, test = train_test_split(df_clean, test_size=0.15, stratify=df_clean['cardiotox_label'], random_state=CFG.RANDOM_SEED)
    train, val = train_test_split(train_val, test_size=0.15/0.85, stratify=train_val['cardiotox_label'], random_state=CFG.RANDOM_SEED)
    
    df_clean['drug_split'] = 'train'
    df_clean.loc[df_clean['inchikey_block'].isin(val['inchikey_block']), 'drug_split'] = 'val'
    df_clean.loc[df_clean['inchikey_block'].isin(test['inchikey_block']), 'drug_split'] = 'test'
    
    # --- B. Scaffold-Level Split (Murcko WITHOUT Chirality + Distinct Acyclics + Greedy Compound-Proportional Packing) ---
    def get_nonchiral_scaffold(smi, idx):
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
            if scaf:
                return scaf
        except:
            pass
        return f"acyclic_{idx}"
        
    df_clean['scaffold'] = [get_nonchiral_scaffold(s, i) for i, s in enumerate(df_clean['parent_smiles'])]
    
    # Group compounds by scaffold
    scaf_groups = df_clean.groupby('scaffold')
    scaf_counts = scaf_groups.size().sort_values(ascending=False, kind='mergesort')
    
    total_compounds = len(df_clean)
    train_target = int(total_compounds * 0.70)
    val_target = int(total_compounds * 0.15)
    test_target = total_compounds - train_target - val_target
    
    train_scafs, val_scafs, test_scafs = set(), set(), set()
    train_c, val_c, test_c = 0, 0, 0
    
    for scaf, count in scaf_counts.items():
        if train_c + count <= train_target or (train_c < train_target and val_c >= val_target and test_c >= test_target):
            train_scafs.add(scaf)
            train_c += count
        elif val_c + count <= val_target or (val_c < val_target and test_c >= test_target):
            val_scafs.add(scaf)
            val_c += count
        else:
            test_scafs.add(scaf)
            test_c += count
            
    df_clean['scaffold_split'] = 'train'
    df_clean.loc[df_clean['scaffold'].isin(val_scafs), 'scaffold_split'] = 'val'
    df_clean.loc[df_clean['scaffold'].isin(test_scafs), 'scaffold_split'] = 'test'
    
    # Save Split CSVs
    cols = ['query_name', 'cardiotox_label', 'DICT_Concern', 'parent_smiles', 'pert_id', 'inchikey_block', 'scaffold', 'match_method']
    
    drug_split_df = df_clean[cols + ['drug_split']].rename(columns={'drug_split': 'split'})
    scaf_split_df = df_clean[cols + ['scaffold_split']].rename(columns={'split': 'scaffold_split_old'}).rename(columns={'scaffold_split': 'split'})
    
    os.makedirs(os.path.join(CFG.DATA_DIR, 'splits'), exist_ok=True)
    drug_split_df.to_csv(os.path.join(CFG.DATA_DIR, 'splits', 'drug_split.csv'), index=False)
    scaf_split_df.to_csv(os.path.join(CFG.DATA_DIR, 'splits', 'scaffold_split.csv'), index=False)
    
    print("\n--- DRUG SPLIT COMPOUND COUNTS ---")
    print(drug_split_df['split'].value_counts())
    print("\n--- SCAFFOLD SPLIT COMPOUND COUNTS ---")
    print(scaf_split_df['split'].value_counts())
    
    return drug_split_df, scaf_split_df

def write_audit_log(landmark_gene_ids, df_clean, drug_df, scaf_df, funnel, info):
    log_path = os.path.join(CFG.RESULTS_DIR, 'pipeline_rebuild_audit.md')
    os.makedirs(CFG.RESULTS_DIR, exist_ok=True)

    def blocks(df, s):
        return set(df[df['split'] == s]['inchikey_block'])
    leak = {}
    for name, df in [('drug', drug_df), ('scaffold', scaf_df)]:
        leak[name] = (len(blocks(df, 'train') & blocks(df, 'test')), len(blocks(df, 'train') & blocks(df, 'val')),
                      len(blocks(df, 'val') & blocks(df, 'test')))
    test = scaf_df[scaf_df['split'] == 'test']
    y_test = test['cardiotox_label'].values
    lo, hi = random_classifier_aucpr_interval(y_test, seed=CFG.RANDOM_SEED)
    n_acyc = int(scaf_df['scaffold'].str.startswith('acyclic_').sum())
    conf = "; ".join(" / ".join(c) for c in info['conflicts']) or "none"
    dropped = "; ".join(f"{r.query_name} (label {r.cardiotox_label}, {r.match_method}) -> LINCS {r.pert_id} already used by {r.kept_instead}"
                        for _, r in info['pert_dropped'].iterrows()) or "none"
    n_neg = int((df_clean['cardiotox_label'] == 0).sum())

    content = f"""# Pipeline Rebuild & Data Audit Log (auto-generated by scripts/03_rebuild_pipeline.py)

## Cohort funnel
| Stage | Count |
|---|---|
| Labeled DICTrank compounds (ambiguous dropped) | {funnel['labeled']} |
| ...with a resolved structure (SMILES) | {funnel['with_smiles']} |
| ...unique structures (14-char InChIKey blocks) | {funnel['smiles_blocks']} |
| Unique skeletons with ANY LINCS trt_cp signature (any cell line / dose / time) | {funnel['any_blocks']} |
| Unique skeletons with a STRICT signature (HA1E / {CFG.LINCS_DOSE} uM / {CFG.LINCS_TIME}) before pert_id dedup | {info['n_blocks']} |
| **Final strict cohort (one LINCS pert_id each)** | **{len(df_clean)}** ({int(df_clean['cardiotox_label'].sum())} toxic / {n_neg} safe) |

## Landmark genes
- Source: `gene_info.txt.gz`, `pr_is_lm == 1`: {len(landmark_gene_ids)} genes, mapped to their real GCTX row indices.

## Deduplication and label conflicts
- Formal charges neutralised with RDKit `Uncharger`; grouping on the 14-character InChIKey connectivity block.
- Label-conflict skeletons (rule: any toxic form -> label 1): {conf}
- Dropped because their LINCS perturbagen was already attached to another cohort compound: {dropped}
- Cross-split skeleton overlap (train/test, train/val, val/test): drug split {leak['drug']}, scaffold split {leak['scaffold']}

## Scaffold split
- Murcko scaffolds without chirality; {n_acyc} acyclic compounds each given a unique pseudo-scaffold id; greedy packing by compound count.
- Counts: train {len(scaf_df[scaf_df['split']=='train'])}, val {len(scaf_df[scaf_df['split']=='val'])}, test {len(test)}.

## Statistical power
- Scaffold test set: {len(test)} compounds, {int((y_test == 0).sum())} safe (prevalence {y_test.mean():.4f}).
- Random-classifier AUC-PR 95% bootstrap interval on this test set: [{lo:.3f}, {hi:.3f}].
"""
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\nSaved updated audit log to {log_path}")

def main():
    lm_indices, lm_gene_ids = verify_landmark_genes()
    sig_f, sig_any, p_map = load_lincs_signatures()
    df_m = match_compounds(sig_f, p_map)
    df_any = match_compounds(sig_any, p_map, verbose=False)
    smiles_df = pd.read_csv(CFG.SMILES_CSV)
    valid = smiles_df[smiles_df['parent_smiles'].notna() & (smiles_df['parent_smiles'] != '')]
    funnel = {
        'labeled': len(smiles_df),
        'with_smiles': len(valid),
        'smiles_blocks': valid['parent_smiles'].map(get_inchikey_block).nunique(),
        'any_blocks': df_any['inchikey_block'].nunique(),
    }
    print(f"Coverage funnel: {funnel}")
    df_c, info = deduplicate_and_resolve_conflicts(df_m)
    extract_and_aggregate_expression(df_c, lm_indices, lm_gene_ids)
    drug_df, scaf_df = generate_splits(df_c)
    write_audit_log(lm_gene_ids, df_c, drug_df, scaf_df, funnel, info)
    with open(os.path.join(CFG.RESULTS_DIR, 'pipeline_rebuild_audit.md'), 'a', encoding='utf-8') as f:
        f.write("\n## Name matches rejected for a molecular-formula mismatch\n")
        f.write("A name-based LINCS match is accepted only if LINCS's structure has the same molecular formula as the DICTrank structure.\n\n")
        seen = set()
        for q, pid, fd, fl in FORMULA_REJECTED:
            if (q, pid) not in seen:
                seen.add((q, pid)); f.write(f"- REJECTED {q} (LINCS {pid}): DICTrank {fd} vs LINCS {fl}\n")
        if not seen:
            f.write("None.\n")
    print("\n" + "=" * 70)
    print("PIPELINE REBUILD COMPLETE & VERIFIED")
    print("=" * 70)

if __name__ == "__main__":
    main()
