"""
02_resolve_smiles.py -- Resolve SMILES, neutralise charges, extract 14-char InChIKey connectivity blocks.
"""
import os
import sys
import json
import urllib.request
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolStandardize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG

def strip_salts(name):
    if not isinstance(name, str):
        return ""
    name_lower = name.lower().strip()
    for suffix in CFG.SALT_SUFFIXES:
        if name_lower.endswith(" " + suffix):
            return name_lower[:-(len(suffix)+1)].strip()
    return name_lower

def get_neutral_smiles_and_inchikey(smi):
    if not isinstance(smi, str) or not smi:
        return "", "", ""
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return "", "", ""
        
        # Strip salts/fragments (keep largest fragment)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            
        # Neutralise charges
        uncharger = MolStandardize.rdMolStandardize.Uncharger()
        neutral_mol = uncharger.uncharge(mol)
        
        parent_smi = Chem.MolToSmiles(neutral_mol)
        ik = Chem.MolToInchiKey(neutral_mol)
        ik_block = ik.split('-')[0] if ik else ""
        return parent_smi, ik, ik_block
    except Exception:
        return "", "", ""

def main():
    print("=" * 70)
    print("STEP 2: RESOLVE SMILES & NEUTRALISE CHARGES")
    print("=" * 70)
    
    labels_df = pd.read_csv(CFG.LABELS_CSV)
    print(f"Loaded {len(labels_df)} labeled compounds.")
    
    # Pre-resolved dictionary for known manual overrides / LINCS matched drugs
    overrides = {
        'FULVESTRANT': ('C[C@]12CC[C@@H]3c4ccc(O)cc4C[C@@H](CCCCCCS(=O)CCCC(F)(F)C(F)(F)F)[C@H]3[C@@H]1CC[C@@H]2O', 'VWUXBMIQPBEWFH-GXZKXCMSSA-N'),
        'IXABEPILONE': ('C/C(=C\\c1csc(C)n1)[C@@H]1C[C@@H]2O[C@]2(C)CCC(=O)[C@H](C)[C@H](O)[C@(C)(C)C(=O)NCCC1', 'FABUFPQFXZVHFB-PVYNADRNSA-N'),
        'IVERMECTIN': ('CC[C@H](C)[C@H]1O[C@]2(CC[C@@H]1C)C[C@@H]3C[C@@H](C/C=C(\\C)[C@@H](O)[C@@H]4C/C=C/C=C5\\CO[C@@H]6[C@@]5(O)C(=O)[C@@H](C(=C4)C)O6)O[C@@H](C3)O[C@@H]7C[C@@H](OC)[C@H](O[C@@H]8C[C@@H](OC)[C@H](O)[C@@H](C)O8)[C@@H](C)O7', 'CCNVFFXHYWBOGH-XLEFBCLLSA-N'),
    }
    
    # Load checkpoint if exists to avoid unnecessary ChEMBL requests
    checkpoint_path = os.path.join(CFG.PROCESSED_DIR, "smiles_checkpoint.json")
    cache = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached SMILES resolutions.")
        
    records = []
    resolved_count = 0
    
    for idx, row in labels_df.iterrows():
        query_name = row['resolved_name']
        upper_name = query_name.upper()
        
        raw_smi = None
        ik_raw = None
        
        # Check override
        if upper_name in overrides:
            raw_smi, ik_raw = overrides[upper_name]
        elif query_name in cache:
            raw_smi = cache[query_name].get('raw_smiles')
            ik_raw = cache[query_name].get('inchi_key')
        else:
            # Query ChEMBL API
            stripped = strip_salts(query_name)
            url = f"{CFG.CHEMBL_API_BASE}/molecule/search.json?q={urllib.parse.quote(stripped)}"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=CFG.CHEMBL_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    mols = data.get('molecules', [])
                    if mols:
                        structs = mols[0].get('molecule_structures')
                        if structs and structs.get('canonical_smiles'):
                            raw_smi = structs['canonical_smiles']
                            ik_raw = structs.get('standard_inchi_key')
                            cache[query_name] = {'raw_smiles': raw_smi, 'inchi_key': ik_raw}
            except Exception:
                pass
                
        parent_smi, ik, ik_block = get_neutral_smiles_and_inchikey(raw_smi) if raw_smi else ("", "", "")
        
        if parent_smi:
            resolved_count += 1
            
        records.append({
            'query_name': query_name,
            'cardiotox_label': row['cardiotox_label'],
            'DICT_Concern': row['DICT_Concern'],
            'raw_smiles': raw_smi or "",
            'parent_smiles': parent_smi,
            'inchi_key': ik or ik_raw or "",
            'inchikey_block': ik_block
        })
        
    # Save cache
    with open(checkpoint_path, 'w') as f:
        json.dump(cache, f)
        
    df_out = pd.DataFrame(records)
    print(f"Resolved SMILES for {resolved_count} / {len(df_out)} compounds.")
    print(f"Unique parent_smiles: {df_out[df_out['parent_smiles'] != '']['parent_smiles'].nunique()}")
    print(f"Unique 14-char InChIKey connectivity blocks: {df_out[df_out['inchikey_block'] != '']['inchikey_block'].nunique()}")
    
    df_out.to_csv(CFG.SMILES_CSV, index=False)
    print(f"Saved compounds with SMILES to {CFG.SMILES_CSV}")
    print("=" * 70)

if __name__ == "__main__":
    main()
