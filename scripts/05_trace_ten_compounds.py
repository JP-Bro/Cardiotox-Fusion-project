"""
05_trace_ten_compounds.py

MANDATORY PRE-PIPELINE TRACE-THROUGH SCRIPT
============================================
Per the Cardiotox-Fusion project brief: this script MUST be run and
verified BEFORE any full training pipeline is built.

What it does:
  1. Loads the FDA DICTrank cardiotoxicity labels from data/raw/dictrank_dataset_508.xlsx
  2. Samples 10 real DICTrank compounds (fixed seed=42, stratified by concern level)
  3. Resolves each compound to a SMILES string via the ChEMBL REST API
  4. Checks each compound for coverage in the LINCS L1000 GSE70138 metadata
  5. Logs all findings to data/processed/trace_through_log.txt

Purpose:
  Surface real failure modes early -- missing data, salt/counter-ion issues,
  LINCS coverage gaps -- rather than discovering them mid-training.

Known issues found and handled:
  - ChEMBL may return multiple candidates for one name (disambiguation needed)
  - ChEMBL pref_name can be None (not just absent) -- must guard with `or ""`
  - ~30% of compounds resolve to multi-component SMILES (salts) -- strip to parent
  - LINCS coverage is ~35% of the full 1,211-compound labeled set
  - Some compounds are genuinely absent from LINCS (not a matching bug)

Requirements:
  pip install rdkit torch torch_geometric pandas numpy requests openpyxl
  LINCS metadata files must be pre-downloaded to data/raw/lincs/
    - sig_info.txt.gz   (~2.1 MB)
    - pert_info.txt.gz  (~82 KB)

Outputs:
  - Appends a structured trace log to: data/processed/trace_through_log.txt
  - Prints per-compound results to stdout

Run with:
  python scripts/05_trace_ten_compounds.py
"""

import os
import time
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_ROOT = r"C:\Users\patel\Desktop\group_project\cardiotox-fusion"
DICTRANK_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "dictrank_dataset_508.xlsx")
LINCS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "lincs")
LOG_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "trace_through_log.txt")
RANDOM_SEED = 42
SALT_SUFFIXES = ["phosphate", "hydrochloride", "tromethamine", "sodium",
                  "sulfate", "maleate", "acetate", "citrate"]


# ---------------------------------------------------------------------------
# STEP 1: Load and clean DICTrank labels
# ---------------------------------------------------------------------------
def load_dictrank_labels():
    """
    Load and clean FDA DICTrank cardiotoxicity labels from the local Excel file.

    Cleaning steps applied:
      - Strip whitespace from all column names (known issue: 'Label Section ' has
        a trailing space in the raw file)
      - Rename 'DICT _ Concern' -> 'DICT_Concern' (inconsistent spacing in raw)
      - Normalize concern values: strip whitespace + lowercase (raw file has
        mixed cases: 'less', 'Less', 'less ' were three separate categories)

    Binarization rule (per project brief):
      - 'no'            -> label 0 (non-cardiotoxic)
      - 'less', 'most'  -> label 1 (cardiotoxic concern)
      - 'ambiguous'     -> dropped entirely

    Returns:
        pd.DataFrame: 1,211 rows with columns incl. 'DICT_Concern',
                      'cardiotox_label', 'Active Ingredient(s)',
                      'Generic/Proper Name(s)', 'Trade Name'
    """
    df = pd.read_excel(DICTRANK_PATH)  # requires openpyxl: pip install openpyxl
    df.columns = df.columns.str.strip()  # fix 'Label Section ' trailing space
    df = df.rename(columns={"DICT _ Concern": "DICT_Concern"})
    df["DICT_Concern"] = df["DICT_Concern"].str.strip().str.lower()  # fix casing

    df_labeled = df[df["DICT_Concern"] != "ambiguous"].copy()  # drop 107 ambiguous
    df_labeled["cardiotox_label"] = df_labeled["DICT_Concern"].map({
        "no": 0, "less": 1, "most": 1
    })
    return df_labeled


# ---------------------------------------------------------------------------
# STEP 2: ChEMBL SMILES resolution (exact-name-match rule, None-safe)
# ---------------------------------------------------------------------------
def resolve_smiles(drug_name):
    """
    Resolve a drug name to a canonical SMILES string via the ChEMBL REST API.

    Disambiguation rule (Design Decision #2 in PROJECT_DOCUMENTATION.md):
      ChEMBL sometimes returns multiple candidate molecules for one name.
      (e.g. 'OLAPARIB' returned the real drug + an unrelated analog AZD2461)
      We prefer an exact pref_name match (case-insensitive) over the highest
      relevance score. Only fall back to highest-score if no exact name match.

    None-guard on pref_name (Bug found during trace-through):
      ChEMBL's pref_name field can be present in JSON but have value None.
      A naive .get('pref_name', '').lower() would crash -- guard with `or ""`.

    Args:
        drug_name (str): Drug name to search (typically from DICTrank's
                         'Active Ingredient(s)' or 'Generic/Proper Name(s)')

    Returns:
        dict: {
            'query_name':   str   -- the input name
            'smiles':       str|None -- canonical SMILES (None on failure)
            'chembl_id':    str|None -- ChEMBL molecule ID
            'match_method': str|None -- 'exact_pref_name_match' or 'fallback_highest_score'
            'n_candidates': int  -- number of ChEMBL candidates returned
            'warning':      str|None -- any issue encountered (logged, non-fatal)
        }
    """
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule/search"
    params = {"q": drug_name, "format": "json"}
    result = {"query_name": drug_name, "smiles": None, "chembl_id": None,
              "match_method": None, "n_candidates": 0, "warning": None}
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
    except Exception as e:
        result["warning"] = f"Request failed: {e}"
        return result

    molecules = data.get("molecules", [])
    result["n_candidates"] = len(molecules)
    if not molecules:
        result["warning"] = "No ChEMBL matches found"
        return result

    # Prefer exact pref_name match; (m.get('pref_name') or '') guards against None
    exact_matches = [m for m in molecules
                      if (m.get("pref_name") or "").lower() == drug_name.lower()]
    if exact_matches:
        chosen = exact_matches[0]
        result["match_method"] = "exact_pref_name_match"
    else:
        # Fallback: highest relevance score (less reliable, log explicitly)
        chosen = max(molecules, key=lambda m: m.get("score") or 0)
        result["match_method"] = "fallback_highest_score"
        result["warning"] = (f"No exact name match; used highest score "
                              f"({chosen.get('score')}) among {len(molecules)} candidates")

    result["chembl_id"] = chosen.get("molecule_chembl_id")
    structures = chosen.get("molecule_structures")
    if structures:
        result["smiles"] = structures.get("canonical_smiles")
    else:
        result["warning"] = (result["warning"] or "") + " | No structure data"
    return result


def get_parent_fragment(smiles):
    """
    Strip salt counter-ions from a multi-component SMILES string.

    Background (Design Decision #3 in PROJECT_DOCUMENTATION.md):
      ~30% of DICTrank compounds resolve to multi-component SMILES (drug +
      counter-ion or buffer molecule, joined by '.' in SMILES notation).
      Examples found in trace-through:
        - SONIDEGIB PHOSPHATE  -> drug + 2 phosphate ions
        - FLAVOXATE HCl        -> drug + chloride
        - CARBOPROST TROMETHAMINE -> drug + tromethamine buffer

    Rule: The parent molecule = largest fragment by heavy-atom count.
    Every stripped fragment is logged explicitly (never silently discarded).
    Verified correct on all 3 real cases by manual inspection.

    Args:
        smiles (str): Canonical SMILES string (may contain '.' separators)

    Returns:
        dict: {
            'parent_smiles': str|None -- SMILES of the largest fragment
            'n_fragments':   int      -- total fragments found (1 = no salt)
            'stripped':      list[str] -- SMILES of removed fragments (for logging)
            'error':         str|None -- error message if parsing failed
        }
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"parent_smiles": None, "n_fragments": 0, "stripped": None,
                "error": "MolFromSmiles failed"}
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) == 1:
        # Single-component SMILES -- no stripping needed
        return {"parent_smiles": Chem.MolToSmiles(frags[0]), "n_fragments": 1,
                "stripped": [], "error": None}
    # Sort by heavy-atom count descending; largest = parent drug molecule
    frags_sorted = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
    parent = frags_sorted[0]
    stripped = [Chem.MolToSmiles(f) for f in frags_sorted[1:]]  # log these, don't discard
    return {"parent_smiles": Chem.MolToSmiles(parent), "n_fragments": len(frags),
            "stripped": stripped, "error": None}


def strip_salt_suffix(name):
    """
    Remove common salt/ester suffixes from a drug name for LINCS name-matching.

    Example: 'FLAVOXATE HYDROCHLORIDE' -> 'flavoxate'
    This recovered the LINCS match for flavoxate in the trace-through.
    NOTE: Not sufficient on its own -- sonidegib (phosphate stripped) and
    carboprost (tromethamine stripped) were still absent from LINCS.
    Use InChIKey matching as the more robust fallback.

    Args:
        name (str): Drug name, possibly with a salt suffix

    Returns:
        str: Lowercased, stripped drug name (without salt suffix if matched)
    """
    name_lower = name.lower().strip()
    for suffix in SALT_SUFFIXES:
        if name_lower.endswith(" " + suffix):
            return name_lower[: -(len(suffix) + 1)].strip()
    return name_lower


def smiles_to_inchikey(smiles):
    """
    Convert a SMILES string to an InChIKey for structure-based LINCS matching.

    InChIKey matching is immune to naming differences entirely -- it compares
    the actual molecular structure against LINCS's own inchi_key column.
    Used to confirm genuine absence (not a matching bug) for sonidegib and
    carboprost in the trace-through.

    Args:
        smiles (str): Canonical SMILES string

    Returns:
        str|None: 27-character InChIKey string, or None if parsing failed
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return inchi.MolToInchiKey(mol)


# ---------------------------------------------------------------------------
# STEP 3: LINCS matching
# ---------------------------------------------------------------------------
def load_lincs_metadata():
    """
    Load LINCS L1000 GSE70138 signature and perturbation metadata files.

    Files loaded (must be pre-downloaded to data/raw/lincs/):
      - sig_info.txt.gz  : 118,050 signatures; columns incl. sig_id, pert_id,
                           cell_id, pert_iname, pert_dose, pert_time
      - pert_info.txt.gz : 2,170 unique compounds; columns incl. pert_id,
                           pert_iname, inchi_key

    NOTE: The full Level 5 expression matrix (~5 GB GCTX file) is NOT loaded
    here. Only metadata is loaded at this stage to confirm match rates before
    committing to the large download.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (sig_info, pert_info)
            pert_info has an added 'pert_iname_lower' column for matching.
    """
    sig_info = pd.read_csv(os.path.join(LINCS_DIR, "sig_info.txt.gz"),
                            sep="\t", compression="gzip")
    pert_info = pd.read_csv(os.path.join(LINCS_DIR, "pert_info.txt.gz"),
                             sep="\t", compression="gzip")
    pert_info["pert_iname_lower"] = pert_info["pert_iname"].str.lower().str.strip()
    return sig_info, pert_info


def check_lincs_match(name, pert_info):
    """
    Check if a drug name (or its salt-stripped version) appears in LINCS metadata.

    Two matching strategies are tried in order:
      1. Direct name match (lowercased + stripped)
      2. Salt-suffix-stripped name match (e.g. 'flavoxate' from 'flavoxate hydrochloride')

    NOTE: This function does NOT perform InChIKey (structure-based) matching.
    InChIKey matching is more robust but requires a SMILES -> InChIKey conversion
    per compound. It should be used as the primary strategy in the full pipeline
    script (scripts/03_match_lincs.py) rather than just name-matching.

    Trace-through finding:
      - 6/10 sample compounds matched (4 direct, 2 via salt-strip)
      - 4/10 were absent -- confirmed genuine absence (not a bug) via InChIKey
      - Full dataset: 423/1,211 matched (34.9%) by name only

    Args:
        name (str): Drug name to check
        pert_info (pd.DataFrame): LINCS pert_info with 'pert_iname_lower' column

    Returns:
        tuple[bool, bool, bool]: (has_any_match, direct_match, via_salt_strip_match)
    """
    name_lower = name.lower().strip()
    stripped = strip_salt_suffix(name)
    direct = pert_info["pert_iname_lower"].eq(name_lower).any()
    via_strip = pert_info["pert_iname_lower"].eq(stripped).any()
    return direct or via_strip, direct, via_strip


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    log_lines = []
    log_lines.append("TRACE-THROUGH SCRIPT RUN")
    log_lines.append("=" * 50)

    print("Loading DICTrank labels...")
    df_labeled = load_dictrank_labels()
    print(f"  {len(df_labeled)} labeled compounds after dropping ambiguous")

    print("\nSampling 10 compounds (seed=42)...")
    no_c = df_labeled[df_labeled["DICT_Concern"] == "no"].sample(3, random_state=RANDOM_SEED)
    less_c = df_labeled[df_labeled["DICT_Concern"] == "less"].sample(4, random_state=RANDOM_SEED)
    most_c = df_labeled[df_labeled["DICT_Concern"] == "most"].sample(3, random_state=RANDOM_SEED)
    sample_10 = pd.concat([no_c, less_c, most_c])

    print("Loading LINCS metadata (sig_info, pert_info)...")
    sig_info, pert_info = load_lincs_metadata()

    print("\nResolving SMILES + checking LINCS matches for each compound...\n")
    for idx, row in sample_10.iterrows():
        name = row["Active Ingredient(s)"]
        if pd.isna(name):
            name = row["Generic/Proper Name(s)"]

        smiles_res = resolve_smiles(name)
        line = f"{row['Trade Name']} / {name} (label={row['cardiotox_label']})"
        print(line)
        log_lines.append(line)

        if smiles_res["smiles"] is None:
            msg = f"  SMILES: FAILED ({smiles_res['warning']})"
            print(msg); log_lines.append(msg)
        else:
            parent_info = get_parent_fragment(smiles_res["smiles"])
            n_frags = parent_info["n_fragments"]
            msg = f"  SMILES resolved ({smiles_res['match_method']}), {n_frags} fragment(s)"
            print(msg); log_lines.append(msg)
            if n_frags > 1:
                msg2 = f"    Stripped salt fragments: {parent_info['stripped']}"
                print(msg2); log_lines.append(msg2)

        has_match, direct, via_strip = check_lincs_match(name, pert_info)
        msg = f"  LINCS match: {has_match} (direct={direct}, via_salt_strip={via_strip})"
        print(msg); log_lines.append(msg)
        print()
        log_lines.append("")

        time.sleep(0.5)  # be polite to ChEMBL API

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Log appended to: {LOG_PATH}")


if __name__ == "__main__":
    main()
