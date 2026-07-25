"""
02_resolve_smiles.py -- Step 2: Resolve all labeled compounds to SMILES via ChEMBL.

What this script does:
  1. Loads data/processed/labeled_compounds.csv (output of 01_fetch_labels.py)
  2. For each compound: queries ChEMBL REST API by name
  3. Applies disambiguation rule: exact pref_name match > highest score fallback
  4. Applies salt stripping: keeps largest fragment by heavy-atom count
  5. Computes InChIKey for each parent SMILES (for later LINCS structure-matching)
  6. Saves results to data/processed/compounds_with_smiles.csv
  7. Logs every decision: match method, stripped fragments, warnings

Design decisions encoded here (from trace-through):
  - Prefer exact pref_name match over ChEMBL score (OLAPARIB: 2 candidates, took correct one)
  - Guard pref_name against None (not just absent) with `or ""`
  - Parent = largest heavy-atom fragment (verified on 3 real salt cases)
  - Fallback to Generic/Proper Name(s) for 27 rows missing Active Ingredient(s)
  - 0.4s sleep between API calls (rate-limit courtesy)
  - Exponential backoff on transient errors

Expected outcome:
  - ~1,200+ of 1,211 compounds resolve successfully (10/10 in sample)
  - ~30% will require salt stripping
  - Full run takes ~10-15 minutes (API rate limited)

Run with:
  python scripts/02_resolve_smiles.py
  
  # Resume from checkpoint (skips already-resolved rows):
  python scripts/02_resolve_smiles.py --resume
"""

import os
import sys
import argparse
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from config import CFG
from scripts.utils import (
    get_logger, ensure_dirs, robust_get, get_parent_smiles,
    smiles_to_inchikey, validate_smiles
)

LOG_FILE = os.path.join(CFG.RESULTS_DIR, "logs", "02_resolve_smiles.log")
CHECKPOINT_FILE = os.path.join(CFG.PROCESSED_DIR, "smiles_checkpoint.json")

logger = get_logger("02_resolve_smiles", LOG_FILE)


def resolve_smiles_chembl(drug_name: str) -> dict:
    """
    Resolve a drug name to canonical SMILES via the ChEMBL REST API.

    Strategy (Design Decision #2):
      1. Query ChEMBL search endpoint with drug name
      2. From all candidates, prefer one with exact pref_name match (case-insensitive)
      3. If no exact match exists, fall back to candidate with highest relevance score
      4. Log whenever fallback path is used

    ChEMBL pref_name None-guard:
      pref_name can be present in JSON with value None -- guard with `(... or "")`.
      A naive .get("pref_name", "").lower() would crash on this case.

    Args:
        drug_name: Drug name string (from resolved_name column)

    Returns:
        dict with:
            query_name    : input name
            raw_smiles    : raw SMILES from ChEMBL (may be multi-component)
            parent_smiles : largest fragment SMILES (salt-stripped)
            inchi_key     : InChIKey of parent_smiles
            chembl_id     : ChEMBL molecule identifier
            match_method  : 'exact_pref_name_match' | 'fallback_highest_score' | None
            n_candidates  : number of ChEMBL results returned
            n_fragments   : number of SMILES components in raw_smiles
            stripped_frags: list of removed counter-ion SMILES
            status        : 'success' | 'no_match' | 'no_structure' | 'invalid_smiles' | 'api_error'
            warning       : any non-fatal issue (logged)
    """
    result = {
        "query_name": drug_name,
        "raw_smiles": None,
        "parent_smiles": None,
        "inchi_key": None,
        "chembl_id": None,
        "match_method": None,
        "n_candidates": 0,
        "n_fragments": 0,
        "stripped_frags": [],
        "status": "api_error",
        "warning": None,
    }

    url = f"{CFG.CHEMBL_API_BASE}/molecule/search"
    params = {"q": drug_name, "format": "json"}
    data = robust_get(url, params=params, logger=logger)

    if data is None:
        result["warning"] = "API request failed after all retries"
        return result

    molecules = data.get("molecules", [])
    result["n_candidates"] = len(molecules)

    if not molecules:
        result["status"] = "no_match"
        result["warning"] = "No ChEMBL candidates returned"
        return result

    # --- Disambiguation ---
    # Guard: pref_name can be None (not just absent in JSON)
    exact = [m for m in molecules
             if (m.get("pref_name") or "").lower() == drug_name.lower()]

    if exact:
        chosen = exact[0]
        result["match_method"] = "exact_pref_name_match"
    else:
        chosen = max(molecules, key=lambda m: m.get("score") or 0)
        result["match_method"] = "fallback_highest_score"
        result["warning"] = (
            f"No exact pref_name match; chose highest score "
            f"({chosen.get('score')}) among {len(molecules)} candidates"
        )
        logger.warning(f"  {drug_name}: fallback to score-based selection -- verify manually if critical")

    result["chembl_id"] = chosen.get("molecule_chembl_id")
    structures = chosen.get("molecule_structures") or {}
    raw_smiles = structures.get("canonical_smiles")

    if not raw_smiles:
        result["status"] = "no_structure"
        result["warning"] = (result["warning"] or "") + " | No canonical_smiles in ChEMBL response"
        return result

    result["raw_smiles"] = raw_smiles

    # --- Salt stripping ---
    parent_info = get_parent_smiles(raw_smiles)

    if parent_info["error"]:
        result["status"] = "invalid_smiles"
        result["warning"] = (result["warning"] or "") + f" | RDKit error: {parent_info['error']}"
        return result

    result["parent_smiles"] = parent_info["parent_smiles"]
    result["n_fragments"] = parent_info["n_fragments"]
    result["stripped_frags"] = parent_info["stripped"]

    if parent_info["n_fragments"] > 1:
        logger.info(f"  {drug_name}: salt stripped ({parent_info['n_fragments']} frags), "
                    f"removed: {parent_info['stripped']}")

    # --- InChIKey ---
    result["inchi_key"] = smiles_to_inchikey(result["parent_smiles"])
    result["status"] = "success"
    return result


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load previously resolved results from checkpoint file (for --resume mode)."""
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded checkpoint: {len(data)} previously resolved compounds")
        return data
    return {}


def save_checkpoint(checkpoint_path: str, resolved: dict):
    """Save current resolved results to checkpoint file."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Resolve drug names to SMILES via ChEMBL")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint -- skip already-resolved compounds")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N compounds (for testing)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SCRIPT: 02_resolve_smiles.py")
    logger.info("PURPOSE: ChEMBL SMILES resolution for all labeled compounds")
    logger.info("=" * 60)

    ensure_dirs(CFG.PROCESSED_DIR, os.path.join(CFG.RESULTS_DIR, "logs"))

    # Load labels
    if not os.path.isfile(CFG.LABELS_CSV):
        raise FileNotFoundError(
            f"Labels CSV not found at {CFG.LABELS_CSV}. "
            "Run scripts/01_fetch_labels.py first."
        )
    df = pd.read_csv(CFG.LABELS_CSV)
    logger.info(f"Loaded {len(df)} labeled compounds from {CFG.LABELS_CSV}")

    if args.limit:
        df = df.head(args.limit)
        logger.info(f"  [--limit] Processing only first {args.limit} compounds")

    # Load checkpoint if resuming
    resolved_cache = load_checkpoint(CHECKPOINT_FILE) if args.resume else {}

    results = []
    n_success = 0
    n_fail = 0
    n_skipped = 0

    for i, row in df.iterrows():
        name = row["resolved_name"]

        # Skip if already in checkpoint
        if name in resolved_cache:
            results.append(resolved_cache[name])
            n_skipped += 1
            continue

        logger.info(f"[{i+1}/{len(df)}] Resolving: {name}")
        result = resolve_smiles_chembl(name)
        result["cardiotox_label"] = int(row["cardiotox_label"])
        result["DICT_Concern"] = row["DICT_Concern"]

        resolved_cache[name] = result
        results.append(result)

        if result["status"] == "success":
            n_success += 1
            logger.info(f"  OK {name} -> {result['parent_smiles'][:40]}... "
                        f"(method={result['match_method']}, frags={result['n_fragments']})")
        else:
            n_fail += 1
            logger.warning(f"  X {name}: {result['status']} -- {result['warning']}")

        # Checkpoint every 50 compounds
        if (i + 1) % 50 == 0:
            save_checkpoint(CHECKPOINT_FILE, resolved_cache)
            logger.info(f"  [Checkpoint saved at {i+1} compounds]")

    # Final save
    save_checkpoint(CHECKPOINT_FILE, resolved_cache)

    # Convert to DataFrame and save
    df_results = pd.DataFrame(results)
    df_results.to_csv(CFG.SMILES_CSV, index=False)

    # Summary
    logger.info("=" * 60)
    logger.info("RESOLUTION SUMMARY")
    logger.info(f"  Total compounds         : {len(df)}")
    logger.info(f"  Successfully resolved   : {n_success + n_skipped} ({(n_success + n_skipped)/len(df):.1%})")
    logger.info(f"  - Newly resolved        : {n_success}")
    logger.info(f"  - From checkpoint       : {n_skipped}")
    logger.info(f"  Failed                  : {n_fail}")
    if len(df_results) > 0:
        n_salts = (df_results["n_fragments"] > 1).sum()
        n_fallback = (df_results["match_method"] == "fallback_highest_score").sum()
        logger.info(f"  Salt-stripped compounds : {n_salts} ({n_salts/len(df_results):.1%})")
        logger.info(f"  Fallback-to-score used  : {n_fallback}")
    logger.info("=" * 60)
    logger.info(f"Output: {CFG.SMILES_CSV}")
    logger.info("02_resolve_smiles.py COMPLETE")


if __name__ == "__main__":
    main()
