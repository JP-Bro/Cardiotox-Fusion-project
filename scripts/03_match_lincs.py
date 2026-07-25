"""
03_match_lincs.py -- Step 3: Match resolved compounds against LINCS L1000 signatures.

What this script does:
  1. Loads compounds_with_smiles.csv (output of 02_resolve_smiles.py)
  2. Loads LINCS metadata: sig_info.txt.gz and pert_info.txt.gz
  3. Matches compounds using two strategies:
       a. Name-based: direct name match + salt-suffix-stripped name match
       b. InChIKey-based: structure match against LINCS inchi_key column (more robust)
  4. For matched compounds: selects the canonical signature
       cell_id=HA1E, highest available dose, 24h timepoint (Design Decision #5)
  5. Reports full coverage statistics
  6. Optionally downloads the 5 GB Level 5 GCTX expression matrix (--download flag)
  7. If GCTX available: extracts the 978-gene expression vectors for matched compounds
  8. Saves: lincs_matched_compounds.csv and (optionally) expression_matrix.csv

Key findings from trace-through encoded here:
  - Name matching alone finds ~35% of compounds
  - InChIKey matching adds a few more percentage points (recovered flavoxate case)
  - Some compounds (sonidegib, carboprost) are genuinely absent -- not a matching bug
  - Coverage is uneven: 28% negative class, 38% positive class -- MUST be reported
  - Selected condition: HA1E cell line, 10 µM, 24h -- present for all matched compounds

Run with:
  python scripts/03_match_lincs.py                  # match only, no download
  python scripts/03_match_lincs.py --download       # match + download 5GB GCTX
  python scripts/03_match_lincs.py --extract        # extract expression matrix (GCTX must exist)
"""

import os
import sys
import argparse
import gzip
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from config import CFG
from scripts.utils import get_logger, ensure_dirs, strip_salt_suffix

LOG_FILE = os.path.join(CFG.RESULTS_DIR, "logs", "03_match_lincs.log")
logger = get_logger("03_match_lincs", LOG_FILE)


# ---------------------------------------------------------------------------
# LINCS Metadata Loading
# ---------------------------------------------------------------------------

def load_lincs_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load LINCS L1000 GSE70138 signature and perturbation metadata.

    Files (must be pre-downloaded to data/raw/lincs/):
      sig_info.txt.gz  : 118,050 rows -- one per signature
                         Columns: sig_id, pert_id, pert_iname, cell_id,
                                  pert_dose, pert_dose_unit, pert_time, pert_time_unit
      pert_info.txt.gz : 2,170 rows -- one per unique compound
                         Columns: pert_id, pert_iname, inchi_key, ...

    Returns:
        (sig_info DataFrame, pert_info DataFrame)
        pert_info has 'pert_iname_lower' column added for case-insensitive matching.
    """
    for f, path in [("sig_info", CFG.LINCS_SIG_INFO), ("pert_info", CFG.LINCS_PERT_INFO)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"LINCS {f} not found at {path}. "
                "Download from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138"
            )

    sig_info = pd.read_csv(CFG.LINCS_SIG_INFO, sep="\t", compression="gzip")
    pert_info = pd.read_csv(CFG.LINCS_PERT_INFO, sep="\t", compression="gzip")

    logger.info(f"LINCS sig_info: {sig_info.shape} ({sig_info['sig_id'].nunique()} unique sigs)")
    logger.info(f"LINCS pert_info: {pert_info.shape} ({len(pert_info)} unique compounds)")

    pert_info["pert_iname_lower"] = pert_info["pert_iname"].str.lower().str.strip()
    return sig_info, pert_info


# ---------------------------------------------------------------------------
# Matching Strategies
# ---------------------------------------------------------------------------

def match_by_name(drug_name: str, pert_info: pd.DataFrame) -> tuple[bool, bool, bool]:
    """
    Check LINCS match by drug name (direct + salt-suffix-stripped).

    Returns:
        (has_match, direct_match, via_salt_strip_match) -- all bool
    """
    name_lower = drug_name.lower().strip()
    stripped = strip_salt_suffix(drug_name)
    direct = pert_info["pert_iname_lower"].eq(name_lower).any()
    via_strip = pert_info["pert_iname_lower"].eq(stripped).any() if stripped != name_lower else False
    return (direct or via_strip), direct, via_strip


def match_by_inchikey(inchi_key: str, pert_info: pd.DataFrame) -> bool:
    """
    Check LINCS match by InChIKey (structure-based, immune to naming differences).

    InChIKey matching confirmed genuine absence of sonidegib and carboprost
    during trace-through -- these are not a matching bug.

    Note: LINCS pert_info's inchi_key column may have partial InChIKeys
    (first 14 chars = connectivity layer only). We match on first 14 chars
    to handle both full and partial InChIKeys.

    Args:
        inchi_key: Full 27-char InChIKey computed from resolved SMILES
        pert_info: LINCS pert_info DataFrame

    Returns:
        True if any LINCS compound matches by structure
    """
    if not isinstance(inchi_key, str) or len(inchi_key) < 14:
        return False

    connectivity_key = inchi_key[:14]  # first block = connectivity layer

    if "inchi_key" not in pert_info.columns:
        logger.warning("No 'inchi_key' column in pert_info -- skipping InChIKey matching")
        return False

    lincs_keys = pert_info["inchi_key"].dropna().str[:14]
    return lincs_keys.eq(connectivity_key).any()


def get_pert_id_for_name(drug_name: str, pert_info: pd.DataFrame) -> str:
    """
    Get LINCS pert_id for a matched compound name.

    Tries direct name match first, then salt-stripped name.
    Returns empty string if not found.
    """
    name_lower = drug_name.lower().strip()
    match = pert_info[pert_info["pert_iname_lower"] == name_lower]
    if match.empty:
        stripped = strip_salt_suffix(drug_name)
        match = pert_info[pert_info["pert_iname_lower"] == stripped]
    if match.empty:
        return ""
    return str(match.iloc[0]["pert_id"])


def get_pert_id_for_inchikey(inchi_key: str, pert_info: pd.DataFrame) -> str:
    """Get LINCS pert_id for a matched compound by InChIKey."""
    if not isinstance(inchi_key, str) or "inchi_key" not in pert_info.columns:
        return ""
    connectivity_key = inchi_key[:14]
    lincs_keys = pert_info["inchi_key"].dropna().str[:14]
    match_idx = lincs_keys[lincs_keys == connectivity_key].index
    if len(match_idx) == 0:
        return ""
    return str(pert_info.loc[match_idx[0], "pert_id"])


# ---------------------------------------------------------------------------
# Canonical Signature Selection
# ---------------------------------------------------------------------------

def select_canonical_signature(pert_id: str, sig_info: pd.DataFrame) -> pd.Series:
    """
    Select the canonical LINCS signature for a compound.

    Design Decision #5 (from trace-through):
      Use cell_id=HA1E, highest available dose, 24h timepoint.
      Rationale: HA1E is present for all matched compounds and is a relatively
      non-transformed reference line (unlike the 5 cancer lines in the panel).
      Explicitly logged with rationale -- not a silent .iloc[0].

    Args:
        pert_id:  LINCS compound identifier
        sig_info: LINCS signatures metadata DataFrame

    Returns:
        pd.Series: The selected signature row, or empty Series if no match.
    """
    sigs = sig_info[sig_info["pert_id"] == pert_id]
    if sigs.empty:
        return pd.Series(dtype=object)

    # Filter to canonical condition
    cell_sigs = sigs[sigs["cell_id"] == CFG.LINCS_CELL_LINE]
    if cell_sigs.empty:
        # Fallback: try other cell lines
        logger.warning(f"  {pert_id}: no HA1E signatures found, trying any cell line")
        cell_sigs = sigs

    # Select highest dose
    if "pert_dose" in cell_sigs.columns:
        try:
            cell_sigs = cell_sigs.copy()
            cell_sigs["pert_dose_float"] = pd.to_numeric(cell_sigs["pert_dose"], errors="coerce")
            max_dose = cell_sigs["pert_dose_float"].max()
            cell_sigs = cell_sigs[cell_sigs["pert_dose_float"] == max_dose]
        except Exception:
            pass

    # Select 24h timepoint
    if "pert_time" in cell_sigs.columns:
        time_filter = cell_sigs["pert_time"].astype(str).str.contains("24", na=False)
        if time_filter.any():
            cell_sigs = cell_sigs[time_filter]

    return cell_sigs.iloc[0] if not cell_sigs.empty else pd.Series(dtype=object)


# ---------------------------------------------------------------------------
# GCTX Expression Matrix Extraction
# ---------------------------------------------------------------------------

def download_gctx(url: str, dest_path: str):
    """
    Download the LINCS Level 5 GCTX expression matrix (~5 GB).

    Progress is printed every 100 MB. This is a one-time download -- the file
    is too large to include in the repository.

    Args:
        url:       Direct download URL
        dest_path: Local destination file path (uncompressed .gctx)
    """
    logger.info(f"Downloading LINCS Level 5 GCTX from: {url}")
    logger.info(f"Destination: {dest_path}")
    logger.info("WARNING: This file is ~5 GB and may take 10-30 minutes on a typical connection.")

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    gz_path = dest_path + ".gz"

    try:
        def progress_hook(block, block_size, total):
            downloaded = block * block_size
            if downloaded % (100 * 1024 * 1024) < block_size:  # every 100MB
                pct = downloaded / total * 100 if total > 0 else 0
                logger.info(f"  Downloaded: {downloaded / 1024**3:.2f} GB / {total / 1024**3:.2f} GB ({pct:.1f}%)")

        urllib.request.urlretrieve(url, gz_path, reporthook=progress_hook)

        logger.info("Decompressing .gz file...")
        with gzip.open(gz_path, "rb") as f_in, open(dest_path, "wb") as f_out:
            while True:
                chunk = f_in.read(8 * 1024 * 1024)  # 8MB chunks
                if not chunk:
                    break
                f_out.write(chunk)

        os.remove(gz_path)
        logger.info(f"GCTX ready at: {dest_path}")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def extract_expression_matrix(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 978-gene expression vectors for all matched compounds from GCTX.
    Uses direct h5py parsing to avoid index ordering and duplicate mapping bugs in cmapPy.

    Args:
        matched_df: DataFrame with 'sig_id' column for matched compounds

    Returns:
        pd.DataFrame: shape (n_matched, 978) -- rows=sig_id, cols=gene landmarks
    """
    import h5py

    if not os.path.isfile(CFG.LINCS_GCTX):
        raise FileNotFoundError(
            f"GCTX not found at {CFG.LINCS_GCTX}. Run with --download flag first."
        )

    sig_ids = matched_df["sig_id"].dropna().tolist()
    logger.info(f"Extracting expression for {len(sig_ids)} signatures from GCTX...")

    with h5py.File(CFG.LINCS_GCTX, 'r') as f:
        # 1. Load column metadata (signatures)
        col_ids_bytes = f['0/META/COL/id'][:]
        col_ids = [s.decode('utf-8') for s in col_ids_bytes]
        
        # 2. Map target sig_ids to their index position in the file
        sig_to_idx = {sig: idx for idx, sig in enumerate(col_ids)}
        mapped_indices = []
        valid_sig_ids = []
        
        for sig in sig_ids:
            if sig in sig_to_idx:
                mapped_indices.append(sig_to_idx[sig])
                valid_sig_ids.append(sig)
                
        if not mapped_indices:
            raise ValueError("None of the matched sig_ids were found in GCTX column metadata")
            
        logger.info(f"  Mapped {len(mapped_indices)} signatures out of {len(sig_ids)} requested")
        
        # 3. Find UNIQUE sorted indices for h5py (strictly increasing, no duplicates)
        unique_indices = sorted(list(set(mapped_indices)))
        logger.info(f"  Fetching {len(unique_indices)} unique signature rows from data matrix...")
        
        # 4. Slice the matrix (first 978 columns are landmark genes)
        matrix_dataset = f['0/DATA/0/matrix']
        expr_data = matrix_dataset[unique_indices, :CFG.NUM_GENES]
        
        # 5. Reconstruct final matrix in the exact order of the requested valid_sig_ids
        idx_to_row = {h5_idx: row_idx for row_idx, h5_idx in enumerate(unique_indices)}
        final_rows = []
        for sig in valid_sig_ids:
            h5_idx = sig_to_idx[sig]
            row_idx = idx_to_row[h5_idx]
            final_rows.append(expr_data[row_idx])
            
        final_expr_data = np.array(final_rows, dtype=np.float32)
        
        # 6. Load row metadata (gene IDs) for columns
        row_ids_bytes = f['0/META/ROW/id'][:CFG.NUM_GENES]
        gene_ids = [g.decode('utf-8') for g in row_ids_bytes]
        
        # 7. Convert to DataFrame
        expr_df = pd.DataFrame(final_expr_data, index=valid_sig_ids, columns=gene_ids)
        logger.info(f"  Expression matrix extraction complete. Shape: {expr_df.shape}")
        
    return expr_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Match compounds against LINCS L1000")
    parser.add_argument("--download", action="store_true",
                        help="Download the Level 5 GCTX expression matrix (~5 GB)")
    parser.add_argument("--extract", action="store_true",
                        help="Extract expression matrix from GCTX (requires GCTX to be present)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SCRIPT: 03_match_lincs.py")
    logger.info("PURPOSE: Match compounds against LINCS L1000 GSE70138")
    logger.info("=" * 60)

    ensure_dirs(CFG.PROCESSED_DIR, os.path.join(CFG.RESULTS_DIR, "logs"))

    # Load compounds -- prefer SMILES CSV (has InChIKey), fall back to labels CSV for name-only matching
    if os.path.isfile(CFG.SMILES_CSV):
        df = pd.read_csv(CFG.SMILES_CSV)
        df_success = df[df["status"] == "success"].copy()
        logger.info(f"Loaded {len(df)} compounds from SMILES CSV; {len(df_success)} with valid SMILES")
        logger.info("InChIKey-based matching ENABLED (SMILES CSV present)")
    elif os.path.isfile(CFG.LABELS_CSV):
        logger.warning("SMILES CSV not found -- using labeled_compounds.csv for name-only matching.")
        logger.warning("Run 02_resolve_smiles.py later to enable InChIKey-based matching.")
        df_labels = pd.read_csv(CFG.LABELS_CSV)
        # Build a minimal dataframe compatible with the matching code
        df_success = pd.DataFrame({
            "query_name": df_labels["resolved_name"],
            "cardiotox_label": df_labels["cardiotox_label"],
            "parent_smiles": None,
            "inchi_key": None,
            "chembl_id": None,
        })
        logger.info(f"Loaded {len(df_success)} compounds from labels CSV (name-only matching)")
    else:
        raise FileNotFoundError(
            f"Neither SMILES CSV ({CFG.SMILES_CSV}) nor labels CSV ({CFG.LABELS_CSV}) found.\n"
            "Run at minimum: python scripts/01_fetch_labels.py"
        )

    # Load LINCS metadata
    sig_info, pert_info = load_lincs_metadata()

    # Match all compounds
    logger.info("Matching compounds against LINCS L1000...")
    match_results = []

    for i, row in df_success.iterrows():
        name = str(row["query_name"])
        inchi_key = str(row["inchi_key"]) if pd.notna(row.get("inchi_key")) else ""

        name_match, direct, via_strip = match_by_name(name, pert_info)
        ik_match = match_by_inchikey(inchi_key, pert_info) if not name_match else False
        has_match = name_match or ik_match

        pert_id = ""
        sig_id = ""
        match_method = "none"

        if name_match:
            pert_id = get_pert_id_for_name(name, pert_info)
            match_method = "direct_name" if direct else "salt_strip_name"
        elif ik_match:
            pert_id = get_pert_id_for_inchikey(inchi_key, pert_info)
            match_method = "inchikey"

        # Select canonical signature
        if pert_id:
            canon_sig = select_canonical_signature(pert_id, sig_info)
            if not canon_sig.empty:
                sig_id = canon_sig.get("sig_id", "")

        match_results.append({
            "query_name": name,
            "cardiotox_label": int(row["cardiotox_label"]),
            "parent_smiles": row["parent_smiles"],
            "inchi_key": inchi_key,
            "chembl_id": row.get("chembl_id", ""),
            "lincs_match": has_match,
            "match_method": match_method,
            "pert_id": pert_id,
            "sig_id": sig_id,
            "direct_name_match": direct,
            "salt_strip_match": via_strip,
            "inchikey_match": ik_match,
        })

    df_matches = pd.DataFrame(match_results)

    # --- Coverage Report ---
    n_total = len(df_success)
    n_matched = df_matches["lincs_match"].sum()
    n_label0 = df_matches[df_matches["cardiotox_label"] == 0]["lincs_match"].sum()
    n_label1 = df_matches[df_matches["cardiotox_label"] == 1]["lincs_match"].sum()
    total_label0 = (df_matches["cardiotox_label"] == 0).sum()
    total_label1 = (df_matches["cardiotox_label"] == 1).sum()

    logger.info("=" * 60)
    logger.info("LINCS COVERAGE REPORT")
    logger.info(f"  Total compounds (valid SMILES) : {n_total}")
    logger.info(f"  Matched to LINCS              : {n_matched} ({n_matched/n_total:.1%})")
    logger.info(f"  - By direct name              : {(df_matches['direct_name_match']).sum()}")
    logger.info(f"  - By salt-stripped name       : {(df_matches['salt_strip_match'] & ~df_matches['direct_name_match']).sum()}")
    logger.info(f"  - By InChIKey only            : {(df_matches['inchikey_match'] & ~df_matches['lincs_match'].shift(fill_value=False)).sum()}")
    logger.info(f"  Coverage - label 0 (no concern): {n_label0}/{total_label0} ({n_label0/max(total_label0,1):.1%})")
    logger.info(f"  Coverage - label 1 (concern)  : {n_label1}/{total_label1} ({n_label1/max(total_label1,1):.1%})")
    logger.info(f"  Coverage gap (pos-neg)     : {n_label1/max(total_label1,1) - n_label0/max(total_label0,1):.1%} more coverage for toxic class")
    logger.info(f"  => Stratified split REQUIRED to handle imbalance")
    logger.info("=" * 60)

    # Save match results
    df_matches.to_csv(CFG.LINCS_MATCHED_CSV, index=False)
    logger.info(f"Match results saved to: {CFG.LINCS_MATCHED_CSV}")

    # Optional: download GCTX
    if args.download:
        download_gctx(CFG.LINCS_GCTX_URL, CFG.LINCS_GCTX)

    # Optional: extract expression matrix
    if args.extract:
        df_matched_only = df_matches[df_matches["lincs_match"] & df_matches["sig_id"].ne("")]
        logger.info(f"Extracting expression for {len(df_matched_only)} matched compounds...")
        expr_df = extract_expression_matrix(df_matched_only)
        expr_df.to_csv(CFG.EXPRESSION_CSV)
        logger.info(f"Expression matrix saved to: {CFG.EXPRESSION_CSV}")

    logger.info("03_match_lincs.py COMPLETE")


if __name__ == "__main__":
    main()
