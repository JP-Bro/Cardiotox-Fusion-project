"""
utils.py -- Shared utility functions for the Cardiotox-Fusion pipeline.

Imported by all pipeline scripts. Contains:
  - Logging setup
  - Salt stripping helpers
  - RDKit molecule/graph helpers
  - Reproducibility helpers (seed setting)
  - Progress bar wrapper
"""

import os
import sys
import time
import random
import logging
import hashlib
import functools
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import inchi as rdkit_inchi, Descriptors, rdMolDescriptors

# Add project root to sys.path so `from config import CFG` works from any script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import CFG


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    Create a logger that writes to stdout AND optionally to a file.

    Args:
        name:     Logger name (use __name__ in calling modules)
        log_file: Optional path to a .log file. Directory created if needed.
        level:    Logging level (default INFO)

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = CFG.RANDOM_SEED):
    """Set random seeds for Python, NumPy, and PyTorch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not required for data prep scripts


# ---------------------------------------------------------------------------
# Salt / Name Helpers
# ---------------------------------------------------------------------------

def strip_salt_suffix(name: str, salt_suffixes: Optional[List[str]] = None) -> str:
    """
    Strip common pharmaceutical salt/counter-ion suffixes from a drug name.

    Design Decision #3 (from trace-through): required before LINCS name-matching.
    Example: 'FLAVOXATE HYDROCHLORIDE' -> 'flavoxate'

    Note: Stripping alone is not sufficient -- some compounds (sonidegib, carboprost)
    are genuinely absent from LINCS even after stripping. Use InChIKey matching
    as the more robust fallback.

    Args:
        name:          Drug name string (any case)
        salt_suffixes: List of suffixes to try. Defaults to CFG.SALT_SUFFIXES.

    Returns:
        Lowercased, stripped name string.
    """
    if not isinstance(name, str):
        return ""
    suffixes = salt_suffixes or CFG.SALT_SUFFIXES
    name_lower = name.lower().strip()
    for suffix in suffixes:
        if name_lower.endswith(" " + suffix):
            return name_lower[: -(len(suffix) + 1)].strip()
    return name_lower


def get_drug_name(row: pd.Series) -> str:
    """
    Extract drug name from a DICTrank row, with fallback.

    Design Decision #1: Use 'Active Ingredient(s)', fall back to
    'Generic/Proper Name(s)'. 27/1,211 rows need the fallback.

    Args:
        row: DataFrame row from labeled_compounds.csv

    Returns:
        Drug name string (never empty -- confirmed safe across full dataset)
    """
    name = row.get("Active Ingredient(s)", None)
    if pd.isna(name) or not isinstance(name, str) or not name.strip():
        name = row.get("Generic/Proper Name(s)", "")
    return str(name).strip()


# ---------------------------------------------------------------------------
# RDKit Chemistry Helpers
# ---------------------------------------------------------------------------

def get_parent_smiles(smiles: str) -> Dict[str, Any]:
    """
    Extract the parent drug molecule from a potentially multi-component SMILES.

    Background: ~30% of DICTrank compounds resolve to salt forms (multi-component
    SMILES joined by '.'). We keep the largest fragment by heavy-atom count.
    Every stripped fragment is returned for logging -- never silently discarded.

    Args:
        smiles: Canonical SMILES string (may contain '.' separator)

    Returns:
        dict with keys:
            parent_smiles  : str|None -- SMILES of largest fragment
            n_fragments    : int      -- number of components (1 = no salt)
            stripped       : list[str]-- SMILES of removed fragments (for audit log)
            error          : str|None -- error message if RDKit parsing failed
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"parent_smiles": None, "n_fragments": 0,
                    "stripped": [], "error": "MolFromSmiles returned None"}

        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) == 1:
            return {"parent_smiles": Chem.MolToSmiles(frags[0]),
                    "n_fragments": 1, "stripped": [], "error": None}

        # Sort descending by heavy-atom count
        frags_sorted = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
        parent = frags_sorted[0]
        stripped = [Chem.MolToSmiles(f) for f in frags_sorted[1:]]
        return {"parent_smiles": Chem.MolToSmiles(parent),
                "n_fragments": len(frags), "stripped": stripped, "error": None}

    except Exception as e:
        return {"parent_smiles": None, "n_fragments": 0,
                "stripped": [], "error": str(e)}


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    """
    Convert a SMILES string to its InChIKey (27-character structural identifier).

    InChIKey matching is immune to naming/salt-suffix differences -- used to
    confirm genuine LINCS absence (not a matching bug) in the trace-through.

    Args:
        smiles: Valid canonical SMILES string

    Returns:
        InChIKey string or None if parsing failed
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return rdkit_inchi.MolToInchiKey(mol)
    except Exception:
        return None


def validate_smiles(smiles: str) -> bool:
    """Return True if RDKit can parse and sanitize the SMILES without error."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# HTTP Request Helpers (with retry + rate limiting)
# ---------------------------------------------------------------------------

def robust_get(url: str, params: Optional[Dict] = None,
               timeout: int = CFG.CHEMBL_TIMEOUT,
               max_retry: int = CFG.CHEMBL_MAX_RETRY,
               sleep: float = CFG.CHEMBL_SLEEP,
               logger: Optional[logging.Logger] = None) -> Optional[Dict]:
    """
    Perform a GET request with exponential backoff retries.

    Args:
        url:       Endpoint URL
        params:    Query parameters dict
        timeout:   Per-request timeout in seconds
        max_retry: Number of retries on failure
        sleep:     Base sleep between requests (doubled on each retry)
        logger:    Optional logger for warnings

    Returns:
        Parsed JSON dict on success, None on all retries exhausted.
    """
    for attempt in range(1, max_retry + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            time.sleep(sleep)
            return resp.json()
        except requests.exceptions.Timeout:
            msg = f"Timeout on attempt {attempt}/{max_retry}: {url}"
        except requests.exceptions.HTTPError as e:
            msg = f"HTTP {e.response.status_code} on attempt {attempt}/{max_retry}: {url}"
        except Exception as e:
            msg = f"Request error on attempt {attempt}/{max_retry}: {e}"

        if logger:
            logger.warning(msg)
        if attempt < max_retry:
            time.sleep(sleep * (2 ** attempt))  # exponential backoff

    return None


# ---------------------------------------------------------------------------
# Filesystem Helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*paths: str):
    """Create directories (and parents) if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def file_exists_nonempty(path: str) -> bool:
    """Return True if file exists and has non-zero size."""
    return os.path.isfile(path) and os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_class_weight(labels: np.ndarray) -> float:
    """
    Compute the positive class weight for BCEWithLogitsLoss.

    pos_weight = n_negative / n_positive
    This balances loss contribution between majority (neg) and minority (pos) class.

    Args:
        labels: 1D array of binary labels (0/1)

    Returns:
        float: pos_weight for BCEWithLogitsLoss
    """
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        raise ValueError("No positive samples found -- cannot compute class weight")
    return n_neg / n_pos
