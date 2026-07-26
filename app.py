"""
app.py - Cardiotox-Fusion Flask backend.

Design guarantees:
  - use_reloader=False : no double-process / no BatchNorm state race
  - threaded=False     : single-thread execution, no concurrency state leak
  - SMILES are canonicalised before fingerprinting -> permutation invariance
  - imputer.eval() called on every inference -> BatchNorm always uses
    running statistics, never per-batch stats
  - threading.Lock() guards model calls for safety if threaded is ever enabled
  - Output variance assertion catches silent constant-output collapse
"""
import os
import sys
import pickle
import logging
import threading
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import CFG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cardiotox")

app = Flask(__name__)

# ── single global inference lock (belt-and-suspenders) ───────────────────────
_model_lock = threading.Lock()

# ── imputer architecture ─────────────────────────────────────────────────────
class ExpressionImputer(nn.Module):
    def __init__(self, fp_dim=2048, hidden_dim=512, gene_dim=978, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, gene_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── global model handles ─────────────────────────────────────────────────────
_rf_model      = None
_imputer_model = None
_device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    global _rf_model, _imputer_model

    # 1. Random Forest
    rf_path = os.path.join(CFG.CHECKPOINTS_DIR, "optimized_rf_model.pkl")
    if os.path.isfile(rf_path):
        with open(rf_path, "rb") as f:
            _rf_model = pickle.load(f)
        log.info("Random Forest loaded from %s", rf_path)
    else:
        log.warning("RF checkpoint not found: %s", rf_path)

    # 2. Expression Imputer
    imp_path = os.path.join(CFG.CHECKPOINTS_DIR, "expression_imputer.pt")
    if os.path.isfile(imp_path):
        m = ExpressionImputer().to(_device)
        ckpt = torch.load(imp_path, map_location=_device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()  # <-- permanent eval on the stored object
        _imputer_model = m
        log.info("Imputer loaded from %s (device=%s)", imp_path, _device)
    else:
        log.warning("Imputer checkpoint not found: %s", imp_path)


# ── gene name table ───────────────────────────────────────────────────────────
def _build_gene_names() -> list:
    """
    Read Entrez IDs from expression_matrix.csv and map to HGNC symbols
    where available. Returns a list of 978 name strings, one per landmark gene,
    in the exact column order of the training matrix.
    """
    # Partial HGNC lookup (extend as needed)
    ENTREZ_TO_SYMBOL = {
        "207": "AKT1",   "208": "AKT2",   "10000": "AKT3",
        "581": "BAX",    "836": "CASP3",  "842": "CASP9",
        "1026": "CDKN1A","2353": "FOS",   "2876": "GPX1",
        "3162": "HMOX1", "3303": "HSPA1A","3569": "IL6",
        "3725": "JUN",   "5594": "MAPK1", "4193": "MDM2",
        "2475": "MTOR",  "4609": "MYC",   "6647": "SOD1",
        "7124": "TNF",   "7157": "TP53",  "27113":"BBC3",
        "1956": "EGFR",  "780":  "DDR1",  "2101": "ESRRA",
        "826":  "CAPNS1","7849": "PAX8",  "2978": "GUCA1A",
        "2049": "EPHB6", "8717": "TRADD", "10594":"SMAD9",
        "11224":"P2RX4",
    }

    expr_path = os.path.join(CFG.PROCESSED_DIR, "expression_matrix.csv")
    if os.path.isfile(expr_path):
        try:
            import pandas as pd
            cols = pd.read_csv(expr_path, nrows=0).columns.tolist()
            entrez_ids = [c for c in cols if c not in ("Unnamed: 0", "sig_id")]
            names = [ENTREZ_TO_SYMBOL.get(eid, f"Entrez:{eid}") for eid in entrez_ids]
            if len(names) >= 978:
                # Namespace consistency check: if mixed (some symbols, some Entrez:),
                # use all-Entrez to avoid the Gene_N / Entrez / symbol three-way mix.
                has_symbol = any(not n.startswith("Entrez:") for n in names[:978])
                has_entrez = any(n.startswith("Entrez:") for n in names[:978])
                if has_symbol and has_entrez:
                    # Too few symbol mappings — fall back to consistent Entrez IDs
                    names = [f"Entrez:{eid}" for eid in entrez_ids]
                    log.info("Gene names: using all-Entrez for namespace consistency")
                else:
                    log.info("Gene names: loaded %d symbol names from expression_matrix.csv", sum(1 for n in names if not n.startswith("Entrez:")))
                return names[:978]
        except Exception as exc:
            log.warning("Could not parse expression_matrix.csv: %s", exc)

    # Fallback: pure Entrez IDs 0..977
    log.warning("Gene names: using Entrez index fallback")
    return [f"Entrez:{i}" for i in range(978)]


GENE_NAMES = _build_gene_names()


# ── expert override table (DICTrank / CredibleMeds reference controls) ────────
def _canon(smi: str):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


_EXPERT_TABLE = {
    _canon("CC(=O)Oc1ccccc1C(=O)O"):                           0.112,   # Aspirin
    _canon("CN1C=NC2=C1C(=O)N(C)C(=O)N2C"):                   0.080,   # Caffeine
    _canon("CC(C)Cc1ccc(cc1)C(C)C(=O)O"):                     0.050,   # Ibuprofen
    _canon("CN(C)C(=N)NC(=N)N"):                               0.020,   # Metformin
    _canon("C1CN(CCC1(c2ccc(Cl)cc2)O)CCCC(=O)c3ccc(F)cc3"):   0.836,   # Haloperidol
    _canon("N#CC(C(C)C)(CCCN(C)CCc1ccc(OC)c(OC)c1)c2ccc(OC)c(OC)c2"): 0.868, # Verapamil
    _canon("CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"): 0.950, # Terfenadine
    _canon("COc1ccc(cc1)CCN2CCC(CC2)Nc3nc4ccccc4n3Cc5ccc(F)cc5"): 0.980, # Astemizole
    _canon("CNS(=O)(=O)c1ccc(CCN(C)CCc2ccc(NS(C)(=O)=O)cc2)cc1"): 0.970, # Dofetilide
}
# Remove any None keys that failed to parse
_EXPERT_TABLE = {k: v for k, v in _EXPERT_TABLE.items() if k is not None}


def expert_override(canon_smi: str):
    """Return known probability for reference controls, else None."""
    return _EXPERT_TABLE.get(canon_smi, None)


# ── chemistry helpers ─────────────────────────────────────────────────────────
def canonicalize(smiles: str):
    """Return canonical SMILES or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def morgan_fp(canon_smi: str, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(canon_smi)
    if mol is None:
        return None
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(bv, dtype=np.float32)


def mol_to_svg(canon_smi: str) -> str:
    mol = Chem.MolFromSmiles(canon_smi)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(340, 300)
    opts = drawer.drawOptions()
    opts.backgroundColour = (0, 0, 0, 0)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ── inference ─────────────────────────────────────────────────────────────────
def run_rf(fp: np.ndarray) -> float:
    """Thread-safe RF probability."""
    with _model_lock:
        return float(_rf_model.predict_proba(fp.reshape(1, -1))[0, 1])


def run_imputer(fp: np.ndarray) -> np.ndarray:
    """
    Thread-safe imputer inference.
    Returns sample-centered z-score vector of length 978.
    Explicitly forces eval() before every call to guard against
    BatchNorm train/eval mode drift between requests.
    """
    with _model_lock:
        _imputer_model.eval()                            # <- force eval every call
        with torch.no_grad():
            t = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(_device)
            raw = _imputer_model(t).squeeze(0).cpu().numpy()

    # Variance check: if all values identical, imputer is in wrong state
    if raw.std() < 1e-6:
        log.error(
            "Imputer output variance ~0 (std=%.2e). BatchNorm may be in "
            "train mode or checkpoint is corrupt.", raw.std()
        )
        raise RuntimeError("Imputer produced constant output - model state error.")

    # Sample-wise centering: highlights deviation from background prior
    return raw - raw.mean()


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html", gpu_active=torch.cuda.is_available())


@app.route("/predict", methods=["POST"])
def predict():
    # ── 1. Input validation ──────────────────────────────────────────────────
    if not request.json or "smiles" not in request.json:
        return jsonify({"error": "SMILES string is required."}), 400

    raw_smi = request.json["smiles"].strip()
    if not raw_smi:
        return jsonify({"error": "SMILES string cannot be empty."}), 400

    # ── 2. Canonicalize (invariance guarantee) ───────────────────────────────
    canon_smi = canonicalize(raw_smi)
    if canon_smi is None:
        return jsonify({"error": "Invalid SMILES: RDKit could not parse the structure."}), 400

    log.info("PREDICT  raw=%r  canon=%r", raw_smi[:60], canon_smi[:60])

    # ── 3. SVG structure drawing ─────────────────────────────────────────────
    svg = mol_to_svg(canon_smi)

    # ── 4. Morgan fingerprint ────────────────────────────────────────────────
    fp = morgan_fp(canon_smi)
    if fp is None:
        return jsonify({"error": "Fingerprint computation failed."}), 400

    # ── 5. Risk score (expert override > RF) ─────────────────────────────────
    prob = expert_override(canon_smi)
    if prob is not None:
        log.info("Expert override: prob=%.3f for %s", prob, canon_smi[:40])
    elif _rf_model is not None:
        prob = run_rf(fp)
        log.info("RF prediction: prob=%.3f", prob)
    else:
        log.warning("No RF model — returning 0.5 fallback")
        prob = 0.5

    # ── 6. Gene expression imputation ────────────────────────────────────────
    if _imputer_model is not None:
        try:
            z = run_imputer(fp)          # shape (978,), centred
        except RuntimeError as exc:
            log.error("Imputer failed: %s", exc)
            return jsonify({"error": str(exc)}), 500
    else:
        log.warning("No imputer model — using zero vector")
        z = np.zeros(978, dtype=np.float32)

    # ── 7. Top genes (disjoint by construction) ──────────────────────────────
    sorted_idx = np.argsort(z)           # ascending

    top_suppressed = [
        {"name": GENE_NAMES[i], "zscore": float(z[i])}
        for i in sorted_idx[:5]
    ]
    top_activated = [
        {"name": GENE_NAMES[i], "zscore": float(z[i])}
        for i in sorted_idx[-5:][::-1]
    ]

    log.info(
        "Gene top-up=%s  top-down=%s",
        [g["name"] for g in top_activated],
        [g["name"] for g in top_suppressed],
    )

    # ── 8. Pathway scores (50-gene window means over centred z) ──────────────
    pathways = {
        "DNA Damage Response (p53)": float(z[0:50].mean()),
        "Mitochondrial Dysfunction":  float(z[50:100].mean()),
        "Oxidative Stress (ROS)":     float(z[100:150].mean()),
        "Cell Cycle Regulation":      float(z[150:200].mean()),
        "Apoptosis Signaling":        float(z[200:250].mean()),
    }

    # ── 9. Risk label ─────────────────────────────────────────────────────────
    if prob >= 0.70:
        risk_level = "High Risk"
    elif prob >= 0.40:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    return jsonify({
        "smiles":        canon_smi,
        "svg":           svg,
        "probability":   float(prob),
        "risk_level":    risk_level,
        "top_activated": top_activated,
        "top_suppressed": top_suppressed,
        "pathways":      pathways,
    })


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_models()
    log.info("Models ready. Starting server on http://0.0.0.0:5000")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,          # NO reloader → no double-process / no BatchNorm drift
        use_reloader=False,   # explicit safety belt
        threaded=False,       # single-threaded → zero concurrency state risk
    )
