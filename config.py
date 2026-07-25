"""
config.py -- Central configuration for the Cardiotox-Fusion project.

ALL paths, hyperparameters, and constants live here.
No hardcoded values anywhere else in the codebase -- import from here instead.

Usage:
    from config import CFG
    print(CFG.DICTRANK_PATH)
"""

import os

# ---------------------------------------------------------------------------
# Project root -- auto-detected relative to this file
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class CFG:
    """Central configuration namespace. All fields are class attributes -- no instantiation needed."""

    # -----------------------------------------------------------------------
    # Paths -- Data
    # -----------------------------------------------------------------------
    DATA_DIR          = os.path.join(PROJECT_ROOT, "data")
    RAW_DIR           = os.path.join(PROJECT_ROOT, "data", "raw")
    PROCESSED_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
    GRAPHS_DIR        = os.path.join(PROJECT_ROOT, "data", "processed", "graphs")
    LINCS_RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "lincs")

    # Source data files
    DICTRANK_PATH     = os.path.join(RAW_DIR, "dictrank_dataset_508.xlsx")
    CREDIBLEMEDS_PATH = os.path.join(RAW_DIR, "crediblemeds_dta_list_2026-07-23.pdf")
    LINCS_SIG_INFO    = os.path.join(LINCS_RAW_DIR, "sig_info.txt.gz")
    LINCS_PERT_INFO   = os.path.join(LINCS_RAW_DIR, "pert_info.txt.gz")
    # Level 5 expression matrix -- ~5 GB, downloaded only after match rate confirmed
    LINCS_GCTX        = os.path.join(LINCS_RAW_DIR, "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx")

    # Processed data files (outputs of pipeline scripts)
    LABELS_CSV        = os.path.join(PROCESSED_DIR, "labeled_compounds.csv")
    SMILES_CSV        = os.path.join(PROCESSED_DIR, "compounds_with_smiles.csv")
    LINCS_MATCHED_CSV = os.path.join(PROCESSED_DIR, "lincs_matched_compounds.csv")
    EXPRESSION_CSV    = os.path.join(PROCESSED_DIR, "expression_matrix.csv")
    TRACE_LOG         = os.path.join(PROCESSED_DIR, "trace_through_log.txt")

    # -----------------------------------------------------------------------
    # Paths -- Models & Results
    # -----------------------------------------------------------------------
    MODELS_DIR        = os.path.join(PROJECT_ROOT, "models")
    RESULTS_DIR       = os.path.join(PROJECT_ROOT, "results")
    CHECKPOINTS_DIR   = os.path.join(PROJECT_ROOT, "models", "checkpoints")

    GNN_CHECKPOINT    = os.path.join(CHECKPOINTS_DIR, "gnn_best.pt")
    TRANSFORMER_CKPT  = os.path.join(CHECKPOINTS_DIR, "transformer_best.pt")
    FUSION_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "fusion_best.pt")

    RESULTS_CSV       = os.path.join(RESULTS_DIR, "model_results.csv")
    BASELINE_REPORT   = os.path.join(RESULTS_DIR, "baseline_comparison.md")
    INTERP_REPORT     = os.path.join(RESULTS_DIR, "interpretability_validation.md")

    # -----------------------------------------------------------------------
    # DICTrank / Label Settings
    # -----------------------------------------------------------------------
    # Binarization: no -> 0, less/most -> 1, ambiguous -> dropped
    LABEL_MAP         = {"no": 0, "less": 1, "most": 1}
    DROP_LABEL        = "ambiguous"
    RANDOM_SEED       = 42

    # -----------------------------------------------------------------------
    # ChEMBL API Settings
    # -----------------------------------------------------------------------
    CHEMBL_API_BASE   = "https://www.ebi.ac.uk/chembl/api/data"
    CHEMBL_TIMEOUT    = 20          # seconds per request
    CHEMBL_SLEEP      = 0.4         # seconds between requests (rate-limit courtesy)
    CHEMBL_MAX_RETRY  = 3           # retries on transient errors

    # Salt suffixes to strip for LINCS name-matching
    SALT_SUFFIXES     = [
        "phosphate", "hydrochloride", "hcl", "tromethamine", "sodium",
        "sulfate", "maleate", "acetate", "citrate", "mesylate",
        "tosylate", "fumarate", "tartrate", "bromide", "chloride",
        "gluconate", "succinate", "besylate", "malate", "lactate",
    ]

    # -----------------------------------------------------------------------
    # LINCS Settings
    # -----------------------------------------------------------------------
    # Condition selection (Design Decision #5 from trace-through)
    LINCS_CELL_LINE   = "HA1E"      # most consistent, non-transformed reference line
    LINCS_DOSE        = 10.0        # µM -- highest available dose
    LINCS_TIME        = "24 h"      # 24-hour timepoint

    # GEO download URL for Level 5 expression matrix
    LINCS_GCTX_URL    = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file="
        "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz"
    )

    # -----------------------------------------------------------------------
    # Molecular Graph Features
    # -----------------------------------------------------------------------
    ATOM_TYPES        = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "other"]
    BOND_TYPES        = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    NODE_FEATURE_DIM  = len(ATOM_TYPES) + 4   # atom_type(10) + degree + aromatic + charge + H
    EDGE_FEATURE_DIM  = len(BOND_TYPES)        # 4

    # -----------------------------------------------------------------------
    # Model Hyperparameters
    # -----------------------------------------------------------------------
    # Embedding dimensions
    EMBED_DIM         = 128         # output dim of both GNN and Transformer branches
    FUSION_DIM        = 256         # = 2 * EMBED_DIM after cross-attention concat

    # GNN
    GNN_HIDDEN_DIM    = 256
    GNN_NUM_LAYERS    = 4
    GNN_DROPOUT       = 0.3

    # Transformer
    TRANS_D_MODEL     = 128         # model dimension inside Transformer
    TRANS_NHEAD       = 4           # attention heads
    TRANS_NUM_LAYERS  = 3           # encoder layers
    TRANS_DIM_FF      = 512         # feed-forward hidden dim
    TRANS_DROPOUT     = 0.2
    NUM_GENES         = 978         # LINCS L1000 landmark genes

    # Cross-attention fusion
    FUSION_NHEAD      = 4
    FUSION_DROPOUT    = 0.2

    # MLP head
    MLP_HIDDEN_DIMS   = [256, 128, 64]
    MLP_DROPOUT       = 0.3

    # -----------------------------------------------------------------------
    # Training Hyperparameters
    # -----------------------------------------------------------------------
    BATCH_SIZE        = 32
    MAX_EPOCHS        = 150
    LEARNING_RATE     = 3e-4
    WEIGHT_DECAY      = 1e-4
    LR_WARMUP_EPOCHS  = 5
    EARLY_STOP_PATIENCE = 15        # epochs without val-AUC improvement

    # Dataset splits
    TRAIN_FRAC        = 0.70
    VAL_FRAC          = 0.15
    TEST_FRAC         = 0.15        # must sum to 1.0

    # Positive class weight for BCEWithLogitsLoss (rough imbalance 327/96 in LINCS-covered set)
    # Will be recomputed dynamically in training scripts based on actual split
    POS_WEIGHT_FALLBACK = 2.5

    # -----------------------------------------------------------------------
    # Interpretability
    # -----------------------------------------------------------------------
    TOP_K_GENES       = 20          # report top-K genes by attention weight
    TOP_K_ATOMS       = 10          # report top-K atoms by GNN attention weight

    # Known cardiotoxicity toxicophores (structural alerts from literature)
    # Used to validate GNN attention -- not assert explanation
    KNOWN_TOXICOPHORES = [
        "anthracycline",
        "quinone",
        "hERG blocker",
        "nitrogen mustard",
        "reactive oxygen species generator",
    ]
