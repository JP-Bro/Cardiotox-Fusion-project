"""
04_build_graphs.py -- Step 4: Convert SMILES strings to PyTorch Geometric molecular graphs.

What this script does:
  1. Loads lincs_matched_compounds.csv (output of 03_match_lincs.py)
  2. For each compound with valid SMILES:
       - Parses molecule with RDKit
       - Extracts node features (atom-level)
       - Extracts edge features (bond-level)
       - Builds a torch_geometric.data.Data object
  3. Saves graphs to data/processed/graphs/{compound_name}.pt
  4. Saves a manifest CSV linking compound names to graph file paths

Node features (NODE_FEATURE_DIM = 14):
  - Atom type: one-hot over [C, N, O, S, F, Cl, Br, I, P, other] (10 dims)
  - Degree (normalized)
  - Is aromatic (bool)
  - Formal charge (normalized)
  - Number of attached Hs (normalized)

Edge features (EDGE_FEATURE_DIM = 4):
  - Bond type: one-hot over [SINGLE, DOUBLE, TRIPLE, AROMATIC]
  - Edges are bidirectional (undirected graph with 2 directed edges per bond)

Design decision:
  - Salt stripping is NOT re-applied here (already done in 02_resolve_smiles.py)
    The 'parent_smiles' column already contains the cleaned, single-component SMILES
  - Any compound that fails RDKit graph construction is logged and excluded
  - Graphs are saved as individual .pt files (not one big tensor) to allow
    efficient on-demand loading during training

Run with:
  python scripts/04_build_graphs.py
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from torch_geometric.data import Data

from config import CFG
from scripts.utils import get_logger, ensure_dirs, validate_smiles

LOG_FILE = os.path.join(CFG.RESULTS_DIR, "logs", "04_build_graphs.log")
logger = get_logger("04_build_graphs", LOG_FILE)


# ---------------------------------------------------------------------------
# Atom / Bond Feature Extractors
# ---------------------------------------------------------------------------

ATOM_TYPE_MAP = {at: i for i, at in enumerate(CFG.ATOM_TYPES[:-1])}  # excludes 'other'
BOND_TYPE_MAP = {bt: i for i, bt in enumerate(CFG.BOND_TYPES)}


def get_atom_features(atom: Chem.Atom) -> list:
    """
    Extract a fixed-size feature vector for a single RDKit atom.

    Features (total 14 dimensions):
      [0:10]  Atom type one-hot (C/N/O/S/F/Cl/Br/I/P/other)
      [10]    Degree (number of bonds) -- clipped at 6, normalized by 6
      [11]    Is aromatic (0 or 1)
      [12]    Formal charge -- clipped to [-2, 2], normalized by 2
      [13]    Number of attached Hs -- clipped at 4, normalized by 4

    Args:
        atom: RDKit Atom object

    Returns:
        list of 14 floats
    """
    # Atom type one-hot (10 dims)
    atom_symbol = atom.GetSymbol()
    type_vec = [0.0] * len(CFG.ATOM_TYPES)
    idx = ATOM_TYPE_MAP.get(atom_symbol, len(CFG.ATOM_TYPES) - 1)  # -1 = 'other'
    type_vec[idx] = 1.0

    feats = type_vec + [
        min(atom.GetDegree(), 6) / 6.0,           # degree (norm)
        float(atom.GetIsAromatic()),                # aromaticity
        max(-2.0, min(2.0, atom.GetFormalCharge())) / 2.0,  # formal charge (norm)
        min(atom.GetTotalNumHs(), 4) / 4.0,        # H count (norm)
    ]
    return feats


def get_bond_features(bond: Chem.Bond) -> list:
    """
    Extract a fixed-size feature vector for a single RDKit bond.

    Features (total 4 dimensions):
      One-hot over [SINGLE, DOUBLE, TRIPLE, AROMATIC]

    Args:
        bond: RDKit Bond object

    Returns:
        list of 4 floats
    """
    bond_type = bond.GetBondTypeAsDouble()
    type_str_map = {1.0: "SINGLE", 2.0: "DOUBLE", 3.0: "TRIPLE", 1.5: "AROMATIC"}
    type_str = type_str_map.get(bond_type, "SINGLE")
    feat = [0.0] * len(CFG.BOND_TYPES)
    feat[BOND_TYPE_MAP.get(type_str, 0)] = 1.0
    return feat


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def smiles_to_graph(smiles: str, label: int, compound_name: str) -> Data | None:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Graph representation:
      - Nodes: atoms with 14-dim feature vectors
      - Edges: bonds (bidirectional) with 4-dim feature vectors
      - Target: binary cardiotoxicity label

    Args:
        smiles:        Canonical SMILES string (parent/salt-stripped)
        label:         Binary cardiotoxicity label (0 or 1)
        compound_name: Name for logging purposes

    Returns:
        torch_geometric.data.Data or None if parsing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"  RDKit failed to parse SMILES for {compound_name}: {smiles[:50]}")
        return None

    # --- Node features ---
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(get_atom_features(atom))

    x = torch.tensor(atom_feats, dtype=torch.float)  # shape: (n_atoms, 14)

    # --- Edge index + edge features ---
    # Edges are bidirectional (add both directions for each bond)
    edge_indices = []
    edge_feats = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = get_bond_features(bond)

        edge_indices += [[i, j], [j, i]]  # both directions
        edge_feats += [feat, feat]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    else:
        # Isolated atom (e.g., single-atom molecule) -- no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, CFG.EDGE_FEATURE_DIM), dtype=torch.float)

    y = torch.tensor([label], dtype=torch.float)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        smiles=smiles,
        name=compound_name,
    )
    return graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build molecular graphs from SMILES")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild all graphs even if .pt file already exists")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SCRIPT: 04_build_graphs.py")
    logger.info("PURPOSE: SMILES -> PyTorch Geometric molecular graphs")
    logger.info(f"  Node feature dim: {CFG.NODE_FEATURE_DIM}")
    logger.info(f"  Edge feature dim: {CFG.EDGE_FEATURE_DIM}")
    logger.info("=" * 60)

    ensure_dirs(CFG.GRAPHS_DIR, os.path.join(CFG.RESULTS_DIR, "logs"))

    # Load matched compounds (all of them -- both LINCS-covered and structure-only)
    if not os.path.isfile(CFG.LINCS_MATCHED_CSV):
        raise FileNotFoundError(
            f"Matched compounds CSV not found at {CFG.LINCS_MATCHED_CSV}. "
            "Run 03_match_lincs.py first."
        )
    df = pd.read_csv(CFG.LINCS_MATCHED_CSV)
    # Also load structure-only compounds (those not in LINCS)
    if os.path.isfile(CFG.SMILES_CSV):
        df_all_smiles = pd.read_csv(CFG.SMILES_CSV)
        df_all_smiles = df_all_smiles[df_all_smiles["status"] == "success"]
        logger.info(f"Total compounds with SMILES: {len(df_all_smiles)}")
        logger.info(f"LINCS-matched compounds: {df['lincs_match'].sum() if 'lincs_match' in df.columns else 'N/A'}")
        # Build graphs for all compounds with SMILES (not just LINCS-matched)
        # The GNN-only baseline uses the full set
        build_df = df_all_smiles
    else:
        build_df = df

    n_built = 0
    n_skipped = 0
    n_failed = 0
    manifest_rows = []

    for i, row in build_df.iterrows():
        name = str(row.get("query_name", f"compound_{i}"))
        smiles = str(row.get("parent_smiles", ""))
        label = int(row.get("cardiotox_label", -1))

        if not smiles or smiles == "nan":
            n_failed += 1
            logger.warning(f"  [{i}] {name}: empty SMILES -- skipping")
            continue

        # Safe filename (replace special chars)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        graph_path = os.path.join(CFG.GRAPHS_DIR, f"{safe_name}.pt")

        if os.path.isfile(graph_path) and not args.overwrite:
            n_skipped += 1
            manifest_rows.append({
                "name": name, "graph_path": graph_path,
                "label": label, "smiles": smiles,
                "status": "cached",
            })
            continue

        graph = smiles_to_graph(smiles, label, name)
        if graph is None:
            n_failed += 1
            manifest_rows.append({
                "name": name, "graph_path": "",
                "label": label, "smiles": smiles,
                "status": "failed",
            })
            continue

        torch.save(graph, graph_path)
        n_built += 1

        if n_built % 100 == 0:
            logger.info(f"  [{n_built} built] {name}: {graph.num_nodes} atoms, {graph.num_edges} edges")

        manifest_rows.append({
            "name": name, "graph_path": graph_path,
            "label": label, "smiles": smiles,
            "status": "built",
            "n_atoms": graph.num_nodes,
            "n_edges": graph.num_edges // 2,  # undirected count
        })

    # Save manifest
    manifest_path = os.path.join(CFG.PROCESSED_DIR, "graph_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    logger.info("=" * 60)
    logger.info("GRAPH BUILD SUMMARY")
    logger.info(f"  Built       : {n_built}")
    logger.info(f"  Skipped     : {n_skipped} (already exist)")
    logger.info(f"  Failed      : {n_failed}")
    logger.info(f"  Manifest    : {manifest_path}")
    logger.info("=" * 60)
    logger.info("04_build_graphs.py COMPLETE")


if __name__ == "__main__":
    main()
