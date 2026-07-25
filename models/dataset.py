"""
dataset.py -- PyTorch Dataset and DataLoader builders for Cardiotox-Fusion.

Contains:
  - CardiotoxGraphDataset: PyG InMemoryDataset for structure-only training
  - CardiotoxFusionDataset: Paired graph + expression dataset for fusion training
  - build_dataloaders(): Creates stratified train/val/test splits with proper loaders

Stratified split is MANDATORY (Design Decision #6):
  The LINCS-covered subset is more imbalanced than the full dataset
  (~38% positive vs ~28% negative coverage). A random split could
  produce unrepresentative test sets. Stratified split preserves class
  balance across all three splits.
"""

import os
import sys
import pickle
from typing import Optional, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split

from config import CFG
from scripts.utils import get_logger, set_seed

logger = get_logger("dataset")


# ---------------------------------------------------------------------------
# Structure-Only Dataset (GNN baseline)
# ---------------------------------------------------------------------------

class CardiotoxGraphDataset(Dataset):
    """
    PyTorch Dataset for molecular graphs (structure-only GNN baseline).

    Loads pre-built .pt graph files from data/processed/graphs/ based on
    the manifest CSV produced by 04_build_graphs.py.

    Args:
        manifest_df: DataFrame with columns: name, graph_path, label, status
        filter_status: Only include rows with this status (default: any non-'failed')
    """

    def __init__(self, manifest_df: pd.DataFrame, filter_status: str = None):
        if filter_status:
            manifest_df = manifest_df[manifest_df["status"] == filter_status]
        else:
            manifest_df = manifest_df[manifest_df["status"] != "failed"]

        self.records = manifest_df.reset_index(drop=True)
        logger.info(f"CardiotoxGraphDataset: {len(self.records)} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Data:
        row = self.records.iloc[idx]
        graph = torch.load(row["graph_path"], weights_only=False)
        # Ensure label is float tensor for BCEWithLogitsLoss
        graph.y = torch.tensor([row["label"]], dtype=torch.float)
        return graph

    @property
    def labels(self) -> np.ndarray:
        """Return all labels as numpy array (for stratified split)."""
        return self.records["label"].values


# ---------------------------------------------------------------------------
# Fusion Dataset (Graph + Expression)
# ---------------------------------------------------------------------------

class CardiotoxFusionDataset(Dataset):
    """
    Paired dataset: molecular graph + LINCS gene expression vector.

    Used for both the Transformer-only baseline and the full Fusion model.

    Args:
        manifest_df:   DataFrame from graph_manifest.csv (must include 'name' col)
        expr_df:       DataFrame from expression_matrix.csv, index=sig_id/name, cols=978 genes
        name_to_sigid: Dict mapping compound name -> LINCS sig_id
        include_graph: If True, loads .pt graph files (for fusion model)
                       If False, returns only expression (for Transformer-only baseline)
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        expr_df: pd.DataFrame,
        name_to_sigid: dict,
        include_graph: bool = True,
    ):
        self.include_graph = include_graph
        self.expr_df = expr_df
        self.name_to_sigid = name_to_sigid

        # Only keep samples that have both graph AND expression
        valid_names = {name for name, sig_id in name_to_sigid.items() if sig_id in expr_df.index}
        manifest_df = manifest_df[
            (manifest_df["status"] != "failed") &
            (manifest_df["name"].isin(valid_names))
        ]
        self.records = manifest_df.reset_index(drop=True)
        logger.info(f"CardiotoxFusionDataset: {len(self.records)} paired samples "
                    f"(graph+expression)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records.iloc[idx]
        name = row["name"]
        label = torch.tensor([row["label"]], dtype=torch.float)

        # Gene expression vector
        sig_id = self.name_to_sigid.get(name)
        expr_row = self.expr_df.loc[sig_id]
        if isinstance(expr_row, pd.DataFrame):
            expr_row = expr_row.iloc[0]
            
        expr = torch.tensor(
            expr_row.values.astype(np.float32),
            dtype=torch.float,
        )  # shape: (978,)

        result = {"gene_expr": expr, "label": label, "name": name}

        if self.include_graph:
            graph = torch.load(row["graph_path"], weights_only=False)
            graph.y = label
            result["graph"] = graph

        return result

    @property
    def labels(self) -> np.ndarray:
        return self.records["label"].values


# ---------------------------------------------------------------------------
# Stratified Data Split
# ---------------------------------------------------------------------------

def stratified_split(
    dataset: Dataset,
    labels: np.ndarray,
    train_frac: float = CFG.TRAIN_FRAC,
    val_frac: float = CFG.VAL_FRAC,
    seed: int = CFG.RANDOM_SEED,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create stratified train/val/test splits from a dataset.

    Stratification is MANDATORY for this project (Design Decision #6):
    The LINCS-covered subset has unequal positive coverage between label
    classes (~38% positive, ~28% negative). Random splits would produce
    unrepresentative test sets.

    Args:
        dataset:    Source dataset (any __getitem__ Dataset)
        labels:     1D array of binary labels for stratification
        train_frac: Fraction for training (default 0.70)
        val_frac:   Fraction for validation (default 0.15)
        seed:       Random seed for reproducibility

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    test_frac = 1.0 - train_frac - val_frac

    indices = np.arange(len(dataset))

    # First split: train vs (val+test)
    idx_train, idx_temp = train_test_split(
        indices, test_size=(val_frac + test_frac),
        stratify=labels, random_state=seed,
    )

    # Second split: val vs test from the temp set
    labels_temp = labels[idx_temp]
    val_size_relative = val_frac / (val_frac + test_frac)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=(1 - val_size_relative),
        stratify=labels_temp, random_state=seed,
    )

    train_ds = _SubsetDataset(dataset, idx_train)
    val_ds   = _SubsetDataset(dataset, idx_val)
    test_ds  = _SubsetDataset(dataset, idx_test)

    logger.info(f"Split sizes -- Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    logger.info(f"Train pos rate: {labels[idx_train].mean():.2%}")
    logger.info(f"Val pos rate  : {labels[idx_val].mean():.2%}")
    logger.info(f"Test pos rate : {labels[idx_test].mean():.2%}")

    return train_ds, val_ds, test_ds


class _SubsetDataset(Dataset):
    """Wraps a Dataset with a subset of indices."""
    def __init__(self, dataset: Dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    @property
    def labels(self) -> np.ndarray:
        if hasattr(self.dataset, "labels"):
            return self.dataset.labels[self.indices]
        raise AttributeError("Underlying dataset has no 'labels' property")


# ---------------------------------------------------------------------------
# Collate Functions
# ---------------------------------------------------------------------------

def graph_collate_fn(batch: list) -> Batch:
    """Collate function for graph-only batches (GNN baseline)."""
    return Batch.from_data_list(batch)


def fusion_collate_fn(batch: list) -> dict:
    """
    Collate function for fusion batches (graph + expression).

    Returns:
        dict with:
            'graph'    : Batched PyG Data object
            'gene_expr': Tensor (batch_size, 978)
            'label'    : Tensor (batch_size, 1)
            'names'    : list of compound names
    """
    has_graph = "graph" in batch[0]

    collated = {
        "gene_expr": torch.stack([item["gene_expr"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "names": [item["name"] for item in batch],
    }
    if has_graph:
        collated["graph"] = Batch.from_data_list([item["graph"] for item in batch])

    return collated


# ---------------------------------------------------------------------------
# DataLoader Builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    dataset: Dataset,
    labels: np.ndarray,
    batch_size: int = CFG.BATCH_SIZE,
    use_weighted_sampler: bool = True,
    num_workers: int = 0,
    collate_fn=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders with stratified splits.

    Args:
        dataset:              Source dataset
        labels:               All binary labels (for stratification)
        batch_size:           Batch size
        use_weighted_sampler: If True, use WeightedRandomSampler on training
                              set to handle class imbalance
        num_workers:          DataLoader workers (0 = main process only)
        collate_fn:           Custom collate function

    Returns:
        (train_loader, val_loader, test_loader)
    """
    set_seed(CFG.RANDOM_SEED)
    train_ds, val_ds, test_ds = stratified_split(dataset, labels)

    # Weighted random sampler for training (handles class imbalance)
    train_sampler = None
    if use_weighted_sampler:
        train_labels = train_ds.labels
        class_counts = np.bincount(train_labels.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels.astype(int)]
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float),
            num_samples=len(train_ds),
            replacement=True,
        )
        logger.info(f"WeightedRandomSampler: class weights = {class_weights}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
