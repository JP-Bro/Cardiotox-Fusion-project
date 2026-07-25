"""
gnn_model.py -- Graph Neural Network (GNN) branch for molecular structure encoding.

Architecture:
  - Input: Molecular graph (atoms as nodes with 14-dim features, bonds as edges with 4-dim features)
  - Backbone: GINEConv layers (Graph Isomorphism Network with Edge features)
    GINEConv was chosen over GCNConv because:
      a) It's theoretically as powerful as the Weisfeiler-Lehman graph isomorphism test
      b) It incorporates edge features natively (critical for bond-type information)
      c) GCNConv ignores edge features entirely
  - Pooling: Global mean pool + Global max pool (concatenated) for graph-level readout
    Using both captures different aspects: mean=average structure, max=strongest features
  - Output: CFG.EMBED_DIM (128) dimensional embedding vector

Usage:
  from models.gnn_model import GNNEncoder
  model = GNNEncoder()
  out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # shape: (batch, 128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import CFG


class GNNEncoder(nn.Module):
    """
    GNN encoder for molecular structure.

    Takes a batched molecular graph and produces a fixed-size embedding
    for each graph in the batch.

    Args:
        node_feat_dim:  Dimensionality of node (atom) features. Default: CFG.NODE_FEATURE_DIM
        edge_feat_dim:  Dimensionality of edge (bond) features. Default: CFG.EDGE_FEATURE_DIM
        hidden_dim:     Hidden dimension of GINEConv layers. Default: CFG.GNN_HIDDEN_DIM
        embed_dim:      Output embedding dimension. Default: CFG.EMBED_DIM
        num_layers:     Number of GINEConv layers. Default: CFG.GNN_NUM_LAYERS
        dropout:        Dropout rate. Default: CFG.GNN_DROPOUT
    """

    def __init__(
        self,
        node_feat_dim: int = CFG.NODE_FEATURE_DIM,
        edge_feat_dim: int = CFG.EDGE_FEATURE_DIM,
        hidden_dim: int = CFG.GNN_HIDDEN_DIM,
        embed_dim: int = CFG.EMBED_DIM,
        num_layers: int = CFG.GNN_NUM_LAYERS,
        dropout: float = CFG.GNN_DROPOUT,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # --- Input projection ---
        # Project raw node features to hidden_dim before GNN layers
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Edge feature projection (shared across all layers)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)

        # --- GINEConv layers ---
        # Each layer: MLP(concat(node, aggregated_neighbors))
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # --- Skip connections (residual) ---
        # Added after layer 2 onwards to help with gradient flow in deeper networks
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim)

        # --- Readout projection ---
        # Global pool concatenates mean + max -> 2 * hidden_dim
        # Project down to embed_dim
        self.readout_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x          : Node features -- shape (total_atoms_in_batch, node_feat_dim)
            edge_index : Edge connectivity -- shape (2, total_edges_in_batch)
            edge_attr  : Edge features -- shape (total_edges_in_batch, edge_feat_dim)
            batch      : Batch vector -- shape (total_atoms_in_batch,)
                         Maps each atom to its graph in the batch (0-indexed)

        Returns:
            Tensor of shape (batch_size, embed_dim) -- one embedding per graph
        """
        # Project inputs to hidden_dim
        h = self.node_proj(x)
        e = self.edge_proj(edge_attr)

        # GINEConv layers with batch norm, ReLU, dropout, and residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, e)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            # Residual connection after first layer
            if i > 0:
                h_new = h_new + self.skip_proj(h)

            h = h_new

        # Global pooling: mean + max concatenated
        h_mean = global_mean_pool(h, batch)   # shape: (batch_size, hidden_dim)
        h_max = global_max_pool(h, batch)     # shape: (batch_size, hidden_dim)
        h_graph = torch.cat([h_mean, h_max], dim=-1)  # shape: (batch_size, 2*hidden_dim)

        # Project to embed_dim
        out = self.readout_proj(h_graph)  # shape: (batch_size, embed_dim)
        return out


class GNNClassifier(nn.Module):
    """
    Complete GNN-only cardiotoxicity classifier (structure-only baseline).

    Adds an MLP classification head on top of GNNEncoder.
    Used for the structure-only baseline model (Script 06).

    Args:
        Same as GNNEncoder, plus:
        mlp_hidden_dims: List of hidden layer dims for classification head.
        mlp_dropout:     Dropout in classification MLP.
    """

    def __init__(
        self,
        node_feat_dim: int = CFG.NODE_FEATURE_DIM,
        edge_feat_dim: int = CFG.EDGE_FEATURE_DIM,
        hidden_dim: int = CFG.GNN_HIDDEN_DIM,
        embed_dim: int = CFG.EMBED_DIM,
        num_layers: int = CFG.GNN_NUM_LAYERS,
        dropout: float = CFG.GNN_DROPOUT,
        mlp_hidden_dims: list = CFG.MLP_HIDDEN_DIMS,
        mlp_dropout: float = CFG.MLP_DROPOUT,
    ):
        super().__init__()
        self.encoder = GNNEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.classifier = _build_mlp_head(embed_dim, mlp_hidden_dims, mlp_dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Returns:
            logits -- shape (batch_size, 1). Apply sigmoid for probabilities.
        """
        emb = self.encoder(x, edge_index, edge_attr, batch)
        return self.classifier(emb)

    def get_embedding(self, x, edge_index, edge_attr, batch):
        """Return the embedding vector (before classification head)."""
        return self.encoder(x, edge_index, edge_attr, batch)


def _build_mlp_head(in_dim: int, hidden_dims: list, dropout: float) -> nn.Sequential:
    """Build a standard MLP classification head (Linear=>BN=>ReLU=>Dropout)*N => Linear(1)."""
    layers = []
    current_dim = in_dim
    for h_dim in hidden_dims:
        layers += [
            nn.Linear(current_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ]
        current_dim = h_dim
    layers.append(nn.Linear(current_dim, 1))  # output logit
    return nn.Sequential(*layers)
