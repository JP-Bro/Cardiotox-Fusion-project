"""
fusion_model.py -- Cross-attention fusion of GNN structure and Transformer biology embeddings.

Architecture:
  - Takes embeddings from both branches (each 128-dim)
  - Applies BIDIRECTIONAL cross-attention:
      * Structure queries Biology: which biological signals are most relevant
        to the structural features of this molecule?
      * Biology queries Structure: which structural features correspond to
        the observed biological response?
  - Concatenates both attended outputs -> 256-dim joint representation
  - MLP classification head -> binary prediction

Why cross-attention (not simple concatenation)?
  Concatenation treats structure and biology as fully independent and just
  combines them additively. Cross-attention allows each branch to *query*
  the other -- learning which structural features co-occur with which
  biological responses in cardiotoxic compounds. This is the architectural
  novelty of this project over Seal et al. (2023), which used independent
  feature combination.

Expected performance (based on literature):
  - GNN-only baseline: ~0.84 AUC-ROC (Seal et al.)
  - Transformer-only baseline: ~0.76 AUC-ROC (Seal et al.)
  - Fusion model: target >0.84 -- even a modest improvement is publishable

Usage:
  from models.fusion_model import FusionClassifier
  model = FusionClassifier()
  logits = model(struct_emb, bio_emb)  # shapes: (B, 128), (B, 128) -> (B, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import CFG
from models.gnn_model import GNNEncoder, _build_mlp_head
from models.transformer_model import TransformerEncoder


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block: Query attends to Key/Value.

    Used twice in the fusion:
      1. Structure => Biology: structure queries biology's keys/values
      2. Biology => Structure: biology queries structure's keys/values

    Args:
        embed_dim: Dimensionality of both query and key/value embeddings.
        nhead:     Number of attention heads.
        dropout:   Dropout on attention weights.
    """

    def __init__(
        self,
        embed_dim: int = CFG.EMBED_DIM,
        nhead: int = CFG.FUSION_NHEAD,
        dropout: float = CFG.FUSION_DROPOUT,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query:     Query tensor -- shape (B, embed_dim)
            key_value: Key and Value source tensor -- shape (B, embed_dim)

        Returns:
            (attended_out, attention_weights)
              attended_out:     shape (B, embed_dim)
              attention_weights: shape (B, 1, 1) -- scalar per sample
        """
        # Expand to sequence dim: (B, 1, embed_dim)
        q = self.norm_q(query).unsqueeze(1)
        kv = self.norm_kv(key_value).unsqueeze(1)

        # Cross-attention
        attn_out, attn_weights = self.attn(q, kv, kv, need_weights=True)
        attn_out = attn_out.squeeze(1)  # (B, embed_dim)

        # Residual + feedforward (pre-norm style)
        h = query + self.dropout(attn_out)
        h = h + self.dropout(self.ff(self.norm_out(h)))
        return h, attn_weights.squeeze()


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion module.

    Takes both branch embeddings and produces a fused joint representation.

    Args:
        embed_dim: Dimensionality of each branch embedding (default: 128)
        nhead:     Number of attention heads (default: 4)
        dropout:   Dropout rate (default: 0.2)
    """

    def __init__(
        self,
        embed_dim: int = CFG.EMBED_DIM,
        nhead: int = CFG.FUSION_NHEAD,
        dropout: float = CFG.FUSION_DROPOUT,
    ):
        super().__init__()
        # Structure attends to Biology
        self.struct_to_bio = CrossAttentionBlock(embed_dim, nhead, dropout)
        # Biology attends to Structure
        self.bio_to_struct = CrossAttentionBlock(embed_dim, nhead, dropout)

        # After concatenating attended outputs (256-dim), project to 128-dim fusion rep
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        struct_emb: torch.Tensor,
        bio_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Fuse structure and biology embeddings via bidirectional cross-attention.

        Args:
            struct_emb: Structure embedding -- shape (B, embed_dim)
            bio_emb:    Biology embedding -- shape (B, embed_dim)

        Returns:
            (fused, attention_dict)
              fused: shape (B, embed_dim) -- joint representation
              attention_dict: {
                  'struct_to_bio': attention weights (for interpretability)
                  'bio_to_struct': attention weights
              }
        """
        # Structure attends to Biology
        struct_attended, s2b_weights = self.struct_to_bio(struct_emb, bio_emb)
        # Biology attends to Structure
        bio_attended, b2s_weights = self.bio_to_struct(bio_emb, struct_emb)

        # Concatenate and project
        combined = torch.cat([struct_attended, bio_attended], dim=-1)  # (B, 256)
        fused = self.fusion_proj(combined)  # (B, 128)

        return fused, {
            "struct_to_bio": s2b_weights,
            "bio_to_struct": b2s_weights,
        }


class FusionClassifier(nn.Module):
    """
    End-to-end GNN + Transformer + Cross-Attention Fusion cardiotoxicity classifier.

    This is the main model for the project. It:
      1. Encodes molecular structure via GNNEncoder (GINEConv)
      2. Encodes gene expression via TransformerEncoder (CLS pooling)
      3. Fuses both via bidirectional cross-attention
      4. Classifies via MLP head

    Can be used in two modes:
      a. Full mode (default): Requires both graph and expression data
      b. Structure-only / Biology-only: Use GNNClassifier / TransformerClassifier instead

    Args:
        gnn_kwargs:         Dict of kwargs for GNNEncoder
        transformer_kwargs: Dict of kwargs for TransformerEncoder
        fusion_embed_dim:   Fusion output dim (default: CFG.EMBED_DIM)
        nhead:              Cross-attention heads
        dropout:            Cross-attention dropout
        mlp_hidden_dims:    MLP head hidden dims
        mlp_dropout:        MLP dropout
    """

    def __init__(
        self,
        gnn_kwargs: dict = None,
        transformer_kwargs: dict = None,
        fusion_embed_dim: int = CFG.EMBED_DIM,
        nhead: int = CFG.FUSION_NHEAD,
        dropout: float = CFG.FUSION_DROPOUT,
        mlp_hidden_dims: list = None,
        mlp_dropout: float = CFG.MLP_DROPOUT,
    ):
        super().__init__()
        gnn_kwargs = gnn_kwargs or {}
        transformer_kwargs = transformer_kwargs or {}
        mlp_hidden_dims = mlp_hidden_dims or CFG.MLP_HIDDEN_DIMS

        # GNN encoder (structure branch)
        self.gnn_encoder = GNNEncoder(**gnn_kwargs)

        # Transformer encoder (biology branch)
        self.transformer_encoder = TransformerEncoder(**transformer_kwargs)

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            embed_dim=fusion_embed_dim,
            nhead=nhead,
            dropout=dropout,
        )

        # MLP classification head
        self.classifier = _build_mlp_head(fusion_embed_dim, mlp_hidden_dims, mlp_dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        gene_expr: torch.Tensor,
    ) -> dict:
        """
        Full forward pass through both encoders and fusion.

        Args:
            x          : Node features (total_atoms, 14)
            edge_index : Edge connectivity (2, total_edges)
            edge_attr  : Edge features (total_edges, 4)
            batch      : Batch vector (total_atoms,)
            gene_expr  : Gene expression (batch_size, 978)

        Returns:
            dict with:
                logits:       (B, 1) -- raw classification logits
                struct_emb:   (B, 128) -- structure embedding
                bio_emb:      (B, 128) -- biology embedding
                fused_emb:    (B, 128) -- fused representation
                attn_weights: dict of cross-attention weights
        """
        # Encode structure
        struct_emb = self.gnn_encoder(x, edge_index, edge_attr, batch)  # (B, 128)

        # Encode gene expression
        bio_emb = self.transformer_encoder(gene_expr)  # (B, 128)

        # Fuse via cross-attention
        fused_emb, attn_weights = self.fusion(struct_emb, bio_emb)  # (B, 128)

        # Classify
        logits = self.classifier(fused_emb)  # (B, 1)

        return {
            "logits": logits,
            "struct_emb": struct_emb,
            "bio_emb": bio_emb,
            "fused_emb": fused_emb,
            "attn_weights": attn_weights,
        }

    def count_parameters(self) -> dict:
        """Count trainable parameters per sub-module."""
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "gnn_encoder": count(self.gnn_encoder),
            "transformer_encoder": count(self.transformer_encoder),
            "fusion": count(self.fusion),
            "classifier": count(self.classifier),
            "total": count(self),
        }
