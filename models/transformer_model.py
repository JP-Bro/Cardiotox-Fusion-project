"""
transformer_model.py -- Transformer branch for gene expression (LINCS L1000) encoding.

Architecture:
  - Input: 978-dimensional gene expression vector (one value per landmark gene)
  - Each gene is treated as a TOKEN (similar to how BERT treats words)
  - Positional encoding: LEARNABLE (not sinusoidal), because genes have no
    natural ordering -- sinusoidal PE would impose a false sequence structure
  - CLS token: prepended to the sequence and used for pooling (BERT-style)
    This avoids the need for explicit pooling and lets the model decide
    which genes to aggregate into the representation
  - TransformerEncoder: 3 layers, 4 attention heads, d_model=128
  - Output: CFG.EMBED_DIM (128) dimensional embedding vector

Design consideration:
  Important biological caveat: LINCS cell lines are NOT cardiomyocytes.
  The biology branch learns a general cross-tissue transcriptional toxicity
  signature. This matches Seal et al.'s approach and their ~0.76 AUC-ROC.
  This limitation MUST be stated in the final paper -- not glossed over.

Usage:
  from models.transformer_model import TransformerEncoder
  model = TransformerEncoder()
  out = model(gene_expr)  # gene_expr shape: (batch, 978), out shape: (batch, 128)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import CFG


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embedding for gene expression tokens.

    Why learnable (not sinusoidal):
      Sinusoidal PE encodes *position in a sequence*, implying gene 1 is
      somehow adjacent to gene 2 in a meaningful way. Genes have no such
      natural ordering in an expression vector -- the 978 genes are in an
      arbitrary order defined by the LINCS assay design.

      Learnable PE lets the model assign each gene position a learned
      identity vector, without imposing false sequential structure.

    Args:
        n_positions: Number of positions (978 genes + 1 CLS token = 979)
        d_model:     Embedding dimension
    """

    def __init__(self, n_positions: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(n_positions, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return positional embeddings for positions 0..seq_len-1."""
        positions = torch.arange(seq_len, device=device)
        return self.pe(positions)  # shape: (seq_len, d_model)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for LINCS L1000 gene expression data.

    Takes a batch of 978-dimensional gene expression vectors and produces
    fixed-size 128-dim embedding vectors (one per sample in the batch).

    Args:
        n_genes:        Number of LINCS landmark genes. Default: CFG.NUM_GENES (978)
        d_model:        Internal transformer dimension. Default: CFG.TRANS_D_MODEL
        nhead:          Number of attention heads. Default: CFG.TRANS_NHEAD
        num_layers:     Number of TransformerEncoder layers. Default: CFG.TRANS_NUM_LAYERS
        dim_feedforward: FF layer hidden dim. Default: CFG.TRANS_DIM_FF
        dropout:        Dropout rate. Default: CFG.TRANS_DROPOUT
        embed_dim:      Output embedding dimension. Default: CFG.EMBED_DIM
    """

    def __init__(
        self,
        n_genes: int = CFG.NUM_GENES,
        d_model: int = CFG.TRANS_D_MODEL,
        nhead: int = CFG.TRANS_NHEAD,
        num_layers: int = CFG.TRANS_NUM_LAYERS,
        dim_feedforward: int = CFG.TRANS_DIM_FF,
        dropout: float = CFG.TRANS_DROPOUT,
        embed_dim: int = CFG.EMBED_DIM,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_genes = n_genes

        # --- Input projection ---
        # Each gene value (scalar) is projected to d_model dimensions
        # This gives each gene its own representation vector
        self.input_proj = nn.Linear(1, d_model)

        # --- CLS token ---
        # A learnable vector prepended to each sequence for pooling
        # After encoding, the CLS output is used as the sequence representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # --- Positional encoding ---
        # +1 for CLS token position
        self.pos_enc = LearnablePositionalEncoding(n_genes + 1, d_model)

        # --- Layer norm before transformer (pre-norm style) ---
        self.pre_norm = nn.LayerNorm(d_model)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",       # GELU > ReLU in Transformer contexts
            batch_first=True,        # (batch, seq, d_model) convention
            norm_first=True,         # Pre-norm (more stable than post-norm)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # --- Output projection ---
        # Project d_model -> embed_dim
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embed_dim),
        )

        # --- Initialize weights ---
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for projection layers."""
        for module in [self.input_proj, self.output_proj]:
            for layer in (module if isinstance(module, nn.Sequential) else [module]):
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, gene_expr: torch.Tensor) -> torch.Tensor:
        """
        Encode gene expression profiles into fixed-size embedding vectors.

        Args:
            gene_expr: Gene expression tensor -- shape (batch_size, n_genes)
                       Values should be z-scored Level 5 LINCS expression values.

        Returns:
            Embedding tensor -- shape (batch_size, embed_dim)
        """
        B = gene_expr.size(0)

        # Reshape to (B, n_genes, 1) and project to (B, n_genes, d_model)
        tokens = self.input_proj(gene_expr.unsqueeze(-1))  # (B, 978, d_model)

        # Prepend CLS token: (B, 979, d_model)
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)           # (B, 979, d_model)

        # Add positional encodings
        pos = self.pos_enc(tokens.size(1), tokens.device)  # (979, d_model)
        tokens = tokens + pos.unsqueeze(0)                 # (B, 979, d_model)

        # Pre-layer norm
        tokens = self.pre_norm(tokens)

        # Transformer encoding
        encoded = self.transformer(tokens)                 # (B, 979, d_model)

        # Extract CLS token output (position 0)
        cls_out = encoded[:, 0, :]                         # (B, d_model)

        # Project to output embedding dim
        out = self.output_proj(cls_out)                    # (B, embed_dim)
        return out

    def get_gene_attention_weights(self, gene_expr: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for each gene from the first transformer layer.

        Used for interpretability: which genes does the model attend to most?
        Note: attention != causation -- must be validated against known biology.

        Args:
            gene_expr: shape (batch_size, n_genes)

        Returns:
            Attention weights -- shape (batch_size, n_genes)
            (averaged over heads, CLS-to-gene attention only)
        """
        B = gene_expr.size(0)
        self.eval()

        with torch.no_grad():
            tokens = self.input_proj(gene_expr.unsqueeze(-1))
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            pos = self.pos_enc(tokens.size(1), tokens.device)
            tokens = tokens + pos.unsqueeze(0)
            tokens = self.pre_norm(tokens)

            # Get attention weights from first layer
            first_layer = self.transformer.layers[0]
            attn_output, attn_weights = first_layer.self_attn(
                tokens, tokens, tokens,
                need_weights=True,
                average_attn_weights=True,  # average over heads
            )

        # CLS token attends to all genes: row 0, cols 1..979
        gene_attn = attn_weights[:, 0, 1:]  # (B, n_genes)
        return gene_attn


class TransformerClassifier(nn.Module):
    """
    Complete Transformer-only cardiotoxicity classifier (biology-only baseline).

    Adds MLP classification head on top of TransformerEncoder.
    Used for biology-only baseline model (Script 07).
    """

    def __init__(
        self,
        n_genes: int = CFG.NUM_GENES,
        d_model: int = CFG.TRANS_D_MODEL,
        nhead: int = CFG.TRANS_NHEAD,
        num_layers: int = CFG.TRANS_NUM_LAYERS,
        dim_feedforward: int = CFG.TRANS_DIM_FF,
        dropout: float = CFG.TRANS_DROPOUT,
        embed_dim: int = CFG.EMBED_DIM,
        mlp_hidden_dims: list = None,
        mlp_dropout: float = CFG.MLP_DROPOUT,
    ):
        super().__init__()
        mlp_hidden_dims = mlp_hidden_dims or CFG.MLP_HIDDEN_DIMS
        self.encoder = TransformerEncoder(
            n_genes=n_genes, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, embed_dim=embed_dim,
        )
        self.classifier = _build_mlp_head(embed_dim, mlp_hidden_dims, mlp_dropout)

    def forward(self, gene_expr: torch.Tensor) -> torch.Tensor:
        """Returns logits -- shape (batch_size, 1). Apply sigmoid for probabilities."""
        emb = self.encoder(gene_expr)
        return self.classifier(emb)

    def get_embedding(self, gene_expr: torch.Tensor) -> torch.Tensor:
        """Return the embedding vector (before classification head)."""
        return self.encoder(gene_expr)


def _build_mlp_head(in_dim: int, hidden_dims: list, dropout: float) -> nn.Sequential:
    """MLP head: (Linear -> BN -> ReLU -> Dropout) * N -> Linear(1)."""
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
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)
