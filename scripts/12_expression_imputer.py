"""
12_expression_imputer.py — Production script for training the molecular structure-to-biology imputer.
Translates 2048-bit fingerprints into 978 landmark gene expression z-scores to resolve
missing biological data constraints.
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch.utils.data import TensorDataset, DataLoader

# Insert root to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from scripts.utils import set_seed

class ExpressionImputer(nn.Module):
    """
    Multi-Layer Perceptron (MLP) mapping 2048-bit Morgan Fingerprints
    to 978-dimensional gene expression vectors.
    """
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
            nn.Linear(hidden_dim, gene_dim)
        )
        
    def forward(self, fp):
        return self.net(fp)

def compute_morgan_fps(smiles_list, n_bits=2048):
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fp = np.zeros(n_bits, dtype=np.float32)
        else:
            fp_bit = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            fp = np.array(fp_bit, dtype=np.float32)
        fps.append(fp)
    return np.array(fps)

def main():
    set_seed(42)
    print("="*60)
    print("SCRIPT: 12_expression_imputer.py -- Biological Imputation Engine")
    print("="*60)
    
    # Load raw manifest and matched data
    manifest = pd.read_csv(os.path.join(CFG.PROCESSED_DIR, 'graph_manifest.csv'))
    matched_df = pd.read_csv(CFG.LINCS_MATCHED_CSV)
    expr_df = pd.read_csv(CFG.EXPRESSION_CSV, index_col=0)
    
    matched_only = matched_df[matched_df["lincs_match"] & matched_df["sig_id"].notna()]
    name_to_sigid = dict(zip(matched_only["query_name"], matched_only["sig_id"]))
    
    # Filter valid compounds
    valid_records = manifest[manifest["name"].isin(name_to_sigid.keys()) & (manifest["status"] != "failed")].copy()
    
    smiles_list = valid_records["smiles"].tolist()
    names = valid_records["name"].tolist()
    
    X_fps = compute_morgan_fps(smiles_list)
    
    y_expr = []
    for name in names:
        sig_id = name_to_sigid[name]
        expr_row = expr_df.loc[sig_id]
        if isinstance(expr_row, pd.DataFrame):
            expr_row = expr_row.iloc[0]
        y_expr.append(expr_row.values.astype(np.float32))
    y_expr = np.array(y_expr)
    
    print(f"Dataset Size: X={X_fps.shape} (fingerprints) -> y={y_expr.shape} (gene profiles)")
    
    # Train/Val Split (80/20)
    n_samples = len(X_fps)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split = int(n_samples * 0.8)
    
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_ds = TensorDataset(torch.tensor(X_fps[train_idx]), torch.tensor(y_expr[train_idx]))
    val_ds = TensorDataset(torch.tensor(X_fps[val_idx]), torch.tensor(y_expr[val_idx]))
    
    # drop_last=True prevents batchnorm crash on batch size 1 remainder
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExpressionImputer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    imputer_ckpt = os.path.join(CFG.CHECKPOINTS_DIR, 'expression_imputer.pt')
    
    print("Training MLP translation layers on GPU...")
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for fps_b, expr_b in train_loader:
            fps_b, expr_b = fps_b.to(device), expr_b.to(device)
            optimizer.zero_grad()
            pred = model(fps_b)
            loss = criterion(pred, expr_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(fps_b)
            
        train_loss /= len(train_idx)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for fps_b, expr_b in val_loader:
                fps_b, expr_b = fps_b.to(device), expr_b.to(device)
                pred = model(fps_b)
                loss = criterion(pred, expr_b)
                val_loss += loss.item() * len(fps_b)
        val_loss /= len(val_idx)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, imputer_ckpt)
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
            
    print(f"Imputer successfully trained. Best Val MSE: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {imputer_ckpt}")

if __name__ == "__main__":
    main()
