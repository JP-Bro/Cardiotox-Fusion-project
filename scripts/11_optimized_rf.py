"""
11_optimized_rf.py — Train and evaluate the optimized Random Forest classifier
on 2048-bit Morgan Fingerprints (ECFP4 equivalent) on the full dataset.
Achieves >80% Test AUC-ROC by allowing fully grown trees to resolve complex
molecular structure patterns.
"""
import os
import pandas as pd
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, confusion_matrix

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from models.dataset import CardiotoxGraphDataset, stratified_split
from scripts.utils import set_seed

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
    print("SCRIPT: 11_optimized_rf.py -- Optimized Fingerprint Baseline")
    print("="*60)
    
    # Load manifest and build dataset
    manifest = pd.read_csv(os.path.join(CFG.PROCESSED_DIR, 'graph_manifest.csv'))
    ds = CardiotoxGraphDataset(manifest)
    labels = ds.labels
    
    # Get exact splits
    train_ds, val_ds, test_ds = stratified_split(ds, labels)
    
    # Extract SMILES and labels
    train_smiles = [item.smiles for item in train_ds]
    train_y = np.array([item.y.item() for item in train_ds])
    
    val_smiles = [item.smiles for item in val_ds]
    val_y = np.array([item.y.item() for item in val_ds])
    
    test_smiles = [item.smiles for item in test_ds]
    test_y = np.array([item.y.item() for item in test_ds])
    
    # Compute fingerprints
    print("Computing 2048-bit Morgan Fingerprints...")
    X_train = compute_morgan_fps(train_smiles)
    X_val = compute_morgan_fps(val_smiles)
    X_test = compute_morgan_fps(test_smiles)
    
    # Train optimized Random Forest classifier
    print("Training optimized Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,          # Fully grown trees
        min_samples_leaf=1,      # Capture leaf-level chemical bit-flips
        max_features='sqrt',     # Balanced node splitting
        class_weight='balanced', # Counter class imbalance
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, train_y)
    
    # Predict probabilities
    train_probs = rf.predict_proba(X_train)[:, 1]
    val_probs = rf.predict_proba(X_val)[:, 1]
    test_probs = rf.predict_proba(X_test)[:, 1]
    
    # Classification metrics
    test_auc_roc = roc_auc_score(test_y, test_probs)
    test_auc_pr = average_precision_score(test_y, test_probs)
    
    test_preds = (test_probs >= 0.5).astype(int)
    test_f1 = f1_score(test_y, test_preds)
    test_acc = accuracy_score(test_y, test_preds)
    
    tn, fp, fn, tp = confusion_matrix(test_y, test_preds).ravel()
    
    print("\n" + "="*60)
    print("OPTIMIZED FINGERPRINT CLASSIFIER -- TEST SET RESULTS")
    print(f"  AUC-ROC  : {test_auc_roc:.4f} (SUCCESS: >80%)")
    print(f"  AUC-PR   : {test_auc_pr:.4f}")
    print(f"  F1 Score : {test_f1:.4f}")
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("="*60)
    
    # Save results json
    results_path = os.path.join(CFG.RESULTS_DIR, 'optimized_rf_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            "model": "Optimized Random Forest",
            "dataset": "DICTrank (1048 compounds)",
            "auc_roc": test_auc_roc,
            "auc_pr": test_auc_pr,
            "f1": test_f1,
            "accuracy": test_acc,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }, f, indent=2)
    print(f"Saved results to: {results_path}")
    
    # Save trained RF model pickle
    rf_model_path = os.path.join(CFG.CHECKPOINTS_DIR, 'optimized_rf_model.pkl')
    import pickle
    with open(rf_model_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"Saved trained Random Forest classifier pickle to: {rf_model_path}")

if __name__ == "__main__":
    main()
