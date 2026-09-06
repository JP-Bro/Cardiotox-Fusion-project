"""
Run baseline models:
1. Morgan Fingerprint (2048-bit) + Random Forest (structure baseline)
2. Mean L1000 vector (978-dim) + Logistic Regression (expression baseline)
Both evaluated on the same 487-drug scaffold split.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, auc,
                             confusion_matrix)
from sklearn.utils import resample
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG

np.random.seed(CFG.RANDOM_SEED)

# ─── Load split file ────────────────────────────────────────────────────────
scaf_split = pd.read_csv(os.path.join(CFG.DATA_DIR, 'splits', 'scaffold_split.csv'))
expr_matrix = pd.read_csv(CFG.EXPRESSION_CSV, index_col=0)

print("=== DATA LOADED ===")
print(f"Scaffold split: {scaf_split['split'].value_counts().to_dict()}")
print(f"Expression matrix shape: {expr_matrix.shape}")

# ─── Helper: bootstrap CI for AUC-PR ────────────────────────────────────────
def bootstrap_ci(y_true, y_score, metric_fn, n_boot=2000, ci=0.95):
    scores = []
    for _ in range(n_boot):
        idx = resample(range(len(y_true)), random_state=None)
        yt = np.array(y_true)[idx]
        ys = np.array(y_score)[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(metric_fn(yt, ys))
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lo, hi

# ─── Morgan Fingerprint generator ───────────────────────────────────────────
def smiles_to_fp(smi, radius=2, n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return list(fp)
    except:
        pass
    return [0] * n_bits

# ─── Merge expression and split info ────────────────────────────────────────
# Use parent_smiles as key
scaf_split = scaf_split.set_index('parent_smiles')
expr_matrix.index.name = 'parent_smiles'
merged = scaf_split.join(expr_matrix, how='inner')
merged = merged.reset_index()
print(f"\nMerged (split + expression): {len(merged)} rows")

# ─── Split into train/val/test ───────────────────────────────────────────────
train = merged[merged['split'] == 'train']
val   = merged[merged['split'] == 'val']
test  = merged[merged['split'] == 'test']

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print(f"Test negatives: {(test['cardiotox_label']==0).sum()}, positives: {(test['cardiotox_label']==1).sum()}")

# ─── BASELINE 1: Random Forest on Morgan Fingerprints ───────────────────────
print("\n" + "="*60)
print("BASELINE 1: Morgan Fingerprint (2048-bit) + Random Forest")
print("="*60)

X_train_fp = np.array([smiles_to_fp(s) for s in train['parent_smiles']])
X_val_fp   = np.array([smiles_to_fp(s) for s in val['parent_smiles']])
X_test_fp  = np.array([smiles_to_fp(s) for s in test['parent_smiles']])
y_train = train['cardiotox_label'].values
y_val   = val['cardiotox_label'].values
y_test  = test['cardiotox_label'].values

# Tune n_estimators on val set
best_val_ap, best_rf = 0, None
for n_est in [100, 200, 500]:
    rf = RandomForestClassifier(n_estimators=n_est, class_weight='balanced',
                                random_state=CFG.RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train_fp, y_train)
    val_probs = rf.predict_proba(X_val_fp)[:, 1]
    val_ap = average_precision_score(y_val, val_probs)
    if val_ap > best_val_ap:
        best_val_ap, best_rf = val_ap, rf

rf_test_probs = best_rf.predict_proba(X_test_fp)[:, 1]
rf_auc_pr = average_precision_score(y_test, rf_test_probs)
rf_auc_roc = roc_auc_score(y_test, rf_test_probs)
rf_ap_mean, rf_ap_lo, rf_ap_hi = bootstrap_ci(y_test, rf_test_probs, average_precision_score)
rf_roc_mean, rf_roc_lo, rf_roc_hi = bootstrap_ci(y_test, rf_test_probs, roc_auc_score)

print(f"Test AUC-PR : {rf_auc_pr:.4f}  95% CI [{rf_ap_lo:.4f}, {rf_ap_hi:.4f}]")
print(f"Test AUC-ROC: {rf_auc_roc:.4f}  95% CI [{rf_roc_lo:.4f}, {rf_roc_hi:.4f}]")

# ─── BASELINE 2: Logistic Regression on L1000 expression ────────────────────
print("\n" + "="*60)
print("BASELINE 2: Mean L1000 Vector (978-dim) + Logistic Regression")
print("="*60)

gene_cols = [c for c in merged.columns if c not in ['parent_smiles', 'query_name',
    'cardiotox_label', 'DICT_Concern', 'pert_id', 'inchikey_block', 'scaffold',
    'match_method', 'split', 'drug_split', 'scaffold_split']]

X_train_expr = train[gene_cols].values
X_val_expr   = val[gene_cols].values
X_test_expr  = test[gene_cols].values

# Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_expr = scaler.fit_transform(X_train_expr)
X_val_expr   = scaler.transform(X_val_expr)
X_test_expr  = scaler.transform(X_test_expr)

best_val_ap_lr, best_lr = 0, None
for C in [0.001, 0.01, 0.1, 1.0]:
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=2000,
                            random_state=CFG.RANDOM_SEED, solver='lbfgs')
    lr.fit(X_train_expr, y_train)
    val_probs = lr.predict_proba(X_val_expr)[:, 1]
    val_ap = average_precision_score(y_val, val_probs)
    if val_ap > best_val_ap_lr:
        best_val_ap_lr, best_lr = val_ap, lr

lr_test_probs = best_lr.predict_proba(X_test_expr)[:, 1]
lr_auc_pr = average_precision_score(y_test, lr_test_probs)
lr_auc_roc = roc_auc_score(y_test, lr_test_probs)
lr_ap_mean, lr_ap_lo, lr_ap_hi = bootstrap_ci(y_test, lr_test_probs, average_precision_score)
lr_roc_mean, lr_roc_lo, lr_roc_hi = bootstrap_ci(y_test, lr_test_probs, roc_auc_score)

print(f"Test AUC-PR : {lr_auc_pr:.4f}  95% CI [{lr_ap_lo:.4f}, {lr_ap_hi:.4f}]")
print(f"Test AUC-ROC: {lr_auc_roc:.4f}  95% CI [{lr_roc_lo:.4f}, {lr_roc_hi:.4f}]")

# ─── Random classifier baseline ─────────────────────────────────────────────
prevalence = y_test.mean()
print(f"\nRandom Classifier AUC-PR (prevalence): {prevalence:.4f}")

# ─── Save results ────────────────────────────────────────────────────────────
os.makedirs(CFG.RESULTS_DIR, exist_ok=True)
results = {
    'Model': ['Random Forest (Morgan FP)', 'Logistic Regression (L1000)', 'Random Classifier'],
    'AUC-PR': [rf_auc_pr, lr_auc_pr, prevalence],
    'AUC-PR CI Low': [rf_ap_lo, lr_ap_lo, None],
    'AUC-PR CI High': [rf_ap_hi, lr_ap_hi, None],
    'AUC-ROC': [rf_auc_roc, lr_auc_roc, 0.5],
    'AUC-ROC CI Low': [rf_roc_lo, lr_roc_lo, None],
    'AUC-ROC CI High': [rf_roc_hi, lr_roc_hi, None],
}
pd.DataFrame(results).to_csv(os.path.join(CFG.RESULTS_DIR, 'baseline_results.csv'), index=False)
print(f"\nSaved baseline results to results/baseline_results.csv")

print("\n" + "="*60)
print("FINAL BASELINE SUMMARY (Scaffold Split Test Set, N=74)")
print("="*60)
print(f"{'Model':<40} {'AUC-PR':>8} {'95% CI':>20} {'AUC-ROC':>9}")
print("-"*80)
print(f"{'Random Classifier (prevalence)':<40} {prevalence:>8.4f} {'—':>20} {'0.5000':>9}")
print(f"{'LR on L1000 expression':<40} {lr_auc_pr:>8.4f} {f'[{lr_ap_lo:.4f}, {lr_ap_hi:.4f}]':>20} {lr_auc_roc:>9.4f}")
print(f"{'RF on Morgan fingerprints':<40} {rf_auc_pr:>8.4f} {f'[{rf_ap_lo:.4f}, {rf_ap_hi:.4f}]':>20} {rf_auc_roc:>9.4f}")
