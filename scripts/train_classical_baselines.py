"""
train_classical_baselines.py -- Classical ML benchmark for Cardiotox-Fusion (canonical baseline script;
`04_baselines.py` simply runs this file).

Structure branch : 2048-bit Morgan fingerprints (radius 2)  -> RF, SVC, LR, HistGBM, KNN, MLP
Biology branch   : 978 L1000 landmark z-scores (HA1E/10uM/24h) -> LR (L2/L1/ElasticNet), SVC, RF, ExtraTrees, HistGBM, MLP

Protocol
  * Fit on the scaffold-split TRAIN set only (scaler too); evaluate once on the scaffold TEST set.
    The validation set is unused here (no hyper-parameter search), so there is nothing to over-fit to it.
  * Primary pre-specified pair (the two controls requested for Phase 1): Morgan-FP Random Forest (200 trees) and
    L1000 Logistic Regression (L2). Every other model is EXPLORATORY: with 17 models scored on one small test set,
    the best of them is optimistically biased.
  * Metrics: AUC-ROC and balanced accuracy are primary. AUC-PR is reported next to its floor (the test-set positive
    prevalence) because the positive class is the MAJORITY class here, which compresses every AUC-PR into [prevalence, 1].
  * 95% CIs are percentile bootstraps over the TEST compounds (seeded; they reflect test-set sampling only, not
    training-set variability).
  * A model is flagged DEGENERATE if it predicts every test compound as toxic (or every one as safe); its AUC-PR
    then carries no information about discrimination.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from scripts.utils import random_classifier_aucpr_interval

SEED = CFG.RANDOM_SEED
N_BOOT = 2000
PRIMARY = {('Structure', 'Random Forest (200 trees)'), ('Biology', 'Logistic Regression (L2)')}

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
scaf = pd.read_csv(os.path.join(CFG.DATA_DIR, 'splits', 'scaffold_split.csv')).set_index('parent_smiles')
expr = pd.read_csv(CFG.EXPRESSION_CSV, index_col=0)
expr.index.name = 'parent_smiles'
assert set(scaf.index) == set(expr.index), "split file and expression matrix cover different compounds"
merged = scaf.join(expr, how='inner').reset_index()
gene_cols = list(expr.columns)
assert len(gene_cols) == CFG.NUM_GENES

train_df, val_df, test_df = [merged[merged['split'] == s].reset_index(drop=True) for s in ('train', 'val', 'test')]
y_train, y_test = train_df['cardiotox_label'].values, test_df['cardiotox_label'].values
prevalence = float(y_test.mean())
n_cohort = len(merged)

print("=" * 80)
print(f"CLASSICAL BASELINES | cohort {n_cohort} skeletons | HA1E / {CFG.LINCS_DOSE} uM / {CFG.LINCS_TIME}")
for nm, d in (('train', train_df), ('val', val_df), ('test', test_df)):
    print(f"  {nm:<5}: {len(d):>3} ({int((d.cardiotox_label==1).sum())} toxic / {int((d.cardiotox_label==0).sum())} safe)")
print("=" * 80)


def fingerprint(smi, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Unparseable SMILES in cohort: {smi}")
    return np.array(list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)), dtype=np.float32)


X_tr_fp = np.stack([fingerprint(s) for s in train_df.parent_smiles])
X_te_fp = np.stack([fingerprint(s) for s in test_df.parent_smiles])
scaler = StandardScaler()
X_tr_ex = scaler.fit_transform(train_df[gene_cols].values)      # scaler fit on TRAIN only
X_te_ex = scaler.transform(test_df[gene_cols].values)

# fixed bootstrap resamples, shared by every model so CIs are comparable and reproducible
_rs = np.random.RandomState(SEED)
BOOT_IDX = [_rs.randint(0, len(y_test), len(y_test)) for _ in range(N_BOOT)]


def boot_ci(y, s, fn, ci=0.95):
    vals = []
    for idx in BOOT_IDX:
        yt = y[idx]
        if yt.min() == yt.max():
            continue
        vals.append(fn(yt, s[idx]))
    return tuple(np.percentile(vals, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100]))


TEST_SCORES = {}


def evaluate(name, modality, model, X_tr, X_te):
    model.fit(X_tr, y_train)
    if hasattr(model, 'predict_proba'):
        score = model.predict_proba(X_te)[:, 1]
    else:
        score = 1.0 / (1.0 + np.exp(-model.decision_function(X_te)))
    TEST_SCORES[f'{modality}: {name}'] = score
    pred = (score >= 0.5).astype(int)
    ap, roc = average_precision_score(y_test, score), roc_auc_score(y_test, score)
    ap_lo, ap_hi = boot_ci(y_test, score, average_precision_score)
    roc_lo, roc_hi = boot_ci(y_test, score, roc_auc_score)
    degenerate = bool(pred.min() == pred.max())
    row = {
        'Modality': modality, 'Model': name, 'Primary': (modality, name) in PRIMARY,
        'AUC-ROC': roc, 'AUC-ROC_95CI_Low': roc_lo, 'AUC-ROC_95CI_High': roc_hi,
        'Balanced_Acc': balanced_accuracy_score(y_test, pred),
        'AUC-PR': ap, 'AUC-PR_95CI_Low': ap_lo, 'AUC-PR_95CI_High': ap_hi,
        'Accuracy': accuracy_score(y_test, pred), 'F1_Score': f1_score(y_test, pred, zero_division=0),
        'Precision': precision_score(y_test, pred, zero_division=0), 'Recall': recall_score(y_test, pred, zero_division=0),
        'Degenerate': degenerate, 'CI_excludes_0.5': bool(roc_lo > 0.5),
    }
    print(f"[{modality:<9}] {name:<34} AUC-ROC {roc:.3f} [{roc_lo:.3f},{roc_hi:.3f}] BalAcc {row['Balanced_Acc']:.3f} "
          f"AUC-PR {ap:.3f} {'DEGENERATE' if degenerate else ''}")
    return row


structure_models = [
    ("Random Forest (200 trees)", RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=-1)),
    ("Random Forest (500 trees)", RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=SEED, n_jobs=-1)),
    ("Logistic Regression (L2)", LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=SEED)),
    ("Support Vector Classifier (RBF)", SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=SEED)),
    ("Support Vector Classifier (Linear)", SVC(C=0.1, kernel='linear', probability=True, class_weight='balanced', random_state=SEED)),
    ("Gradient Boosting (HistGBM)", HistGradientBoostingClassifier(random_state=SEED, class_weight='balanced')),
    ("K-Nearest Neighbors (k=5, Jaccard)", KNeighborsClassifier(n_neighbors=5, metric='jaccard')),
    ("2-Layer MLP (128-64)", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=SEED, early_stopping=True)),
]
biology_models = [
    ("Logistic Regression (L2)", LogisticRegression(C=0.01, class_weight='balanced', max_iter=2000, random_state=SEED)),
    ("Sparse LR (L1 / Lasso)", LogisticRegression(C=0.1, penalty='l1', solver='saga', class_weight='balanced', max_iter=2000, random_state=SEED)),
    ("ElasticNet Logistic Regression", LogisticRegression(C=0.1, penalty='elasticnet', l1_ratio=0.5, solver='saga', class_weight='balanced', max_iter=2000, random_state=SEED)),
    ("Support Vector Classifier (Linear)", SVC(C=0.01, kernel='linear', probability=True, class_weight='balanced', random_state=SEED)),
    ("Support Vector Classifier (RBF)", SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=SEED)),
    ("Random Forest (200 trees)", RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=-1)),
    ("Extra Trees (200 trees)", ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=-1)),
    ("Gradient Boosting (HistGBM)", HistGradientBoostingClassifier(random_state=SEED, class_weight='balanced')),
    ("2-Layer MLP (256-64)", MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=400, random_state=SEED, early_stopping=True)),
]

rows = [evaluate(n, 'Structure', m, X_tr_fp, X_te_fp) for n, m in structure_models]
rows += [evaluate(n, 'Biology', m, X_tr_ex, X_te_ex) for n, m in biology_models]
res = pd.DataFrame(rows)
# test-set scores of every model, so later analyses (paired comparisons with the deep models) never refit a baseline
pd.DataFrame({'parent_smiles': test_df.parent_smiles, 'y': y_test, **TEST_SCORES}).to_csv(
    os.path.join(CFG.RESULTS_DIR, 'baseline_test_predictions.csv'), index=False)

rc_lo, rc_hi = random_classifier_aucpr_interval(y_test, seed=SEED)
random_row = pd.DataFrame([{
    'Modality': 'Baseline', 'Model': 'Random Classifier (Prevalence)', 'Primary': False,
    'AUC-ROC': 0.5, 'AUC-ROC_95CI_Low': np.nan, 'AUC-ROC_95CI_High': np.nan, 'Balanced_Acc': 0.5,
    'AUC-PR': prevalence, 'AUC-PR_95CI_Low': rc_lo, 'AUC-PR_95CI_High': rc_hi,
    'Accuracy': np.nan, 'F1_Score': np.nan, 'Precision': np.nan, 'Recall': np.nan,
    'Degenerate': False, 'CI_excludes_0.5': False}])
out = pd.concat([random_row, res], ignore_index=True)
os.makedirs(CFG.RESULTS_DIR, exist_ok=True)
out.to_csv(os.path.join(CFG.RESULTS_DIR, 'classical_baselines_comparison.csv'), index=False)

# legacy 3-row summary (kept so older references to baseline_results.csv stay valid)
def pick(mod, name):
    return res[(res.Modality == mod) & (res.Model == name)].iloc[0]
rf, lr = pick('Structure', 'Random Forest (200 trees)'), pick('Biology', 'Logistic Regression (L2)')
pd.DataFrame({
    'Model': ['Random Forest (Morgan FP)', 'Logistic Regression (L1000)', 'Random Classifier'],
    'AUC-PR': [rf['AUC-PR'], lr['AUC-PR'], prevalence],
    'AUC-PR CI Low': [rf['AUC-PR_95CI_Low'], lr['AUC-PR_95CI_Low'], rc_lo],
    'AUC-PR CI High': [rf['AUC-PR_95CI_High'], lr['AUC-PR_95CI_High'], rc_hi],
    'AUC-ROC': [rf['AUC-ROC'], lr['AUC-ROC'], 0.5],
    'AUC-ROC CI Low': [rf['AUC-ROC_95CI_Low'], lr['AUC-ROC_95CI_Low'], None],
    'AUC-ROC CI High': [rf['AUC-ROC_95CI_High'], lr['AUC-ROC_95CI_High'], None],
}).to_csv(os.path.join(CFG.RESULTS_DIR, 'baseline_results.csv'), index=False)

# ---------------------------------------------------------------------------
# 2. Report (every number below is computed from `res`; nothing is hand-typed)
# ---------------------------------------------------------------------------
def fmt_row(r):
    flag = ' ‡' if r['Degenerate'] else ''
    star = ' ★' if r['Primary'] else ''
    return (f"| {r['Modality']} | {r['Model']}{star}{flag} | {r['AUC-ROC']:.3f} | [{r['AUC-ROC_95CI_Low']:.3f}, {r['AUC-ROC_95CI_High']:.3f}] "
            f"| {r['Balanced_Acc']:.3f} | {r['AUC-PR']:.3f} | [{r['AUC-PR_95CI_Low']:.3f}, {r['AUC-PR_95CI_High']:.3f}] |")

n_test, n_safe = len(y_test), int((y_test == 0).sum())
lines = [
    "# Classical Machine Learning Baselines Report",
    "",
    "*Auto-generated by `scripts/train_classical_baselines.py`; do not edit by hand.*",
    "",
    f"**Cohort:** strict HA1E / {CFG.LINCS_DOSE} µM / {CFG.LINCS_TIME}, {n_cohort} unique chemical skeletons "
    f"({int(merged.cardiotox_label.sum())} toxic / {int((merged.cardiotox_label==0).sum())} safe).  ",
    f"**Evaluation set:** Bemis-Murcko scaffold test set, N = {n_test} ({n_test - n_safe} toxic / {n_safe} safe; "
    f"{100*prevalence:.1f}% positive).  ",
    f"**Splits:** train {len(train_df)} / val {len(val_df)} / test {n_test}; zero InChIKey-skeleton overlap (`results/pipeline_verification.md`).",
    "",
    f"> **Metric note.** Toxic is the majority class ({100*prevalence:.1f}%), so a random classifier already scores "
    f"AUC-PR ≈ {prevalence:.3f} (95% interval [{rc_lo:.3f}, {rc_hi:.3f}] at this test size) and a model that labels everything toxic "
    "scores nearly as well. AUC-PR is therefore not used to rank models. **AUC-ROC and balanced accuracy are primary**; "
    "AUC-PR is shown beside its floor.",
    "",
    "## 1. Results (test set)",
    "",
    "| Modality | Model | AUC-ROC | 95% CI | Bal-Acc | AUC-PR (floor " + f"{prevalence:.3f}" + ") | 95% CI |",
    "|---|---|:-:|:-:|:-:|:-:|:-:|",
    f"| Baseline | Random classifier | 0.500 | — | 0.500 | {prevalence:.3f} | [{rc_lo:.3f}, {rc_hi:.3f}] |",
]
for mod in ('Structure', 'Biology'):
    for _, r in res[res.Modality == mod].sort_values('AUC-ROC', ascending=False).iterrows():
        lines.append(fmt_row(r))
lines += ["",
          "★ pre-specified primary baseline (Morgan-FP Random Forest; L1000 Logistic Regression). All other rows are exploratory.  ",
          "‡ **Degenerate:** predicts every test compound as the same class, so its AUC-PR carries no discrimination information.", ""]

# computed findings
def mod_stats(m):
    d = res[res.Modality == m]
    return dict(n=len(d), degenerate=int(d.Degenerate.sum()), excl=int(d['CI_excludes_0.5'].sum()),
                best=d.sort_values('AUC-ROC', ascending=False).iloc[0],
                best_ap=d.sort_values('AUC-PR', ascending=False).iloc[0])
S, B = mod_stats('Structure'), mod_stats('Biology')
pS = res[(res.Modality == 'Structure') & res.Primary].iloc[0]
pB = res[(res.Modality == 'Biology') & res.Primary].iloc[0]
lines += ["## 2. Findings (computed from the table)", ""]
lines.append(f"- **Primary structure baseline:** Morgan-FP RF reaches AUC-ROC {pS['AUC-ROC']:.3f} "
             f"[{pS['AUC-ROC_95CI_Low']:.3f}, {pS['AUC-ROC_95CI_High']:.3f}], balanced accuracy {pS['Balanced_Acc']:.3f}. "
             + ("Its interval excludes 0.5." if pS['CI_excludes_0.5'] else "Its interval includes 0.5, so no signal is established."))
lines.append(f"- **Primary biology baseline:** L1000 LR (L2) reaches AUC-ROC {pB['AUC-ROC']:.3f} "
             f"[{pB['AUC-ROC_95CI_Low']:.3f}, {pB['AUC-ROC_95CI_High']:.3f}], balanced accuracy {pB['Balanced_Acc']:.3f}. "
             + ("Its interval excludes 0.5." if pB['CI_excludes_0.5'] else "Its interval includes 0.5: no detectable signal."))
lines.append(f"- **Structure models whose AUC-ROC interval excludes 0.5:** {S['excl']} of {S['n']}. "
             f"**Biology models:** {B['excl']} of {B['n']}. (With {S['n']+B['n']} models tried, a few intervals clearing 0.5 by chance "
             "is expected; the primary pair above is the pre-specified comparison.)")
lines.append(f"- **Degenerate models (predict one class for every compound):** {S['degenerate']} structure, {B['degenerate']} biology "
             f"({S['degenerate']+B['degenerate']} of {S['n']+B['n']}). The highest AUC-PR in the biology table is "
             f"{B['best_ap']['Model']} at {B['best_ap']['AUC-PR']:.3f}"
             + (" — a degenerate model, which is why AUC-PR is not used for ranking." if B['best_ap']['Degenerate']
                else f" (balanced accuracy {B['best_ap']['Balanced_Acc']:.3f})."))
excl_names = ", ".join(res[(res.Modality == 'Biology') & res['CI_excludes_0.5']].Model) or "none"
lines.append("- **What this does and does not show:** the pre-specified L1000 baseline "
             + ("shows no detectable signal above chance" if not pB['CI_excludes_0.5'] else "has an interval that excludes 0.5")
             + f"; exploratory biology models whose interval excludes 0.5: {excl_names}. With {S['n']+B['n']} models scored on "
             f"{n_safe} safe compounds, an isolated exploratory interval clearing 0.5 is unconfirmed until it replicates on other splits. "
             "None of this says whether a deep model will succeed or fail; whether a learned representation recovers signal that classical "
             "approaches miss is the question Phase 2 tests, and a null result there is reportable "
             "(Seal et al. 2024 report L1000 alone near chance, AUC ≈ 0.57).")
lines += ["", "## 3. Limitations", "",
          f"- **Power:** the test set has {n_safe} safe compounds; intervals are wide and model differences are mostly not "
          f"resolvable. A random scorer's AUC-PR interval spans [{rc_lo:.3f}, {rc_hi:.3f}].",
          "- **Selection:** 17 models are scored on one test set; the best-looking result is biased upward. Treat non-primary rows as exploratory.",
          "- **Intervals** are percentile bootstraps over test compounds only; they do not include training-set variability.",
          "- **Coverage:** LINCS provides a strict-condition signature for only part of DICTrank; relaxing cell line / dose adds "
          "signatures per drug, not drugs, so it does not change test-set size.", ""]
with open(os.path.join(CFG.RESULTS_DIR, 'classical_baselines_report.md'), 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))

# markdown snippet reused by README / Phase1 report (primary pair only)
snip = ["| Model | AUC-ROC | 95% CI | Balanced Acc | AUC-PR (floor " + f"{prevalence:.3f}" + ") | 95% CI |", "|---|:-:|:-:|:-:|:-:|:-:|",
        f"| Random classifier | 0.500 | — | 0.500 | {prevalence:.3f} | [{rc_lo:.3f}, {rc_hi:.3f}] |"]
for _, r in res[res.Primary].iterrows():
    snip.append(f"| {r['Modality']}: {r['Model']} | {r['AUC-ROC']:.3f} | [{r['AUC-ROC_95CI_Low']:.3f}, {r['AUC-ROC_95CI_High']:.3f}] | "
                f"{r['Balanced_Acc']:.3f} | {r['AUC-PR']:.3f} | [{r['AUC-PR_95CI_Low']:.3f}, {r['AUC-PR_95CI_High']:.3f}] |")
with open(os.path.join(CFG.RESULTS_DIR, 'baseline_table.md'), 'w', encoding='utf-8') as f:
    f.write("\n".join(snip) + "\n")
print("\nSaved: classical_baselines_comparison.csv, classical_baselines_report.md, baseline_results.csv, baseline_table.md")
