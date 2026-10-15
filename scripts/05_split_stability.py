"""
05_split_stability.py -- How much do the primary baselines move with the choice of scaffold split?

The single frozen scaffold test set has only about a dozen safe compounds, so its AUC-ROC is a noisy estimate. This script
measures that noise and gives a tighter pooled estimate, for the two pre-specified baselines:
    Structure : Morgan-FP (2048, r=2) + Random Forest (200 trees, balanced)
    Biology   : L1000 landmark z-scores + Logistic Regression (L2, C=0.01, balanced; scaler fit on train folds only)

A) 30 random scaffold splits (70/15/15 by compound count; whole scaffold groups stay together; very large
   groups go to train). Reports the distribution of test AUC-ROC / balanced accuracy.
B) Repeated 5-fold scaffold-grouped cross-validation (5 repeats, different group shuffles). Out-of-fold scores are
   pooled over all cohort compounds, so the AUC rests on every safe compound (not 14). 95% CI = bootstrap over compounds.

Usage: python scripts/05_split_stability.py [splits|cv|report]   (stages cache to results/split_stability_*.csv)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from scripts.utils import scaffold_group_folds

SEED = CFG.RANDOM_SEED
N_SPLITS, N_REPEATS, N_FOLDS, N_BOOT = 30, 5, 5, 2000
R = CFG.RESULTS_DIR
mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

scaf = pd.read_csv(os.path.join(CFG.DATA_DIR, 'splits', 'scaffold_split.csv')).set_index('parent_smiles')
expr = pd.read_csv(CFG.EXPRESSION_CSV, index_col=0)
expr.index.name = 'parent_smiles'
df = scaf.join(expr, how='inner').reset_index()
y = df.cardiotox_label.values
groups = df.scaffold.values.astype(str)
n = len(df)
Xfp = np.stack([np.array(list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)), dtype=np.float32)
                for s in df.parent_smiles])
Xex = df[list(expr.columns)].values
uniq, inv = np.unique(groups, return_inverse=True)
sizes = np.bincount(inv)


def fit_score(train_idx, test_idx):
    """Return {model: scores on test_idx}; everything (incl. scaler) fit on train_idx only."""
    out = {}
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=-1)
    rf.fit(Xfp[train_idx], y[train_idx])
    out['Structure: Morgan-FP RF'] = rf.predict_proba(Xfp[test_idx])[:, 1]
    sc = StandardScaler().fit(Xex[train_idx])
    lr = LogisticRegression(C=0.01, class_weight='balanced', max_iter=2000, random_state=SEED)
    lr.fit(sc.transform(Xex[train_idx]), y[train_idx])
    out['Biology: L1000 LR (L2)'] = lr.predict_proba(sc.transform(Xex[test_idx]))[:, 1]
    return out


def random_scaffold_split(seed, min_neg_test=5):
    rs = np.random.RandomState(seed)
    for _ in range(200):
        big = [i for i in range(len(uniq)) if sizes[i] > 0.075 * n]
        rest = [i for i in range(len(uniq)) if sizes[i] <= 0.075 * n]
        rs.shuffle(rest)
        tr, va, te, ct, cv = list(big), [], [], 0, 0
        t_target = int(round(0.15 * n))
        for i in rest:
            if ct < t_target:
                te.append(i); ct += sizes[i]
            elif cv < t_target:
                va.append(i); cv += sizes[i]
            else:
                tr.append(i)
        te_idx = np.where(np.isin(inv, te))[0]
        if (y[te_idx] == 0).sum() >= min_neg_test:
            return np.where(np.isin(inv, tr + va))[0], te_idx
    raise RuntimeError("could not draw a split with enough safe test compounds")


if mode in ('splits', 'all'):
    rows = []
    for s in range(N_SPLITS):
        tr, te = random_scaffold_split(1000 + s)
        for m, sc_ in fit_score(tr, te).items():
            rows.append(dict(split=s, model=m, n_test=len(te), n_safe_test=int((y[te] == 0).sum()),
                             auc_roc=roc_auc_score(y[te], sc_), bal_acc=balanced_accuracy_score(y[te], (sc_ >= .5).astype(int))))
        print(f"split {s}: test n={len(te)}, safe={(y[te]==0).sum()}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(R, 'split_stability_splits.csv'), index=False)

if mode in ('cv', 'all'):
    rows, boot, oof_rep0 = [], {}, {}
    rs_boot = np.random.RandomState(SEED)
    for r in range(N_REPEATS):
        fold = scaffold_group_folds(groups, N_FOLDS, 5000 + r)   # shared with 07_train_deep.py
        oof = {m: np.zeros(n) for m in ['Structure: Morgan-FP RF', 'Biology: L1000 LR (L2)']}
        for f in range(N_FOLDS):
            te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
            for m, sc_ in fit_score(tr, te).items():
                oof[m][te] = sc_
        for m, sc_ in oof.items():
            auc = roc_auc_score(y, sc_)
            rows.append(dict(repeat=r, model=m, auc_roc=auc, bal_acc=balanced_accuracy_score(y, (sc_ >= .5).astype(int))))
            if r == 0:
                oof_rep0[m] = sc_
                vals = []
                for _ in range(N_BOOT):
                    idx = rs_boot.randint(0, n, n)
                    if y[idx].min() != y[idx].max():
                        vals.append(roc_auc_score(y[idx], sc_[idx]))
                boot[m] = np.percentile(vals, [2.5, 97.5])
        print(f"cv repeat {r} done", flush=True)
    pd.DataFrame({'parent_smiles': df.parent_smiles, 'fold': scaffold_group_folds(groups, N_FOLDS, 5000), 'y': y,
                  **{m: v for m, v in oof_rep0.items()}}).to_csv(os.path.join(R, 'split_stability_oof.csv'), index=False)
    out = pd.DataFrame(rows)
    out['ci_low'] = out.model.map(lambda m: boot[m][0]); out['ci_high'] = out.model.map(lambda m: boot[m][1])
    out.to_csv(os.path.join(R, 'split_stability_cv.csv'), index=False)

if mode in ('report', 'all'):
    sp = pd.read_csv(os.path.join(R, 'split_stability_splits.csv'))
    cv = pd.read_csv(os.path.join(R, 'split_stability_cv.csv'))
    L = ["# Split Stability of the Primary Baselines", "",
         "*Auto-generated by `scripts/05_split_stability.py`.*", "",
         f"Cohort: {n} skeletons ({int(y.sum())} toxic / {int((y==0).sum())} safe). The frozen scaffold split has {int(((df.split=='test')&(df.cardiotox_label==0)).sum())} safe test compounds; "
         "this study measures how much results depend on that one split.", "",
         f"## A. {N_SPLITS} random scaffold splits (70/15/15; test AUC-ROC)", "",
         "| Model | Mean | SD | Min | Max | % splits AUC-ROC > 0.5 | Mean bal-acc | Safe in test (mean) |", "|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for m, g in sp.groupby('model'):
        L.append(f"| {m} | {g.auc_roc.mean():.3f} | {g.auc_roc.std():.3f} | {g.auc_roc.min():.3f} | {g.auc_roc.max():.3f} | "
                 f"{100*(g.auc_roc>.5).mean():.0f}% | {g.bal_acc.mean():.3f} | {g.n_safe_test.mean():.1f} |")
    L += ["", f"## B. Repeated {N_FOLDS}-fold scaffold-grouped CV x {N_REPEATS} (pooled out-of-fold AUC-ROC over all {n} compounds)", "",
          "| Model | Mean pooled AUC-ROC | Range over repeats | 95% bootstrap CI (repeat 0) | Mean bal-acc |", "|---|:-:|:-:|:-:|:-:|"]
    verdict = {}
    for m, g in cv.groupby('model'):
        lo, hi = g.ci_low.iloc[0], g.ci_high.iloc[0]
        verdict[m] = (g.auc_roc.iloc[0], lo, hi)   # repeat-0 AUC, the one the bootstrap CI belongs to
        L.append(f"| {m} | {g.auc_roc.mean():.3f} | [{g.auc_roc.min():.3f}, {g.auc_roc.max():.3f}] | [{lo:.3f}, {hi:.3f}] | {g.bal_acc.mean():.3f} |")
    s_m, b_m = verdict['Structure: Morgan-FP RF'], verdict['Biology: L1000 LR (L2)']
    s_sd = sp[sp.model == 'Structure: Morgan-FP RF'].auc_roc.std(); b_sd = sp[sp.model == 'Biology: L1000 LR (L2)'].auc_roc.std()
    L += ["", "## Reading (computed)", "",
          f"- **Split-to-split SD of test AUC-ROC:** structure {s_sd:.3f}, biology {b_sd:.3f}. A single-split difference smaller than about "
          f"{2*max(s_sd,b_sd):.2f} (two SDs) should not be read as a real difference.",
          f"- **Pooled CV (repeat 0), structure:** AUC-ROC {s_m[0]:.3f} [{s_m[1]:.3f}, {s_m[2]:.3f}] — "
          + ("interval excludes 0.5: structure carries signal." if s_m[1] > .5 else "interval includes 0.5: no signal established."),
          f"- **Pooled CV (repeat 0), biology:** AUC-ROC {b_m[0]:.3f} [{b_m[1]:.3f}, {b_m[2]:.3f}] — "
          + ("interval excludes 0.5: some signal." if b_m[1] > .5 else "interval includes 0.5: no signal established."),
          "- **Use for Phase 2:** a deep model must beat the pooled-CV baseline of its own modality by more than the split-to-split SD to count as an improvement.",
          "- Intervals are bootstrap over compounds with fixed pooled predictions; they do not include training variability."]
    open(os.path.join(R, 'split_stability_report.md'), 'w', encoding='utf-8').write("\n".join(L) + "\n")
    print("\n".join(L))
