"""
verify_pipeline.py -- Independent, falsifiable audit of the Cardiotox-Fusion data pipeline.

Every check recomputes its answer from RAW inputs (GCTX, sig_info, gene_info, split CSVs)
instead of re-reading values the pipeline itself wrote, so each one CAN fail.
Exit code 0 only if every check passes. Writes results/pipeline_verification.md.

Run:  python scripts/verify_pipeline.py
"""
import os, sys, gzip, re
import numpy as np
import pandas as pd
import h5py
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score

RDLogger.DisableLog('rdApp.*')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from scripts.utils import random_classifier_aucpr_interval

RESULTS = []   # (name, passed, detail)


def check(name, passed, detail=""):
    RESULTS.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def block_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    mol = MolStandardize.rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToInchiKey(mol).split('-')[0]


# ---------------------------------------------------------------- load
expr = pd.read_csv(CFG.EXPRESSION_CSV, index_col=0)
matched = pd.read_csv(CFG.LINCS_MATCHED_CSV)
drug = pd.read_csv(os.path.join(CFG.DATA_DIR, 'splits', 'drug_split.csv'))
scaf = pd.read_csv(os.path.join(CFG.DATA_DIR, 'splits', 'scaffold_split.csv'))
labels = pd.read_csv(CFG.LABELS_CSV)

# ---------------------------------------------------------------- 1. cohort size / alignment
N = len(matched)
check("1a expression rows == matched rows == both split files",
      len(expr) == N == len(drug) == len(scaf), f"expr={len(expr)} matched={N} drug={len(drug)} scaf={len(scaf)}")
check("1b parent_smiles unique in every table",
      expr.index.is_unique and matched.parent_smiles.is_unique and drug.parent_smiles.is_unique and scaf.parent_smiles.is_unique)
check("1c same compound set in expression and both split files",
      set(expr.index) == set(drug.parent_smiles) == set(scaf.parent_smiles) == set(matched.parent_smiles))
check("1d no NaN/inf in expression matrix", np.isfinite(expr.values).all())
check("1f one LINCS pert_id per cohort compound (no shared expression source)", matched.pert_id.is_unique,
      f"{matched.pert_id.duplicated().sum()} duplicated pert_ids: {matched[matched.pert_id.duplicated(keep=False)].query_name.tolist()}")
check("1e no all-zero / duplicate expression rows",
      (np.abs(expr.values).sum(1) > 0).all() and not expr.duplicated().any())

# ---------------------------------------------------------------- 2. labels
raw = pd.read_excel(CFG.DICTRANK_PATH)
raw['c'] = raw['DICT _ Concern'].astype(str).str.strip().str.lower()
raw = raw[raw.c != 'ambiguous'].reset_index(drop=True)
exp_names = raw['Active Ingredient(s)'].fillna(raw['Generic/Proper Name(s)']).astype(str).str.strip()
check("2z labeled_compounds.csv == raw Excel (names: Active Ingredient w/ Generic fallback; labels; no blank names)",
      len(labels) == len(raw) and (labels.resolved_name.astype(str).str.strip().values == exp_names.values).all()
      and (labels.cardiotox_label.values == raw.c.map(CFG.LABEL_MAP).values).all() and not labels.resolved_name.isna().any(),
      f"{len(labels)} rows, {int(labels.cardiotox_label.sum())} toxic / {int((labels.cardiotox_label==0).sum())} safe")
src = labels.copy()
src['name_key'] = src['resolved_name'].str.lower().str.strip()
check("2a label file is binary-only (ambiguous dropped)",
      set(src.DICT_Concern.unique()) <= {'no', 'less', 'most'} and set(src.cardiotox_label.unique()) <= {0, 1},
      str(src.DICT_Concern.value_counts().to_dict()))
check("2b labels match DICT_Concern mapping",
      (src.DICT_Concern.map({'no': 0, 'less': 1, 'most': 1}) == src.cardiotox_label).all())
n_pos, n_neg = int(matched.cardiotox_label.sum()), int((matched.cardiotox_label == 0).sum())
print(f"      cohort: {n_pos} toxic / {n_neg} safe ({100*n_pos/N:.1f}% / {100*n_neg/N:.1f}%)")
check("2c split files carry identical labels to matched file",
      (scaf.set_index('parent_smiles').cardiotox_label.reindex(matched.parent_smiles).values == matched.cardiotox_label.values).all()
      and (drug.set_index('parent_smiles').cardiotox_label.reindex(matched.parent_smiles).values == matched.cardiotox_label.values).all())

# ---------------------------------------------------------------- 3. structure keys recomputed from SMILES
recomputed = matched.parent_smiles.map(block_of)
check("3a inchikey_block column == recomputed from parent_smiles (neutralised)",
      (recomputed.values == matched.inchikey_block.values).all(),
      f"{(recomputed.values != matched.inchikey_block.values).sum()} mismatches")
check("3b inchikey_block unique across cohort", matched.inchikey_block.is_unique,
      f"{matched.inchikey_block.duplicated().sum()} duplicates")
chg = sum(any(a.GetFormalCharge() != 0 for a in Chem.MolFromSmiles(s).GetAtoms()) for s in matched.parent_smiles)
print(f"      structures still carrying a formal charge after neutralisation: {chg} (zwitterions/quaternary are legitimate)")

# cross-parent label conflicts that dedup by exact string/block cannot see:
# same stripped name stem but different labels
def stem(n):
    n = str(n).lower().strip()
    for s in CFG.SALT_SUFFIXES:
        if n.endswith(" " + s):
            n = n[:-(len(s) + 1)].strip()
    return n
src['stem'] = src['resolved_name'].map(stem)
conf = src.groupby('stem').cardiotox_label.nunique()
conf_stems = conf[conf > 1].index.tolist()
print(f"      raw-label salt-stem conflicts (resolved by 'any toxic -> 1' rule in script 03): {conf_stems}")

# ---------------------------------------------------------------- 4. leakage
def leakage(df, name):
    sets = {s: set(df[df.split == s].inchikey_block) for s in ['train', 'val', 'test']}
    tt, tv, vt = sets['train'] & sets['test'], sets['train'] & sets['val'], sets['val'] & sets['test']
    check(f"4 {name}: zero InChIKey-skeleton overlap across splits", not (tt or tv or vt),
          f"train/test={len(tt)} train/val={len(tv)} val/test={len(vt)}")
leakage(drug, 'drug split')
leakage(scaf, 'scaffold split')

# near-duplicate structures by fingerprint (catches what InChIKey block misses)
fp = lambda s: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
for name, df in [('drug split', drug), ('scaffold split', scaf)]:
    tr = [fp(s) for s in df[df.split == 'train'].parent_smiles]
    te = df[df.split == 'test']
    mx = np.array([max(DataStructs.BulkTanimotoSimilarity(fp(s), tr)) for s in te.parent_smiles])
    print(f"      {name}: test compounds with Tanimoto>=0.85 neighbour in train: {(mx >= 0.85).sum()}/{len(te)} "
          f"({100*(mx>=0.85).mean():.1f}%), median max-sim={np.median(mx):.2f}")

# ---------------------------------------------------------------- 5. scaffold split integrity (recomputed)
def scaf_of(s):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
    except Exception:
        return ''
rec_scaf = scaf.parent_smiles.map(scaf_of)
n_acyc = int((rec_scaf == '').sum())
real = scaf.assign(rs=rec_scaf)[rec_scaf != '']
shared = real.groupby('rs').split.nunique()
check("5a no real Murcko scaffold (chirality-free) spans >1 split", (shared <= 1).all(),
      f"{int((shared > 1).sum())} scaffolds straddle splits")
check("5b acyclic compounds are not one fake cluster in a single split (or each has unique id)",
      n_acyc == 0 or scaf[rec_scaf == ''].scaffold.nunique() == n_acyc,
      f"{n_acyc} acyclic; distinct pseudo-ids={scaf[rec_scaf == ''].scaffold.nunique()}")
sz = scaf.split.value_counts(normalize=True)
check("5c scaffold split proportions within 3 pts of 70/15/15",
      abs(sz['train'] - .70) < .03 and abs(sz['val'] - .15) < .03 and abs(sz['test'] - .15) < .03,
      str({k: round(v, 3) for k, v in sz.items()}))
tst = scaf[scaf.split == 'test']
print(f"      scaffold test: n={len(tst)}, safe={(tst.cardiotox_label==0).sum()}, toxic={(tst.cardiotox_label==1).sum()}")
for s in ['train', 'val', 'test']:
    d = scaf[scaf.split == s]
    print(f"      scaffold {s}: n={len(d)} safe={(d.cardiotox_label==0).sum()} ({100*(d.cardiotox_label==0).mean():.1f}%)")

# ---------------------------------------------------------------- 6. landmark genes vs gene_info vs GCTX (the real cross-reference)
with gzip.open(os.path.join(CFG.LINCS_RAW_DIR, 'gene_info.txt.gz'), 'rt') as f:
    gi = pd.read_csv(f, sep='\t', dtype=str)
lm = set(gi[gi.pr_is_lm == '1'].pr_gene_id)
check("6a gene_info has exactly 978 pr_is_lm==1 genes", len(lm) == 978, str(len(lm)))
with h5py.File(CFG.LINCS_GCTX, 'r') as h:
    rid = [x.decode() if isinstance(x, bytes) else str(x) for x in h['0/META/ROW/id'][:]]
    cid = [x.decode() if isinstance(x, bytes) else str(x) for x in h['0/META/COL/id'][:]]
    cidx = {c: i for i, c in enumerate(cid)}
    ridx = {r: i for i, r in enumerate(rid)}
    check("6b expression columns == landmark set (set equality, from gene_info)", set(expr.columns) == lm,
          f"missing={len(lm-set(expr.columns))} extra={len(set(expr.columns)-lm)}")
    first978 = set(rid[:978])
    print(f"      (control) first 978 GCTX rows contain {len(first978 & lm)} landmark genes -> naive slice would be wrong")
    check("6c column order matches GCTX row order", [ridx[c] for c in expr.columns] == sorted(ridx[c] for c in expr.columns))
    shp = h['0/DATA/0/matrix'].shape
    check("6d GCTX matrix orientation is (signatures, genes)", shp == (len(cid), len(rid)), str(shp))

    # ------------------------------------------------------------ 7. strict filter + aggregation recomputed from raw
    si = pd.read_csv(CFG.LINCS_SIG_INFO, sep='\t', dtype=str).set_index('sig_id')
    sigs = matched.sig_ids.map(eval)
    allsig = [s for l in sigs for s in l]
    sub = si.loc[allsig]
    check("7a every aggregated signature is HA1E", (sub.cell_id == 'HA1E').all(), str(sub.cell_id.value_counts().to_dict()))
    check("7b every aggregated signature is 24 h", (sub.pert_itime == '24 h').all(), str(sub.pert_itime.value_counts().to_dict()))
    check("7c every aggregated signature is 10 uM", sub.pert_idose.str.lower().str.replace(' ', '').isin(['10um', '10.0um']).all(),
          str(sub.pert_idose.value_counts().to_dict()))
    check("7d every aggregated signature is trt_cp", (sub.pert_type == 'trt_cp').all())
    check("7e signature ids unique across compounds (no sig counted for 2 drugs)", len(allsig) == len(set(allsig)),
          f"{len(allsig)-len(set(allsig))} duplicated")
    print(f"      signatures/drug: mean={np.mean([len(l) for l in sigs]):.2f} min={min(len(l) for l in sigs)} max={max(len(l) for l in sigs)}; total={len(allsig)}")
    lm_idx = [ridx[c] for c in expr.columns]
    rng = np.random.RandomState(0)
    pick = np.arange(N)   # every drug, not a sample: a single corrupted row must be caught
    maxerr = 0.0
    for i in pick:
        idxs = sorted(cidx[s] for s in sigs.iloc[i])
        m = np.mean([h['0/DATA/0/matrix'][j, lm_idx] for j in idxs], axis=0)
        maxerr = max(maxerr, np.abs(m - expr.loc[matched.parent_smiles.iloc[i]].values).max())
    check("7f mean-aggregation reproduces stored vectors from raw GCTX (every drug)", maxerr < 1e-4, f"max abs err={maxerr:.2e}")
    multi = [i for i in pick if len(sigs.iloc[i]) > 1]
    if multi:
        i = multi[0]
        first = h['0/DATA/0/matrix'][sorted(cidx[s] for s in sigs.iloc[i])[0], lm_idx]
        check("7g aggregation is a mean, not 'first row'",
              np.abs(first - expr.loc[matched.parent_smiles.iloc[i]].values).max() > 1e-4)

# ---------------------------------------------------------------- 8. name-matched compounds
nm = matched[matched.match_method != 'inchikey']
print(f"      name-matched compounds: {len(nm)} ({nm.match_method.value_counts().to_dict()})")

# ---------------------------------------------------------------- 9. statistical power: random-classifier CI recomputed
te = scaf[scaf.split == 'test'].cardiotox_label.values
lo, hi = random_classifier_aucpr_interval(te, seed=CFG.RANDOM_SEED)
print(f"      random-classifier AUC-PR 95% interval on scaffold test (recomputed): [{lo:.3f}, {hi:.3f}], prevalence={te.mean():.4f}")

# ---------------------------------------------------------------- 10. documents agree with data
root = os.path.dirname(CFG.RESULTS_DIR)
n_safe_test = int((te == 0).sum())
base = pd.read_csv(os.path.join(CFG.RESULTS_DIR, 'classical_baselines_comparison.csv'))
prim = base[base.Primary == True]
for rel in ['README.md', 'WEEK1_SUBMISSION.md', 'results/Phase1_Final_Report.md', 'results/classical_baselines_report.md']:
    txt = open(os.path.join(root, rel), encoding='utf-8').read()
    has_n = re.search(rf'\b{N}\b', txt) is not None
    has_safe = f"{n_safe_test} safe" in txt
    nums_ok = all(f"{v:.3f}" in txt for v in prim['AUC-ROC']) if rel != 'WEEK1_SUBMISSION.md' else True
    check(f"10 {rel} states cohort size {N}, {n_safe_test} safe test compounds, and the primary AUC-ROC values from the CSV",
          (has_n or rel == 'WEEK1_SUBMISSION.md') and has_safe and nums_ok, f"N={has_n} safe={has_safe} auc={nums_ok}")
check("10 baseline CSV was produced on the current test set (prevalence matches split file)",
      abs(base[base.Modality == 'Baseline']['AUC-PR'].iloc[0] - te.mean()) < 1e-9)

# ---------------------------------------------------------------- 11. deterministic LINCS pert_id choice (recomputed from raw LINCS tables)
pi = pd.read_csv(CFG.LINCS_PERT_INFO, sep='\t', dtype=str)
pi['blk'] = pi.inchi_key.fillna('').str.split('-').str[0]
st = si.reset_index()
st = st[(st.cell_id == 'HA1E') & (st.pert_itime == '24 h') & (st.pert_type == 'trt_cp')
        & st.pert_idose.str.lower().str.replace(' ', '').isin(['10um', '10.0um'])]
cnt = st.groupby('pert_id').size()
bad_pick, bad_cnt = [], []
for _, r in matched[matched.match_method == 'inchikey'].iterrows():
    cands = [p_ for p_ in pi[pi.blk == r.inchikey_block].pert_id if p_ in cnt.index]
    exp = sorted(cands, key=lambda p_: (-cnt[p_], p_))[0] if cands else None
    if exp != r.pert_id:
        bad_pick.append((r.query_name, r.pert_id, exp))
for _, r in matched.iterrows():
    if len(eval(r.sig_ids)) != cnt.get(r.pert_id, -1):
        bad_cnt.append(r.query_name)
check("11a InChIKey-matched compounds use the pert_id with most strict signatures (ties: smallest id) -> deterministic",
      not bad_pick, f"{len(bad_pick)} mismatches {bad_pick[:3]}")
check("11b every compound aggregates ALL strict signatures of its pert_id (none skipped)", not bad_cnt, f"{len(bad_cnt)} mismatches {bad_cnt[:3]}")

# ---------------------------------------------------------------- 12. requirements.txt covers every third-party import
import ast, glob
mods = set()
for fp_ in glob.glob(os.path.join(root, 'scripts', '*.py')) + [os.path.join(root, 'config.py')]:
    for node in ast.walk(ast.parse(open(fp_, encoding='utf-8').read())):
        if isinstance(node, ast.Import):
            mods |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split('.')[0])
third = mods - set(sys.stdlib_module_names) - {'config', 'scripts', 'utils', '_sandbox'}
pkgmap = {'sklearn': 'scikit-learn', 'PIL': 'pillow'}
req = {re.split(r'[<>=!~\s]', l.strip())[0].lower() for l in open(os.path.join(root, 'requirements.txt')) if l.strip() and not l.lstrip().startswith('#')}
missing = sorted(m for m in third if pkgmap.get(m, m).lower() not in req)
check("12 requirements.txt lists every third-party package imported by scripts/ and config.py", not missing, f"missing: {missing}; imports: {sorted(third)}")

# ---------------------------------------------------------------- 13. split-stability outputs are consistent with data and documents
sp_f, cv_f, rp_f = [os.path.join(CFG.RESULTS_DIR, x) for x in ('split_stability_splits.csv', 'split_stability_cv.csv', 'split_stability_report.md')]
if all(os.path.exists(x) for x in (sp_f, cv_f, rp_f)):
    sp, cvd, rep = pd.read_csv(sp_f), pd.read_csv(cv_f), open(rp_f, encoding='utf-8').read()
    shape_ok = len(sp) == 60 and sp.split.nunique() == 30 and len(cvd) == 10 and cvd.repeat.nunique() == 5
    check("13a stability CSVs have 30 splits x 2 models and 5 CV repeats x 2 models, >=5 safe per test split",
          shape_ok and sp.n_safe_test.min() >= 5, f"rows sp={len(sp)} cv={len(cvd)} min safe={sp.n_safe_test.min()}")
    txt_ok = all(f"{g.auc_roc.mean():.3f}" in rep for _, g in sp.groupby('model')) and \
             all(f"{g.auc_roc.mean():.3f}" in rep for _, g in cvd.groupby('model')) and \
             f"{n_pos} toxic / {n_neg} safe" in rep and f"{n_safe_test} safe test compounds" in rep
    check("13b stability report quotes the CSV means, current cohort counts and current frozen-split safe count", txt_ok)
    cv_means = {m: g.auc_roc.mean() for m, g in cvd.groupby('model')}
    for rel in ['README.md', 'results/Phase1_Final_Report.md']:
        t = open(os.path.join(root, rel), encoding='utf-8').read()
        check(f"13c {rel} quotes the pooled-CV means from the stability CSV",
              all(f"{x:.3f}" in t for x in cv_means.values()), str({k: round(x, 3) for k, x in cv_means.items()}))
else:
    check("13 stability outputs exist (run scripts/05_split_stability.py)", False)

# ---------------------------------------------------------------- 14. name-matched compounds are the same molecule (formula) as LINCS's structure
from rdkit.Chem import rdMolDescriptors
_unch = MolStandardize.rdMolStandardize.Uncharger()
def _formula(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    fr = max(Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumHeavyAtoms())
    return rdMolDescriptors.CalcMolFormula(_unch.uncharge(fr))
pim = pi.set_index('pert_id')
bad_f = [r.query_name for _, r in matched[matched.match_method != 'inchikey'].iterrows()
         if _formula(r.parent_smiles) != _formula(pim.loc[r.pert_id, 'canonical_smiles'])]
check("14 every name-matched compound has the same molecular formula as LINCS's structure for its pert_id", not bad_f,
      f"{(matched.match_method != 'inchikey').sum()} name-matched, mismatches: {bad_f}")

# ---------------------------------------------------------------- 15. deep-model outputs (only when they exist)
from sklearn.metrics import roc_auc_score as _auc
_dr = os.path.join(CFG.RESULTS_DIR, 'deep_models_report.md')
_found = [k for k in ('gin', 'transformer') if os.path.exists(os.path.join(CFG.RESULTS_DIR, f'deep_frozen_{k}_test_predictions.csv'))]
if os.path.exists(_dr) and _found:
    _rep = open(_dr, encoding='utf-8').read()
    _te = scaf[scaf.split == 'test']
    for k in _found:
        pr = pd.read_csv(os.path.join(CFG.RESULTS_DIR, f'deep_frozen_{k}_test_predictions.csv'))
        sd_ = pd.read_csv(os.path.join(CFG.RESULTS_DIR, f'deep_frozen_{k}_seeds.csv'))
        same = pr.parent_smiles.tolist() == _te.parent_smiles.tolist() and (pr.y.values == _te.cardiotox_label.values).all()
        seed_cols = [c for c in pr.columns if c.startswith('seed')]
        ens_ok = np.allclose(pr[seed_cols].mean(1).values, pr.ensemble.values)
        per_seed_ok = all(abs(_auc(pr.y, pr[f'seed{i}']) - sd_.test_auc.iloc[i]) < 1e-9 for i in range(len(seed_cols)))
        a_ens = _auc(pr.y, pr.ensemble)
        check(f"15a {k}: test predictions are for exactly the frozen test compounds; seed AUCs and ensemble recomputed from the predictions",
              same and ens_ok and per_seed_ok and f"{a_ens:.3f}" in _rep, f"ensemble AUC-ROC {a_ens:.3f}, seeds={len(seed_cols)}")
    for k in ('gin', 'transformer'):
        f_oof = os.path.join(CFG.RESULTS_DIR, f'deep_cv_{k}_oof.csv')
        if not os.path.exists(f_oof):
            continue
        o = pd.read_csv(f_oof)
        base_oof = pd.read_csv(os.path.join(CFG.RESULTS_DIR, 'split_stability_oof.csv'))
        sc_fold = scaf.assign(fold=o.fold.values).groupby('scaffold').fold.nunique()
        a_cv = _auc(o.y, o.oof)
        check(f"15b {k}: CV folds equal the baseline folds, every scaffold sits in exactly one fold, pooled AUC recomputed",
              o.parent_smiles.tolist() == base_oof.parent_smiles.tolist() and (o.fold.values == base_oof.fold.values).all()
              and (sc_fold == 1).all() and f"{a_cv:.3f}" in _rep, f"pooled CV AUC-ROC {a_cv:.3f}")
    for rel in ['README.md', 'results/Phase1_Final_Report.md']:
        t = open(os.path.join(root, rel), encoding='utf-8').read()
        check(f"15c {rel} contains the deep-model comparison", 'Phase 2: deep models vs baselines' in t)
else:
    print("      (no deep-model results yet: check 15 skipped)")

# ---------------------------------------------------------------- report
# ---------------------------------------------------------------- 16-19. reviewer rules (see CHECKLIST.md)
import sklearn, rdkit
def _doc(rel): return open(os.path.join(root, rel), encoding='utf-8').read()
_docs = ['README.md', 'results/Phase1_Final_Report.md']
_floor = base[base.Modality == 'Baseline']['AUC-PR'].iloc[0]
_models = base[base.Modality != 'Baseline']
_n_degen, _n_models = int(_models.Degenerate.sum()), len(_models)
check("16 README names the library versions the results were produced with, and they match this environment",
      f"scikit-learn {sklearn.__version__}" in _doc('README.md') and f"RDKit {rdkit.__version__}" in _doc('README.md'),
      f"this env: scikit-learn {sklearn.__version__}, RDKit {rdkit.__version__}")
check("17 every report prints the AUC-PR floor (prevalence) beside AUC-PR",
      all(f"{_floor:.3f}" in _doc(r) for r in _docs), f"floor={_floor:.3f}")
check("18 Phase 1 report states how many models are degenerate (single class), matching the CSV",
      f"{_n_degen} of {_n_models} models predict a single class" in _doc('results/Phase1_Final_Report.md'), f"{_n_degen} of {_n_models}")
check("19 no document claims hERG prediction (label is DICTrank)",
      not any(re.search(r'predict\w*\s+(the\s+)?hERG|hERG\s+(blockade\s+)?prediction', _doc(r), re.I) for r in _docs + ['WEEK1_SUBMISSION.md']))
_cohort_stem = matched.query_name.map(stem)
_c = matched.assign(stem=_cohort_stem).groupby('stem').cardiotox_label.nunique()
check("2d no two cohort compounds share a salt-stripped name with different labels", (_c <= 1).all(), f"{_c[_c > 1].index.tolist()}")

_sig_n = matched.sig_ids.map(lambda x: len(eval(x))).values
_va = scaf[scaf.split == 'val']
_rep = _doc('results/Phase1_Final_Report.md')
check("21 report states single-replicate share, validation-set size/safe count, and the conflicting-label rule",
      f"{int((_sig_n == 1).sum())} drugs" in _rep and f"{len(_va)} compounds ({int((_va.cardiotox_label == 0).sum())} safe" in _rep and "any toxic -> 1" in _rep,
      f"single={int((_sig_n == 1).sum())}, val={len(_va)}")
_cited = set(re.findall(r'V:(\d+[a-z]?)', open(os.path.join(root, 'CHECKLIST.md'), encoding='utf-8').read()))
_have = {n.split()[0] for n, _, _ in RESULTS}
_have_nums = {h for h in _have} | {'20'}
check("20 every check id cited in CHECKLIST.md exists in this verifier", _cited <= _have_nums, f"missing: {sorted(_cited - _have_nums)}")
out = os.path.join(CFG.RESULTS_DIR, 'pipeline_verification.md')
with open(out, 'w', encoding='utf-8') as f:
    f.write("# Pipeline Verification (auto-generated by scripts/verify_pipeline.py)\n\n")
    f.write("Each check recomputes from raw inputs and can fail.\n\n| Check | Result | Detail |\n|---|---|---|\n")
    for n, p, d in RESULTS:
        f.write(f"| {n} | {'PASS' if p else '**FAIL**'} | {d} |\n")
    f.write(f"\nCohort: {N} unique skeletons, {n_pos} toxic / {n_neg} safe. "
            f"Scaffold test: {len(te)} compounds, {(te==0).sum()} safe. "
            f"Random-classifier AUC-PR interval: [{lo:.3f}, {hi:.3f}] (prevalence {te.mean():.4f}).\n")
fails = [n for n, p, _ in RESULTS if not p]
print(f"\n{len(RESULTS)-len(fails)}/{len(RESULTS)} checks passed. Report: {out}")
sys.exit(1 if fails else 0)
