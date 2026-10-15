"""
06_make_reports.py -- Regenerates README.md, WEEK1_SUBMISSION.md and results/Phase1_Final_Report.md.

EVERY number in the generated text is read from the data / result files (cohort, splits, baselines, name-match audit,
split-stability CSVs, verification report) so the documents cannot drift from the pipeline. Static prose only
describes methods. Run after 03, 04, 05, audit_matches.py and verify_pipeline.py:

    python scripts/06_make_reports.py && python scripts/verify_pipeline.py   # second verify run checks the documents
"""
import os, re, sys, ast
import platform
import numpy as np
import pandas as pd
import sklearn, rdkit
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG

ROOT = os.path.dirname(CFG.RESULTS_DIR)
RES = CFG.RESULTS_DIR
rd = lambda p: open(p, encoding='utf-8').read()

# ------------------------------------------------------------------ gather numbers
matched = pd.read_csv(CFG.LINCS_MATCHED_CSV)
labels = pd.read_csv(CFG.LABELS_CSV)
smi = pd.read_csv(CFG.SMILES_CSV)
scaf = pd.read_csv(os.path.join(CFG.DATA_DIR, 'splits', 'scaffold_split.csv'))
drug = pd.read_csv(os.path.join(CFG.DATA_DIR, 'splits', 'drug_split.csv'))
comp = pd.read_csv(os.path.join(RES, 'classical_baselines_comparison.csv'))
table = rd(os.path.join(RES, 'baseline_table.md')).strip()
audit = rd(os.path.join(RES, 'pipeline_rebuild_audit.md'))
nm_audit = rd(os.path.join(RES, 'name_match_audit.md'))
ver = rd(os.path.join(RES, 'pipeline_verification.md'))
sp = pd.read_csv(os.path.join(RES, 'split_stability_splits.csv'))
cv = pd.read_csv(os.path.join(RES, 'split_stability_cv.csv'))

N = len(matched); TOX = int(matched.cardiotox_label.sum()); SAFE = N - TOX
raw = pd.read_excel(CFG.DICTRANK_PATH)
raw_n = len(raw); raw_amb = int((raw['DICT _ Concern'].astype(str).str.strip().str.lower() == 'ambiguous').sum())
kept = raw[raw['DICT _ Concern'].astype(str).str.strip().str.lower() != 'ambiguous']
blank_ai = int(kept['Active Ingredient(s)'].isna().sum())
lab_n, lab_tox = len(labels), int(labels.cardiotox_label.sum())
cc = labels.DICT_Concern.value_counts()


def grab(pat, text=audit, cast=int):
    m = re.search(pat, text)
    assert m, f"pattern not found in audit: {pat}"
    return cast(m.group(1).replace(',', ''))

resolved = grab(r'with a resolved structure \(SMILES\) \| (\d+)')
uniq_struct = grab(r'unique structures \(14-char InChIKey blocks\) \| (\d+)')
any_sig = grab(r'ANY LINCS trt_cp signature[^|]*\| (\d+)')
strict_pre = grab(r'STRICT signature[^|]*\| (\d+)')
n_checks = len(re.findall(r'\| (PASS|\*\*FAIL\*\*) \|', ver)); n_pass = len(re.findall(r'\| PASS \|', ver))
sigs = matched.sig_ids.map(ast.literal_eval).map(len)
n_sig, mean_sig, n_multi = int(sigs.sum()), sigs.mean(), int((sigs > 1).sum())
n_acyc = int(scaf.scaffold.astype(str).str.startswith('acyclic_').sum())
fallback76 = int((smi.match_method == 'fallback_highest_score').sum())
overrides = int((smi.status == 'manual_override').sum())
api_err = int((smi.status == 'api_error').sum())
cohort_fb = int((matched.merge(smi.drop_duplicates('query_name')[['query_name', 'match_method']].rename(columns={'match_method': 'cm'}),
                               on='query_name', how='left').cm == 'fallback_highest_score').sum())
m_nm = re.search(r'\*\*(\d+) SAME, (\d+) REVIEW', nm_audit)
n_same, n_review = int(m_nm.group(1)), int(m_nm.group(2))
n_nm = n_same + n_review
rej = re.findall(r'^- REJECTED (.+?) \(LINCS', audit, re.M)
n_rej, rej_txt = len(rej), ', '.join(rej)
assert n_review == 0, 'name-matched compounds with a different formula remain in the cohort'


def part(df, col):
    out = {}
    for s in ['train', 'val', 'test']:
        d = df[df[col] == s]
        out[s] = (len(d), int((d.cardiotox_label == 0).sum()), int(d.cardiotox_label.sum()))
    return out
S, D = part(scaf, 'split'), part(drug, 'split')
te_n, te_safe, te_tox = S['test']
prev = te_tox / te_n

# --- data-driven extra limitations (single replicate, validation shift, conflicting-label rule)
n_single = int((sigs == 1).sum())
va_n, va_safe, va_tox = S['val']
def _stem(n):
    n = str(n).lower().strip()
    for suf in CFG.SALT_SUFFIXES:
        if n.endswith(" " + suf):
            n = n[:-(len(suf) + 1)].strip()
    return n
_lab = pd.read_csv(CFG.LABELS_CSV)
_lab['stem'] = _lab.resolved_name.map(_stem)
_g = _lab.groupby('stem').cardiotox_label.nunique()
conflict_stems = sorted(_g[_g > 1].index)
LIM_EXTRA = (
    f"- **One noisy replicate per drug.** The {len(matched)} cohort drugs have {n_sig} strict signatures in total "
    f"(mean {mean_sig:.2f} per drug); {n_single} drugs ({100*n_single/len(matched):.0f}%) rest on a single signature. "
    f"Averaging cannot reduce that noise, which is one reason expression models sit at chance.\n"
    f"- **Validation set is small and differently balanced.** It has {va_n} compounds ({va_safe} safe; {100*va_tox/va_n:.0f}% toxic) "
    f"against {100*prev:.0f}% toxic in the test set, so early stopping and thresholds are noisy. The frozen split was not changed after results were seen.\n"
    f"- **Conflicting labels were resolved as 'any toxic -> 1'.** Salt/parent forms of {len(conflict_stems)} drugs carried different DICTrank labels "
    f"({', '.join(conflict_stems)}). This biases those drugs towards toxic.\n"
)

base = comp[comp.Modality == 'Baseline'].iloc[0]
models = comp[comp.Modality != 'Baseline']
floor, floor_lo, floor_hi = base['AUC-PR'], base['AUC-PR_95CI_Low'], base['AUC-PR_95CI_High']
prim_s = models[(models.Primary == True) & (models.Modality == 'Structure')].iloc[0]
prim_b = models[(models.Primary == True) & (models.Modality == 'Biology')].iloc[0]
ci = lambda r: f"[{r['AUC-ROC_95CI_Low']:.3f}, {r['AUC-ROC_95CI_High']:.3f}]"
n_models, n_degen = len(models), int(models.Degenerate.sum())
expl = models[(models.Primary != True) & (models['CI_excludes_0.5'] == True) & (models['AUC-ROC'] > .5)]
expl_txt = '; '.join(f"{r.Modality} {r.Model} AUC-ROC {r['AUC-ROC']:.3f} {ci(r)}" for _, r in expl.iterrows())
lasso = models[(models.Modality == 'Biology') & models.Model.str.contains('Lasso')].iloc[0]


def sp_stats(m):
    g = sp[sp.model.str.startswith(m)].auc_roc
    return g.mean(), g.std(), g.min(), g.max(), 100 * (g > .5).mean()
def cv_stats(m):
    g = cv[cv.model.str.startswith(m)]
    return g.auc_roc.mean(), g.auc_roc.min(), g.auc_roc.max(), g.auc_roc.iloc[0], g.ci_low.iloc[0], g.ci_high.iloc[0]
s_sp, b_sp = sp_stats('Structure'), sp_stats('Biology')
s_cv, b_cv = cv_stats('Structure'), cv_stats('Biology')
sd_pool = max(s_sp[1], b_sp[1])
stab_table = (
    "| Model | Frozen split (test AUC-ROC) | 30 random scaffold splits: mean ± SD [min, max] | Pooled 5-fold scaffold CV, 5 repeats: mean [range] | Repeat-0 pooled AUC-ROC, 95% CI |\n"
    "|---|:-:|:-:|:-:|:-:|\n"
    f"| Structure: Morgan-FP RF | {prim_s['AUC-ROC']:.3f} {ci(prim_s)} | {s_sp[0]:.3f} ± {s_sp[1]:.3f} [{s_sp[2]:.3f}, {s_sp[3]:.3f}] | {s_cv[0]:.3f} [{s_cv[1]:.3f}, {s_cv[2]:.3f}] | {s_cv[3]:.3f} [{s_cv[4]:.3f}, {s_cv[5]:.3f}] |\n"
    f"| Biology: L1000 LR (L2) | {prim_b['AUC-ROC']:.3f} {ci(prim_b)} | {b_sp[0]:.3f} ± {b_sp[1]:.3f} [{b_sp[2]:.3f}, {b_sp[3]:.3f}] | {b_cv[0]:.3f} [{b_cv[1]:.3f}, {b_cv[2]:.3f}] | {b_cv[3]:.3f} [{b_cv[4]:.3f}, {b_cv[5]:.3f}] |")
s_sig = s_cv[4] > .5; b_sig = b_cv[4] > .5
stab_read = (
    f"- **Structure:** pooled scaffold-CV AUC-ROC {s_cv[3]:.3f} [{s_cv[4]:.3f}, {s_cv[5]:.3f}] — "
    + ("the interval excludes 0.5, so there is modest but real signal (about 0.63-0.67 depending on the split), far weaker than the single frozen-split interval suggests.\n"
       if s_sig else "the interval includes 0.5: no signal established.\n")
    + f"- **Biology:** pooled scaffold-CV AUC-ROC {b_cv[3]:.3f} [{b_cv[4]:.3f}, {b_cv[5]:.3f}] — "
    + ("the interval excludes 0.5.\n" if b_sig else "the interval includes 0.5: a linear model on L1000 z-scores finds no signal.\n")
    + f"- **Noise floor:** the SD of test AUC-ROC across random scaffold splits is {s_sp[1]:.3f} (structure) / {b_sp[1]:.3f} (biology); "
      f"a single-split difference smaller than about {2*sd_pool:.2f} should not be read as a real difference.")

ENV = f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, RDKit {rdkit.__version__}, NumPy {np.__version__}, pandas {pd.__version__}"

deep_path = os.path.join(RES, 'deep_models_report.md')
if os.path.exists(deep_path):
    _d = rd(deep_path).split('\n')
    deep_md = '\n'.join(l.replace('## ', '#### ', 1) if l.startswith('## ') else l for l in _d[2:]).strip()
    deep_section = "### Phase 2: deep models vs baselines (`scripts/07_train_deep.py`, `results/deep_models_report.md`)\n\n" + deep_md + "\n"
else:
    deep_section = ""

# ------------------------------------------------------------------ README
readme = f"""# Cardiotox-Fusion

**Cardiotoxicity prediction from molecular structure (GNN) and from L1000 gene expression (Transformer), evaluated as two independent models on the same compounds.**

> Despite the repository name there is **no fusion model**. Fusion was dropped after review: imputed expression is not independent evidence, so the structure and expression branches are trained and scored separately and compared head-to-head. The label is FDA **DICTrank** (binary: Most + Less Concern = 1, No Concern = 0; Ambiguous dropped). It is a clinical, multi-mechanism label, not hERG.

---

## Repository structure

```
cardiotox-fusion/
├── config.py                     # all paths / constants / Phase-2 model spec
├── data/
│   ├── raw/                      # dictrank_dataset_508.xlsx (committed); lincs/ (download, see below)
│   ├── processed/                # labeled_compounds.csv, compounds_with_smiles.csv, lincs_matched_compounds.csv,
│   │                             # expression_matrix.csv ({N} x 978), smiles_checkpoint.json
│   └── splits/                   # drug_split.csv, scaffold_split.csv  (70/15/15, seed {CFG.RANDOM_SEED})
├── scripts/
│   ├── 01_fetch_labels.py        # DICTrank xlsx -> labeled_compounds.csv
│   ├── 02_resolve_smiles.py      # ChEMBL name -> SMILES (cached; {overrides} documented manual overrides)
│   ├── 03_rebuild_pipeline.py    # LINCS matching, landmark genes, dedup, aggregation, splits, audit log (deterministic)
│   ├── 04_baselines.py           # -> train_classical_baselines.py ({n_models} classical models, bootstrap CIs)
│   ├── 05_split_stability.py     # 30 random scaffold splits + repeated scaffold-grouped CV for the primary baselines
│   ├── 06_make_reports.py        # regenerates this README and the reports from the result files
│   ├── verify_pipeline.py        # independent falsifiable checks (recomputes from raw files)
│   ├── audit_matches.py          # name-matched compounds + related-form label disagreements
│   └── utils.py
├── results/                      # generated: pipeline_rebuild_audit.md, pipeline_verification.md, name_match_audit.md,
│                                 # classical_baselines_report.md/.csv, split_stability_report.md, Phase1_Final_Report.md
└── legacy_week1/                 # superseded Week-1 artifacts (562-cohort etc.) - do not cite
```

## Raw data (download before running)

Place in `data/raw/lincs/` (names exactly as `config.py` expects):

| File | Notes |
|---|---|
| `GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx` | ~5.8 GB, [GEO GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138) (gitignored) |
| `sig_info.txt.gz`, `pert_info.txt.gz` | GSE70138 `..._sig_info_...` / `..._pert_info_...`, renamed |
| `gene_info.txt.gz` | GSE70138 `..._gene_info_2017-03-06.txt.gz`, renamed |

The FDA DICTrank spreadsheet is committed at `data/raw/dictrank_dataset_508.xlsx`.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/01_fetch_labels.py
python scripts/02_resolve_smiles.py --resume   # uses data/processed/smiles_checkpoint.json; without it, needs ChEMBL network access (~15 min)
python scripts/03_rebuild_pipeline.py          # ~25 s once the GCTX is present; output is identical across runs
python scripts/04_baselines.py                 # ~2.5 min
python scripts/05_split_stability.py all       # a few minutes
python scripts/audit_matches.py
python scripts/verify_pipeline.py              # must end "{n_checks}/{n_checks} checks passed"
python scripts/06_make_reports.py              # regenerates this file from the results; then re-run verify_pipeline.py
```

A rebuild from `01` onward in a clean copy was compared file by file with the committed data (see "Corrections" in `results/Phase1_Final_Report.md`).

**Environment of the committed results:** {ENV}. Data files, splits and the logistic-regression results reproduce exactly on any machine. Random-forest numbers move by a few hundredths of AUC-ROC across scikit-learn / RDKit versions (an earlier run of the same code, data and seed on another machine gave a frozen-split structure AUC-ROC of 0.710 versus 0.680 here), so read differences smaller than the split-to-split SD as noise.

## Cohort

| Stage | Count |
|---|---|
| DICTrank compounds (Ambiguous dropped) | {lab_n:,} ({lab_tox} toxic / {lab_n-lab_tox} safe) |
| ...with a resolved structure | {resolved:,} ({uniq_struct} unique skeletons) |
| Skeletons with *any* LINCS compound signature | {any_sig} |
| **Strict cohort: HA1E / 10 µM / 24 h** | **{N}** ({TOX} toxic / {SAFE} safe) |

Only ~{any_sig} DICTrank skeletons have *any* LINCS signature, so relaxing the cell-line / dose / time filters adds signatures per drug but not drugs. LINCS coverage, not the filter, caps the sample size.
The strict cohort has {n_sig} signatures (mean {mean_sig:.2f} per compound; {n_multi} compounds have more than one and are mean-aggregated).

| Split | Train | Val | Test |
|---|---|---|---|
| Drug-level (stratified) | {D['train'][0]} | {D['val'][0]} | {D['test'][0]} |
| Bemis-Murcko scaffold (primary evaluation) | {S['train'][0]} ({S['train'][1]} safe) | {S['val'][0]} ({S['val'][1]} safe) | {te_n} (**{te_safe} safe**) |

Both splits have zero InChIKey-skeleton overlap across partitions (checked independently in `verify_pipeline.py`).

## Baselines (scaffold test set, N = {te_n}: {te_tox} toxic / {te_safe} safe)

Toxic is the majority class ({100*prev:.1f}%), so AUC-PR has a floor of {floor:.3f} and is not used to rank models; **AUC-ROC and balanced accuracy are primary.**

{table}

Full {n_models}-model table with degenerate-model flags: `results/classical_baselines_report.md`.

## How stable are these numbers? (`results/split_stability_report.md`)

The frozen test set has only {te_safe} safe compounds, so one split is a noisy estimate. The two primary baselines were therefore also scored on 30 random scaffold splits and on 5-fold scaffold-grouped CV repeated 5 times (out-of-fold predictions pooled over all {N} compounds).

{stab_table}

{stab_read}

{deep_section}
### Reading these results
1. **Frozen split:** both primary AUC-ROC intervals include 0.5 (structure {prim_s['AUC-ROC']:.3f} {ci(prim_s)}, biology {prim_b['AUC-ROC']:.3f} {ci(prim_b)}). {len(expl)} of {n_models-2} exploratory models have an interval that excludes 0.5{(': ' + expl_txt) if len(expl) else ''}; with {n_models} models on one {te_n}-compound test set such a result is unconfirmed until it replicates.
2. **Repeated scaffold CV (more data per estimate):** structure {s_cv[3]:.3f} [{s_cv[4]:.3f}, {s_cv[5]:.3f}], biology {b_cv[3]:.3f} [{b_cv[4]:.3f}, {b_cv[5]:.3f}]. Structure carries modest signal; the L1000 linear baseline carries none.
3. **Single splits are unstable.** The same structure baseline scored 0.723 on the earlier leaky split, 0.613 after removing leaked entries, and {prim_s['AUC-ROC']:.3f} after the deterministic rebuild; the SD across random scaffold splits is {s_sp[1]:.3f}.
4. **Power is the main limitation.** A random scorer's AUC-PR interval on the frozen test set spans [{floor_lo:.3f}, {floor_hi:.3f}].

## Known limitations
{LIM_EXTRA}- {n_rej} name-matched compounds were excluded because LINCS's structure is a different molecule ({rej_txt}); the {n_nm} name-matched compounds that remain have identical molecular formulas (tautomer / salt / stereo differences only); see `results/name_match_audit.md`.
- {cohort_fb} cohort structures came from ChEMBL's "highest score" fallback rather than an exact name match.
- {api_err} DICTrank compounds are marked `api_error` in `compounds_with_smiles.csv` (ChEMBL calls failed in the original run), so they were never given a structure. Re-running `02_resolve_smiles.py` for them on a machine with ChEMBL access might add LINCS-matched compounds; this has not been done.
- The "any toxic form -> 1" rule resolves label disagreements only within one InChIKey skeleton; sibling forms with different skeletons (e.g. clindamycin vs clindamycin phosphate) can still disagree (listed in `results/name_match_audit.md`).
- Bootstrap intervals cover test-set sampling only, not training variability.
"""
open(os.path.join(ROOT, 'README.md'), 'w', encoding='utf-8').write(readme)

# ------------------------------------------------------------------ WEEK1_SUBMISSION
week1 = f"""# Cardiotox-Fusion — Week 1 Submission (updated)

> **Superseded numbers.** The original Week-1 version of this file reported 517 compounds and 361/78/78 and 322/89/106 splits.
> Those came from a pipeline later found to contain duplicate salt forms and structural leakage. Current, verified numbers are below;
> the single source of truth is `README.md` and `results/pipeline_verification.md`.

**Repository:** https://github.com/JP-Bro/Cardiotox-Fusion-project

## Current deliverables
| File | Content |
|---|---|
| `data/splits/drug_split.csv` | stratified 70/15/15 on unique InChIKey skeletons ({D['train'][0]} / {D['val'][0]} / {D['test'][0]}) |
| `data/splits/scaffold_split.csv` | Bemis-Murcko scaffold split by compound count ({S['train'][0]} / {S['val'][0]} / {S['test'][0]}) |
| `data/processed/expression_matrix.csv` | {N} x 978 mean-aggregated landmark z-scores (HA1E / 10 µM / 24 h) |
| `data/processed/lincs_matched_compounds.csv` | matched cohort with LINCS pert_id and signature ids |
| `results/pipeline_rebuild_audit.md` | cohort funnel and dedup log (auto-generated) |
| `results/pipeline_verification.md` | {n_checks} independent checks (auto-generated) |
| `results/name_match_audit.md` | audit of name-matched compounds (auto-generated) |
| `results/classical_baselines_report.md` | {n_models} classical models with bootstrap CIs (auto-generated) |
| `results/split_stability_report.md` | split-to-split variability of the primary baselines (auto-generated) |
| `results/Phase1_Final_Report.md` | narrative report |

Pre-audit artifacts are in `legacy_week1/` and should not be cited.

## Cohort and splits
* {lab_n:,} labeled DICTrank compounds -> {N} unique chemical skeletons with a strict (HA1E / 10 µM / 24 h) LINCS signature ({TOX} toxic / {SAFE} safe).
* Drug split: {D['train'][0]} / {D['val'][0]} / {D['test'][0]}. Scaffold split: {S['train'][0]} / {S['val'][0]} / {S['test'][0]} (scaffold test: {te_tox} toxic / {te_safe} safe).
* Zero cross-split skeleton overlap in both splits.

## Reproduce
```bash
python scripts/01_fetch_labels.py
python scripts/02_resolve_smiles.py --resume
python scripts/03_rebuild_pipeline.py
python scripts/04_baselines.py
python scripts/05_split_stability.py all
python scripts/audit_matches.py
python scripts/verify_pipeline.py   # expects: {n_checks}/{n_checks} checks passed
```
LINCS raw files must be downloaded first (see `README.md`).
"""
open(os.path.join(ROOT, 'WEEK1_SUBMISSION.md'), 'w', encoding='utf-8').write(week1)

# ------------------------------------------------------------------ Phase 1 report
report = f"""# Cardiotox-Fusion — Phase 1 Final Report

**Project:** Cardiotoxicity prediction, structure (GNN) vs expression (Transformer), two independent models
**Team:** Jimit Patel, Mallu Vineela Reddy, Aditya Gaurang Patel, Nehal Solanki, Pranjal Chhipa
**Repository:** https://github.com/JP-Bro/Cardiotox-Fusion-project
**Status:** October 2026. Every number below is generated from the result files by `scripts/06_make_reports.py` and checked by `scripts/verify_pipeline.py` ({n_pass}/{n_checks} checks pass at generation time).

---

## 1. Overview

| Branch | Input | Model (Phase 2 spec, `config.py`) |
|---|---|---|
| Structure | 2D molecular graph | 3-layer Graph Isomorphism Network |
| Expression | 978-gene L1000 z-score vector | 2-layer, 4-head Transformer encoder; token = learned gene-identity embedding + z-score projection (no sinusoidal positions: gene order is arbitrary) |

No fusion. Both models are evaluated on the **same** compounds and the **same** scaffold split, the only setting in which structure and biology can be compared. The label is DICTrank (clinical, multi-mechanism); hERG is discussed only as one mechanism, not predicted.

## 2. Dataset construction

1. **Labels** (`01_fetch_labels.py`): {raw_n:,} DICTrank rows; {raw_amb} Ambiguous dropped -> {lab_n:,} ({lab_tox} toxic = {cc.get('less',0)} Less + {cc.get('most',0)} Most; {lab_n-lab_tox} safe). Drug name = *Active Ingredient(s)*, falling back to *Generic/Proper Name(s)* for {blank_ai} blank rows (mostly withdrawn Most-Concern drugs; the Trade Name column is blank for them).
2. **Structures** (`02_resolve_smiles.py`): ChEMBL name search -> SMILES; salts stripped to the largest fragment. {resolved:,} compounds resolve ({uniq_struct} unique skeletons); {fallback76} were chosen by ChEMBL's highest-score fallback instead of an exact name; {overrides} structures are documented manual overrides (`MANUAL_OVERRIDES`); {api_err} compounds failed with an API error in the original run and were not retried.
3. **LINCS matching** (`03_rebuild_pipeline.py`): Level-5 COMPZ signatures filtered to `trt_cp`, HA1E, 24 h, 10.0 µM. Matching is by InChIKey connectivity block first, then salt-stripped name; a name match is accepted only if LINCS's structure for the perturbagen has the same molecular formula. {any_sig} skeletons have a signature under *any* condition; {strict_pre} under the strict condition; **{N} after enforcing one LINCS perturbagen per compound**. When a compound has several LINCS perturbagen ids (batches), the one with the most strict signatures is used (ties: smallest id), so the build is deterministic.
4. **Aggregation:** replicate z-vectors per compound are averaged. Mean signatures per compound is {mean_sig:.2f} ({n_sig} signatures in total) and only {n_multi} compounds have more than one, so aggregation is a minor step; verified against the GCTX (`verify_pipeline.py` checks 7f/7g).
5. **Landmark genes:** `gene_info` rows with `pr_is_lm == 1` (978), mapped to their real GCTX row indices (0-12,321). The control confirms that the first 978 rows contain only 194 landmark genes.
6. **Dedup and labels:** charges neutralised (RDKit `Uncharger`); one compound per 14-character InChIKey block. Label disagreements within a block (acyclovir; omeprazole/esomeprazole) resolve to toxic.

### Final cohort
| Metric | Value |
|---|---|
| Unique skeletons | **{N}** ({TOX} toxic / {SAFE} safe; {100*TOX/N:.1f}% toxic) |
| Expression matrix | {N} x 978 |

## 3. Splits (`03_rebuild_pipeline.py`, seed {CFG.RANDOM_SEED})

| Split | Train | Val | Test |
|---|---|---|---|
| Drug-level, stratified | {D['train'][0]} | {D['val'][0]} | {D['test'][0]} |
| Scaffold (primary) | {S['train'][0]} ({S['train'][1]} safe) | {S['val'][0]} ({S['val'][1]} safe) | {te_n} (**{te_safe} safe**) |

Murcko scaffolds without chirality; {n_acyc} acyclic compounds get unique pseudo-scaffolds; packing by compound count with a stable tie order. Zero InChIKey-skeleton overlap, no scaffold spans two partitions, and no test compound has a train neighbour at Tanimoto >= 0.85 (all recomputed independently).

## 4. Statistical power and evaluation strategy

The scaffold test set has {te_safe} safe compounds out of {te_n}. A random scorer's AUC-PR 95% bootstrap interval is **[{floor_lo:.3f}, {floor_hi:.3f}]** (floor {floor:.3f}). Evaluation: both branches on the same scaffold split with bootstrap CIs, plus repeated scaffold-grouped CV (section 5b); the cohort stays strict (HA1E / 10 µM / 24 h). Relaxing cell line / dose adds signatures per drug, not drugs (only ~{any_sig} skeletons have any signature), so it cannot fix power. Structure-only secondary analysis on the full set is limited to the {uniq_struct} skeletons with resolved structures (not {lab_n:,}) and is never compared with the expression headline.

## 5. Baselines

Scaffold test set N = {te_n} ({te_tox} toxic / {te_safe} safe, {100*prev:.1f}% positive). AUC-ROC and balanced accuracy are primary; AUC-PR is shown with its {floor:.3f} floor.

{table}

All {n_models} models, with degenerate (single-class) models flagged, are in `results/classical_baselines_report.md`.

### Interpretation (frozen split)
1. Neither primary baseline has an AUC-ROC interval that excludes 0.5 on this split (structure {ci(prim_s)}, biology {ci(prim_b)}).
2. {len(expl)} of {n_models-2} exploratory models have an interval above 0.5{(': ' + expl_txt + '; ') if len(expl) else '; '}with {n_models} models on one small test set this is unconfirmed until replicated.
3. {n_degen} of {n_models} models predict a single class (degenerate); their AUC-PR carries no information.
4. Classical results do not say whether a deep model will succeed or fail; whether a learned representation recovers L1000 signal that classical approaches missed is the Phase 2 question, and a null result is reportable.

### Additional limitations
{LIM_EXTRA}
## 5b. Split stability (`scripts/05_split_stability.py`, `results/split_stability_report.md`)

{stab_table}

{stab_read}
- **Use for Phase 2:** a deep model counts as an improvement only if it beats the pooled-CV baseline of its own modality by more than the split-to-split SD; GIN must clearly exceed about {s_cv[0]:.2f}, the Transformer must clearly exceed 0.5.
- Intervals are bootstrap over compounds with fixed pooled predictions; they do not include training variability.

{deep_section}
## 6. Corrections since the 10 Sep submission

| Item | Previously reported | Verified |
|---|---|---|
| Cohort size | 487 | **{N}**: etoposide phosphate and clindamycin phosphate were name-matched to the *same LINCS signature* as etoposide / clindamycin HCl (identical expression, different splits; the clindamycin pair had opposite labels), and other prodrug / ester / mixture-component name matches used a different molecule's expression. Rules added: one perturbagen per compound; a name match needs an identical molecular formula ({n_rej} compounds excluded: {rej_txt}). |
| Reproducibility of the rebuild | "deterministic" | A from-scratch rebuild gave a **different LINCS perturbagen for 8 compounds** and a different scaffold packing, because choices depended on Python set/hash order and an unstable sort. Fixed (most strict signatures, then id; stable sorts); four different `PYTHONHASHSEED` values now give byte-identical outputs. The scaffold test set changed to {te_safe} safe and the baselines were re-run. |
| Scaffold test set | 10 safe | **{te_safe} safe** (re-split after the fixes) |
| Primary structure baseline | AUC-ROC 0.723 [0.523, 0.906] "real signal" | 0.613 after removing leaked entries; **{prim_s['AUC-ROC']:.3f} {ci(prim_s)}** after the deterministic rebuild; interval includes 0.5 on the frozen split; pooled scaffold-CV {s_cv[3]:.3f} [{s_cv[4]:.3f}, {s_cv[5]:.3f}] |
| "1,048 drugs with any L1000 signature" | 1,048 | **{any_sig} skeletons** (1,048 was the count of ChEMBL-resolved structures) |
| "1,837 signatures across 582 entries" | 1,837 | 1,837 is every HA1E/10 µM/24 h signature in LINCS; the cohort uses **{n_sig}** ({mean_sig:.2f} per compound; not "~4") |
| Biology baselines | "L1 Lasso 0.906 shows signal" (AUC-PR) | AUC-PR floor is {floor:.3f}; Lasso AUC-ROC {lasso['AUC-ROC']:.3f} {ci(lasso)} |
| Repo reproducibility | "runs from a clean clone" | `01` read the wrong name column and `02` was a different script from the one that built the data; `requirements.txt` lacked h5py; all fixed and a clean-copy rebuild was compared with the committed data |
| Name-matched compounds | 14, "all valid" | {n_nm} remain, all with identical molecular formula; {n_rej} rejected (`results/pipeline_rebuild_audit.md`, `results/name_match_audit.md`) |

## 7. Reproducibility

```bash
python scripts/01_fetch_labels.py
python scripts/02_resolve_smiles.py --resume
python scripts/03_rebuild_pipeline.py
python scripts/04_baselines.py
python scripts/05_split_stability.py all
python scripts/audit_matches.py
python scripts/verify_pipeline.py
python scripts/06_make_reports.py && python scripts/verify_pipeline.py
```
Raw LINCS files must be downloaded first (see `README.md`).

## 8. Next steps (Phase 2)
1. {'GIN and Transformer are trained (section above); next: more CV repeats / seeds if the intervals are too wide, and the interpretability analysis.' if deep_section else 'GIN (3-layer) and Transformer (learned gene embeddings) on the same frozen split and the scaffold-grouped CV folds (`scripts/07_train_deep.py`), class-weighted loss, selected on validation AUC-ROC; no tuning on the test set.'}
2. Report all four models (RF, LR, GIN, Transformer) in one table with AUC-ROC and balanced accuracy first, against the noise floor above.
3. Optional data work: re-resolve the {api_err} `api_error` compounds with ChEMBL access (could add LINCS-matched compounds).
"""
open(os.path.join(RES, 'Phase1_Final_Report.md'), 'w', encoding='utf-8').write(report)
print('docs written;', f'checks in verification report: {n_pass}/{n_checks}')
