# Cardiotox-Fusion

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
│   │                             # expression_matrix.csv (486 x 978), smiles_checkpoint.json
│   └── splits/                   # drug_split.csv, scaffold_split.csv  (70/15/15, seed 42)
├── scripts/
│   ├── 01_fetch_labels.py        # DICTrank xlsx -> labeled_compounds.csv
│   ├── 02_resolve_smiles.py      # ChEMBL name -> SMILES (cached; 1 documented manual overrides)
│   ├── 03_rebuild_pipeline.py    # LINCS matching, landmark genes, dedup, aggregation, splits, audit log (deterministic)
│   ├── 04_baselines.py           # -> train_classical_baselines.py (17 classical models, bootstrap CIs)
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
python scripts/verify_pipeline.py              # must end "54/54 checks passed"
python scripts/06_make_reports.py              # regenerates this file from the results; then re-run verify_pipeline.py
```

A rebuild from `01` onward in a clean copy was compared file by file with the committed data (see "Corrections" in `results/Phase1_Final_Report.md`).

**Environment of the committed results:** Python 3.10.12, scikit-learn 1.7.2, RDKit 2026.03.6, NumPy 2.2.6, pandas 2.3.3. Data files, splits and the logistic-regression results reproduce exactly on any machine. Random-forest numbers move by a few hundredths of AUC-ROC across scikit-learn / RDKit versions (an earlier run of the same code, data and seed on another machine gave a frozen-split structure AUC-ROC of 0.710 versus 0.680 here), so read differences smaller than the split-to-split SD as noise.

## Cohort

| Stage | Count |
|---|---|
| DICTrank compounds (Ambiguous dropped) | 1,211 (868 toxic / 343 safe) |
| ...with a resolved structure | 1,102 (976 unique skeletons) |
| Skeletons with *any* LINCS compound signature | 490 |
| **Strict cohort: HA1E / 10 µM / 24 h** | **486** (385 toxic / 101 safe) |

Only ~490 DICTrank skeletons have *any* LINCS signature, so relaxing the cell-line / dose / time filters adds signatures per drug but not drugs. LINCS coverage, not the filter, caps the sample size.
The strict cohort has 528 signatures (mean 1.09 per compound; 28 compounds have more than one and are mean-aggregated).

| Split | Train | Val | Test |
|---|---|---|---|
| Drug-level (stratified) | 340 | 73 | 73 |
| Bemis-Murcko scaffold (primary evaluation) | 340 (66 safe) | 72 (23 safe) | 74 (**12 safe**) |

Both splits have zero InChIKey-skeleton overlap across partitions (checked independently in `verify_pipeline.py`).

## Baselines (scaffold test set, N = 74: 62 toxic / 12 safe)

Toxic is the majority class (83.8%), so AUC-PR has a floor of 0.838 and is not used to rank models; **AUC-ROC and balanced accuracy are primary.**

| Model | AUC-ROC | 95% CI | Balanced Acc | AUC-PR (floor 0.838) | 95% CI |
|---|:-:|:-:|:-:|:-:|:-:|
| Random classifier | 0.500 | — | 0.500 | 0.838 | [0.726, 0.944] |
| Structure: Random Forest (200 trees) | 0.680 | [0.484, 0.863] | 0.542 | 0.910 | [0.837, 0.973] |
| Biology: Logistic Regression (L2) | 0.586 | [0.385, 0.790] | 0.570 | 0.864 | [0.763, 0.966] |

Full 17-model table with degenerate-model flags: `results/classical_baselines_report.md`.

## How stable are these numbers? (`results/split_stability_report.md`)

The frozen test set has only 12 safe compounds, so one split is a noisy estimate. The two primary baselines were therefore also scored on 30 random scaffold splits and on 5-fold scaffold-grouped CV repeated 5 times (out-of-fold predictions pooled over all 486 compounds).

| Model | Frozen split (test AUC-ROC) | 30 random scaffold splits: mean ± SD [min, max] | Pooled 5-fold scaffold CV, 5 repeats: mean [range] | Repeat-0 pooled AUC-ROC, 95% CI |
|---|:-:|:-:|:-:|:-:|
| Structure: Morgan-FP RF | 0.680 [0.484, 0.863] | 0.671 ± 0.086 [0.540, 0.867] | 0.659 [0.638, 0.678] | 0.638 [0.579, 0.697] |
| Biology: L1000 LR (L2) | 0.586 [0.385, 0.790] | 0.467 ± 0.077 [0.304, 0.635] | 0.487 [0.472, 0.504] | 0.504 [0.447, 0.568] |

- **Structure:** pooled scaffold-CV AUC-ROC 0.638 [0.579, 0.697] — the interval excludes 0.5, so there is modest but real signal (about 0.63-0.67 depending on the split), far weaker than the single frozen-split interval suggests.
- **Biology:** pooled scaffold-CV AUC-ROC 0.504 [0.447, 0.568] — the interval includes 0.5: a linear model on L1000 z-scores finds no signal.
- **Noise floor:** the SD of test AUC-ROC across random scaffold splits is 0.086 (structure) / 0.077 (biology); a single-split difference smaller than about 0.17 should not be read as a real difference.

### Phase 2: deep models vs baselines (`scripts/07_train_deep.py`, `results/deep_models_report.md`)

*Auto-generated by `scripts/07_train_deep.py report`.*

Cohort 486 compounds (385 toxic / 101 safe). Frozen scaffold test set: 74 compounds, 12 safe. All models use the same compounds, the same scaffold split and the same scaffold-grouped CV folds. Intervals are percentile bootstraps over compounds with fixed predictions; they do not include training variability, which is reported separately as the SD over training seeds.

#### Frozen scaffold split (test AUC-ROC)

| Model | Seeds | Per-seed AUC-ROC (mean ± SD) | Seed-ensemble AUC-ROC | 95% CI | Balanced acc (val-chosen threshold, mean) |
|---|:-:|:-:|:-:|:-:|:-:|
| Structure RF (baseline) | – | – | 0.680 | [0.484, 0.863] | 0.542 |
| Biology LR (baseline) | – | – | 0.586 | [0.385, 0.790] | 0.570 |
| GIN (structure) | 5 | 0.703 ± 0.034 | 0.716 | [0.556, 0.866] | 0.598 |
| Transformer (expression) | 3 | 0.531 ± 0.112 | 0.496 | [0.306, 0.672] | 0.520 |

#### Scaffold-grouped 5-fold CV, repeat 0 (pooled out-of-fold AUC-ROC over all compounds)

| Model | Pooled AUC-ROC | 95% CI | Paired difference vs same-modality baseline | 95% CI of difference |
|---|:-:|:-:|:-:|:-:|
| Structure RF (baseline) | 0.638 | [0.579, 0.697] | – | – |
| GIN (structure) | 0.622 | [0.558, 0.682] | -0.016 | [-0.090, +0.053] |
| Biology LR (baseline) | 0.504 | [0.447, 0.567] | – | – |
| Transformer (expression) | 0.472 | [0.410, 0.533] | -0.033 | [-0.117, +0.051] |

#### Reading (computed)

- **GIN (structure), frozen split:** AUC-ROC difference +0.036 [-0.144, +0.214] -> not distinguishable from the baseline (paired interval includes 0).
- **Transformer (expression), frozen split:** AUC-ROC difference -0.090 [-0.431, +0.207] -> not distinguishable from the baseline (paired interval includes 0).
- **GIN (structure), pooled CV:** AUC-ROC difference -0.016 [-0.090, +0.053] -> not distinguishable from the baseline (paired interval includes 0).
- **Transformer (expression), pooled CV:** AUC-ROC difference -0.033 [-0.117, +0.051] -> not distinguishable from the baseline (paired interval includes 0).
- A model counts as an improvement only if the paired CV interval is above 0 **and** the gain exceeds the split-to-split SD in `results/split_stability_report.md`.
- Absolute signal check: a pooled-CV interval that includes 0.5 means the model has not been shown to carry any signal.

### Reading these results
1. **Frozen split:** both primary AUC-ROC intervals include 0.5 (structure 0.680 [0.484, 0.863], biology 0.586 [0.385, 0.790]). 4 of 15 exploratory models have an interval that excludes 0.5: Structure Random Forest (500 trees) AUC-ROC 0.710 [0.528, 0.871]; Structure Support Vector Classifier (RBF) AUC-ROC 0.714 [0.530, 0.882]; Structure Gradient Boosting (HistGBM) AUC-ROC 0.698 [0.520, 0.860]; Structure K-Nearest Neighbors (k=5, Jaccard) AUC-ROC 0.704 [0.545, 0.849]; with 17 models on one 74-compound test set such a result is unconfirmed until it replicates.
2. **Repeated scaffold CV (more data per estimate):** structure 0.638 [0.579, 0.697], biology 0.504 [0.447, 0.568]. Structure carries modest signal; the L1000 linear baseline carries none.
3. **Single splits are unstable.** The same structure baseline scored 0.723 on the earlier leaky split, 0.613 after removing leaked entries, and 0.680 after the deterministic rebuild; the SD across random scaffold splits is 0.086.
4. **Power is the main limitation.** A random scorer's AUC-PR interval on the frozen test set spans [0.726, 0.944].

## Known limitations
- **One noisy replicate per drug.** The 486 cohort drugs have 528 strict signatures in total (mean 1.09 per drug); 458 drugs (94%) rest on a single signature. Averaging cannot reduce that noise, which is one reason expression models sit at chance.
- **Validation set is small and differently balanced.** It has 72 compounds (23 safe; 68% toxic) against 84% toxic in the test set, so early stopping and thresholds are noisy. The frozen split was not changed after results were seen.
- **Conflicting labels were resolved as 'any toxic -> 1'.** Salt/parent forms of 6 drugs carried different DICTrank labels (acyclovir, amifampridine, clindamycin, hydroxocobalamin, vancomycin, zinc). This biases those drugs towards toxic.
- 7 name-matched compounds were excluded because LINCS's structure is a different molecule (etoposide phosphate, Clindamycin Phosphate, tedizolid phosphate, FOSTAMATINIB, ixazomib citrate, IVERMECTIN, ESTRADIOL); the 7 name-matched compounds that remain have identical molecular formulas (tautomer / salt / stereo differences only); see `results/name_match_audit.md`.
- 7 cohort structures came from ChEMBL's "highest score" fallback rather than an exact name match.
- 0 DICTrank compounds are marked `api_error` in `compounds_with_smiles.csv` (ChEMBL calls failed in the original run), so they were never given a structure. Re-running `02_resolve_smiles.py` for them on a machine with ChEMBL access might add LINCS-matched compounds; this has not been done.
- The "any toxic form -> 1" rule resolves label disagreements only within one InChIKey skeleton; sibling forms with different skeletons (e.g. clindamycin vs clindamycin phosphate) can still disagree (listed in `results/name_match_audit.md`).
- Bootstrap intervals cover test-set sampling only, not training variability.
