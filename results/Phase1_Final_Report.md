# Cardiotox-Fusion — Phase 1 Final Report

**Project:** Cardiotoxicity prediction, structure (GNN) vs expression (Transformer), two independent models
**Team:** Jimit Patel, Mallu Vineela Reddy, Aditya Gaurang Patel, Nehal Solanki, Pranjal Chhipa
**Repository:** https://github.com/JP-Bro/Cardiotox-Fusion-project
**Status:** October 2026. Every number below is generated from the result files by `scripts/06_make_reports.py` and checked by `scripts/verify_pipeline.py` (54/54 checks pass at generation time).

---

## 1. Overview

| Branch | Input | Model (Phase 2 spec, `config.py`) |
|---|---|---|
| Structure | 2D molecular graph | 3-layer Graph Isomorphism Network |
| Expression | 978-gene L1000 z-score vector | 2-layer, 4-head Transformer encoder; token = learned gene-identity embedding + z-score projection (no sinusoidal positions: gene order is arbitrary) |

No fusion. Both models are evaluated on the **same** compounds and the **same** scaffold split, the only setting in which structure and biology can be compared. The label is DICTrank (clinical, multi-mechanism); hERG is discussed only as one mechanism, not predicted.

## 2. Dataset construction

1. **Labels** (`01_fetch_labels.py`): 1,318 DICTrank rows; 107 Ambiguous dropped -> 1,211 (868 toxic = 527 Less + 341 Most; 343 safe). Drug name = *Active Ingredient(s)*, falling back to *Generic/Proper Name(s)* for 27 blank rows (mostly withdrawn Most-Concern drugs; the Trade Name column is blank for them).
2. **Structures** (`02_resolve_smiles.py`): ChEMBL name search -> SMILES; salts stripped to the largest fragment. 1,102 compounds resolve (976 unique skeletons); 81 were chosen by ChEMBL's highest-score fallback instead of an exact name; 1 structures are documented manual overrides (`MANUAL_OVERRIDES`); 0 compounds failed with an API error in the original run and were not retried.
3. **LINCS matching** (`03_rebuild_pipeline.py`): Level-5 COMPZ signatures filtered to `trt_cp`, HA1E, 24 h, 10.0 µM. Matching is by InChIKey connectivity block first, then salt-stripped name; a name match is accepted only if LINCS's structure for the perturbagen has the same molecular formula. 490 skeletons have a signature under *any* condition; 486 under the strict condition; **486 after enforcing one LINCS perturbagen per compound**. When a compound has several LINCS perturbagen ids (batches), the one with the most strict signatures is used (ties: smallest id), so the build is deterministic.
4. **Aggregation:** replicate z-vectors per compound are averaged. Mean signatures per compound is 1.09 (528 signatures in total) and only 28 compounds have more than one, so aggregation is a minor step; verified against the GCTX (`verify_pipeline.py` checks 7f/7g).
5. **Landmark genes:** `gene_info` rows with `pr_is_lm == 1` (978), mapped to their real GCTX row indices (0-12,321). The control confirms that the first 978 rows contain only 194 landmark genes.
6. **Dedup and labels:** charges neutralised (RDKit `Uncharger`); one compound per 14-character InChIKey block. Label disagreements within a block (acyclovir; omeprazole/esomeprazole) resolve to toxic.

### Final cohort
| Metric | Value |
|---|---|
| Unique skeletons | **486** (385 toxic / 101 safe; 79.2% toxic) |
| Expression matrix | 486 x 978 |

## 3. Splits (`03_rebuild_pipeline.py`, seed 42)

| Split | Train | Val | Test |
|---|---|---|---|
| Drug-level, stratified | 340 | 73 | 73 |
| Scaffold (primary) | 340 (66 safe) | 72 (23 safe) | 74 (**12 safe**) |

Murcko scaffolds without chirality; 13 acyclic compounds get unique pseudo-scaffolds; packing by compound count with a stable tie order. Zero InChIKey-skeleton overlap, no scaffold spans two partitions, and no test compound has a train neighbour at Tanimoto >= 0.85 (all recomputed independently).

## 4. Statistical power and evaluation strategy

The scaffold test set has 12 safe compounds out of 74. A random scorer's AUC-PR 95% bootstrap interval is **[0.726, 0.944]** (floor 0.838). Evaluation: both branches on the same scaffold split with bootstrap CIs, plus repeated scaffold-grouped CV (section 5b); the cohort stays strict (HA1E / 10 µM / 24 h). Relaxing cell line / dose adds signatures per drug, not drugs (only ~490 skeletons have any signature), so it cannot fix power. Structure-only secondary analysis on the full set is limited to the 976 skeletons with resolved structures (not 1,211) and is never compared with the expression headline.

## 5. Baselines

Scaffold test set N = 74 (62 toxic / 12 safe, 83.8% positive). AUC-ROC and balanced accuracy are primary; AUC-PR is shown with its 0.838 floor.

| Model | AUC-ROC | 95% CI | Balanced Acc | AUC-PR (floor 0.838) | 95% CI |
|---|:-:|:-:|:-:|:-:|:-:|
| Random classifier | 0.500 | — | 0.500 | 0.838 | [0.726, 0.944] |
| Structure: Random Forest (200 trees) | 0.680 | [0.484, 0.863] | 0.542 | 0.910 | [0.837, 0.973] |
| Biology: Logistic Regression (L2) | 0.586 | [0.385, 0.790] | 0.570 | 0.864 | [0.763, 0.966] |

All 17 models, with degenerate (single-class) models flagged, are in `results/classical_baselines_report.md`.

### Interpretation (frozen split)
1. Neither primary baseline has an AUC-ROC interval that excludes 0.5 on this split (structure [0.484, 0.863], biology [0.385, 0.790]).
2. 4 of 15 exploratory models have an interval above 0.5: Structure Random Forest (500 trees) AUC-ROC 0.710 [0.528, 0.871]; Structure Support Vector Classifier (RBF) AUC-ROC 0.714 [0.530, 0.882]; Structure Gradient Boosting (HistGBM) AUC-ROC 0.698 [0.520, 0.860]; Structure K-Nearest Neighbors (k=5, Jaccard) AUC-ROC 0.704 [0.545, 0.849]; with 17 models on one small test set this is unconfirmed until replicated.
3. 7 of 17 models predict a single class (degenerate); their AUC-PR carries no information.
4. Classical results do not say whether a deep model will succeed or fail; whether a learned representation recovers L1000 signal that classical approaches missed is the Phase 2 question, and a null result is reportable.

### Additional limitations
- **One noisy replicate per drug.** The 486 cohort drugs have 528 strict signatures in total (mean 1.09 per drug); 458 drugs (94%) rest on a single signature. Averaging cannot reduce that noise, which is one reason expression models sit at chance.
- **Validation set is small and differently balanced.** It has 72 compounds (23 safe; 68% toxic) against 84% toxic in the test set, so early stopping and thresholds are noisy. The frozen split was not changed after results were seen.
- **Conflicting labels were resolved as 'any toxic -> 1'.** Salt/parent forms of 6 drugs carried different DICTrank labels (acyclovir, amifampridine, clindamycin, hydroxocobalamin, vancomycin, zinc). This biases those drugs towards toxic.

## 5b. Split stability (`scripts/05_split_stability.py`, `results/split_stability_report.md`)

| Model | Frozen split (test AUC-ROC) | 30 random scaffold splits: mean ± SD [min, max] | Pooled 5-fold scaffold CV, 5 repeats: mean [range] | Repeat-0 pooled AUC-ROC, 95% CI |
|---|:-:|:-:|:-:|:-:|
| Structure: Morgan-FP RF | 0.680 [0.484, 0.863] | 0.671 ± 0.086 [0.540, 0.867] | 0.659 [0.638, 0.678] | 0.638 [0.579, 0.697] |
| Biology: L1000 LR (L2) | 0.586 [0.385, 0.790] | 0.467 ± 0.077 [0.304, 0.635] | 0.487 [0.472, 0.504] | 0.504 [0.447, 0.568] |

- **Structure:** pooled scaffold-CV AUC-ROC 0.638 [0.579, 0.697] — the interval excludes 0.5, so there is modest but real signal (about 0.63-0.67 depending on the split), far weaker than the single frozen-split interval suggests.
- **Biology:** pooled scaffold-CV AUC-ROC 0.504 [0.447, 0.568] — the interval includes 0.5: a linear model on L1000 z-scores finds no signal.
- **Noise floor:** the SD of test AUC-ROC across random scaffold splits is 0.086 (structure) / 0.077 (biology); a single-split difference smaller than about 0.17 should not be read as a real difference.
- **Use for Phase 2:** a deep model counts as an improvement only if it beats the pooled-CV baseline of its own modality by more than the split-to-split SD; GIN must clearly exceed about 0.66, the Transformer must clearly exceed 0.5.
- Intervals are bootstrap over compounds with fixed pooled predictions; they do not include training variability.

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

## 6. Corrections since the 10 Sep submission

| Item | Previously reported | Verified |
|---|---|---|
| Cohort size | 487 | **486**: etoposide phosphate and clindamycin phosphate were name-matched to the *same LINCS signature* as etoposide / clindamycin HCl (identical expression, different splits; the clindamycin pair had opposite labels), and other prodrug / ester / mixture-component name matches used a different molecule's expression. Rules added: one perturbagen per compound; a name match needs an identical molecular formula (7 compounds excluded: etoposide phosphate, Clindamycin Phosphate, tedizolid phosphate, FOSTAMATINIB, ixazomib citrate, IVERMECTIN, ESTRADIOL). |
| Reproducibility of the rebuild | "deterministic" | A from-scratch rebuild gave a **different LINCS perturbagen for 8 compounds** and a different scaffold packing, because choices depended on Python set/hash order and an unstable sort. Fixed (most strict signatures, then id; stable sorts); four different `PYTHONHASHSEED` values now give byte-identical outputs. The scaffold test set changed to 12 safe and the baselines were re-run. |
| Scaffold test set | 10 safe | **12 safe** (re-split after the fixes) |
| Primary structure baseline | AUC-ROC 0.723 [0.523, 0.906] "real signal" | 0.613 after removing leaked entries; **0.680 [0.484, 0.863]** after the deterministic rebuild; interval includes 0.5 on the frozen split; pooled scaffold-CV 0.638 [0.579, 0.697] |
| "1,048 drugs with any L1000 signature" | 1,048 | **490 skeletons** (1,048 was the count of ChEMBL-resolved structures) |
| "1,837 signatures across 582 entries" | 1,837 | 1,837 is every HA1E/10 µM/24 h signature in LINCS; the cohort uses **528** (1.09 per compound; not "~4") |
| Biology baselines | "L1 Lasso 0.906 shows signal" (AUC-PR) | AUC-PR floor is 0.838; Lasso AUC-ROC 0.618 [0.412, 0.814] |
| Repo reproducibility | "runs from a clean clone" | `01` read the wrong name column and `02` was a different script from the one that built the data; `requirements.txt` lacked h5py; all fixed and a clean-copy rebuild was compared with the committed data |
| Name-matched compounds | 14, "all valid" | 7 remain, all with identical molecular formula; 7 rejected (`results/pipeline_rebuild_audit.md`, `results/name_match_audit.md`) |

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
1. GIN and Transformer are trained (section above); next: more CV repeats / seeds if the intervals are too wide, and the interpretability analysis.
2. Report all four models (RF, LR, GIN, Transformer) in one table with AUC-ROC and balanced accuracy first, against the noise floor above.
3. Optional data work: re-resolve the 0 `api_error` compounds with ChEMBL access (could add LINCS-matched compounds).
