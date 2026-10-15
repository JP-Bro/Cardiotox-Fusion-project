# Cardiotox-Fusion — Week 1 Submission (updated)

> **Superseded numbers.** The original Week-1 version of this file reported 517 compounds and 361/78/78 and 322/89/106 splits.
> Those came from a pipeline later found to contain duplicate salt forms and structural leakage. Current, verified numbers are below;
> the single source of truth is `README.md` and `results/pipeline_verification.md`.

**Repository:** https://github.com/JP-Bro/Cardiotox-Fusion-project

## Current deliverables
| File | Content |
|---|---|
| `data/splits/drug_split.csv` | stratified 70/15/15 on unique InChIKey skeletons (340 / 73 / 73) |
| `data/splits/scaffold_split.csv` | Bemis-Murcko scaffold split by compound count (340 / 72 / 74) |
| `data/processed/expression_matrix.csv` | 486 x 978 mean-aggregated landmark z-scores (HA1E / 10 µM / 24 h) |
| `data/processed/lincs_matched_compounds.csv` | matched cohort with LINCS pert_id and signature ids |
| `results/pipeline_rebuild_audit.md` | cohort funnel and dedup log (auto-generated) |
| `results/pipeline_verification.md` | 53 independent checks (auto-generated) |
| `results/name_match_audit.md` | audit of name-matched compounds (auto-generated) |
| `results/classical_baselines_report.md` | 17 classical models with bootstrap CIs (auto-generated) |
| `results/split_stability_report.md` | split-to-split variability of the primary baselines (auto-generated) |
| `results/Phase1_Final_Report.md` | narrative report |

Pre-audit artifacts are in `legacy_week1/` and should not be cited.

## Cohort and splits
* 1,211 labeled DICTrank compounds -> 486 unique chemical skeletons with a strict (HA1E / 10 µM / 24 h) LINCS signature (385 toxic / 101 safe).
* Drug split: 340 / 73 / 73. Scaffold split: 340 / 72 / 74 (scaffold test: 62 toxic / 12 safe).
* Zero cross-split skeleton overlap in both splits.

## Reproduce
```bash
python scripts/01_fetch_labels.py
python scripts/02_resolve_smiles.py --resume
python scripts/03_rebuild_pipeline.py
python scripts/04_baselines.py
python scripts/05_split_stability.py all
python scripts/audit_matches.py
python scripts/verify_pipeline.py   # expects: 53/53 checks passed
```
LINCS raw files must be downloaded first (see `README.md`).
