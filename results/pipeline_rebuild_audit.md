# Master Pipeline Rebuild & Data Audit Log

## Point 1: Verified Landmark Genes (pr_is_lm == 1)
- Source: `GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz`
- Verified total landmark genes: 978
- Extracted exact GCTX row indices corresponding to `pr_is_lm == 1`.

## Point 2 & 3: Neutralisation & InChIKey Skeleton Deduplication
- Neutralised formal charges using RDKit `Uncharger`.
- Resolved label conflict: ACYCLOVIR (0) vs ACYCLOVIR SODIUM (1) -> Assigned Label 1.
- Grouped compounds by **14-character InChIKey connectivity block**.
- Total clean unique chemical skeletons: **487**.
- Drug-level cross-split structural leakage: **0 leaks (0%)**.

## Point 4: Scaffold Split Rebuild
- Computed Murcko Scaffolds without chirality (`includeChirality=False`).
- Handled 15 acyclic compounds by assigning individual pseudo-scaffold IDs.
- Allocated scaffold clusters using compound-count-proportional greedy packing.
- Scaffold Split compound counts:
  - Train: 340
  - Validation: 73
  - Test: 74

## Point 5: Sample Size & Statistical Power
- Scaffold test set size: 74 compounds (10 negatives).
- Documented AUC-PR 95% Confidence Interval for Random Classifier: `[0.785, 0.915]`.
- Evaluation expandability: Full DICTrank dataset (1,211 drugs, 343 negatives) is available for GNN structure branch benchmarking.
