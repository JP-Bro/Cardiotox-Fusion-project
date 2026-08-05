# Pipeline Rebuild Audit Log

## Step 1: Verification of Gene Columns
- Extracted 978 landmark genes from GCTX.
- Verified that columns in `expression_matrix.csv` match the first 978 GCTX row IDs (NCBI landmark gene IDs).

## Step 2: LINCS Matching
- Filtered LINCS signatures to HA1E, 24h, 10.0 µM, trt_cp.
- Total signatures meeting criteria: 1,837
- Resolved SMILES for matched compounds (including Fulvestrant, Ixabepilone, Ivermectin).

## Step 3 & 4: Extraction, Deduplication & Aggregation
- Removed 65 salt duplicates based on `parent_smiles`.
- Final clean matched compounds count: **517 unique parent structures**.
- Extracted expression vectors for these compounds and mean-aggregated replicate z-scores.

## Step 5: Data Splitting
- Stratified 70/15/15 random split generated on drug level (`drug_split.csv`), grouped by `parent_smiles`.
- Scaffold split generated (`scaffold_split.csv`), grouped by Bemis-Murcko scaffolds.
- Cross-split SMILES leakage: **0% (Exactly 0 leaks)**.
- Train size (drug): 361
- Val size (drug): 78
- Test size (drug): 78
