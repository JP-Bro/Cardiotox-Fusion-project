# Pipeline Rebuild Audit Log

## Step 1: Verification of Gene Columns
- Extracted 978 landmark genes from GCTX.
- First 5 genes: ['780', '7849', '2978', '2049', '2101']
- Verified that these matched the intended landmark subset.

## Step 2: LINCS Matching
- Filtered LINCS signatures to HA1E, 24h, 10.0 uM, trt_cp.
- Total signatures meeting criteria: 1837
- Average replicates per drug: 1.08

## Step 3 & 4: Extraction & Deduplication
- Removed 68 salt duplicates based on `parent_smiles`.
- Final matched compounds count: 515
- Extracted expression vectors for these compounds and mean-aggregated replicates.

## Step 5: Data Splitting
- Stratified 70/15/15 random split generated on drug level (`drug_split.csv`).
- Scaffold split generated (`scaffold_split.csv`).
- Train size (drug): 359
- Val size (drug): 78
- Test size (drug): 78
