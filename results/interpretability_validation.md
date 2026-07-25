# Cardiotox-Fusion: Interpretability Validation Report

> **CRITICAL**: Attention weights are NOT validated explanations by themselves.
> This report cross-checks model attention against KNOWN toxicophores from the
> cardiotoxicity literature. Correlation = supporting evidence. No correlation
> = the attention is not capturing toxicophore-relevant features (report honestly).

## Validation Method

For each test compound:
1. Extract GNN atom importance scores (gradient × input)
2. Extract Transformer gene attention weights (CLS => gene attention)
3. Check which known toxicophore SMARTS patterns are present in the molecule
4. Compare: do high-attention atoms correspond to toxicophore substructures?

## Known Toxicophores Checked

| Toxicophore | SMARTS Pattern |
|---|---|
| anthracycline core | `C1CC(=O)c2cc3cc(OC4CC(N)C(O)C(C)O4)c(O)c...` |
| quinone | `O=C1C=CC(=O)C=C1...` |
| nitro group | `[N+](=O)[O-]...` |
| michael acceptor | `C=CC(=O)...` |
| acyl halide | `C(=O)[F,Cl,Br,I]...` |
| herg blocker basic | `N([CX4])([CX4])[CX4]...` |
| epoxide | `C1OC1...` |
| aldehyde | `[CX3H1](=O)[#6]...` |

## Per-Compound Results

### MILTEFOSINE (non-toxic, pred_prob=0.554)

**Known toxicophores present:** herg_blocker_basic
**Top genes by attention:** GENE_376, GENE_541, GENE_567, GENE_844, GENE_51

### NORETHINDRONE (non-toxic, pred_prob=0.307)

**Known toxicophores present:** michael_acceptor
**Top genes by attention:** GENE_376, GENE_541, GENE_844, GENE_567, GENE_51

### DOPAMINE HYDROCHLORIDE (TOXIC, pred_prob=0.752)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_541, GENE_844, GENE_567, GENE_51

### SITAGLIPTIN PHOSPHATE (TOXIC, pred_prob=0.585)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_541, GENE_844, GENE_108, GENE_567

### TOLVAPTAN (TOXIC, pred_prob=0.511)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_541, GENE_567, GENE_844, GENE_51

### LAPATINIB DITOSYLATE (TOXIC, pred_prob=0.618)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_541, GENE_844, GENE_567, GENE_51

### LISINOPRIL (TOXIC, pred_prob=0.521)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_541, GENE_844, GENE_567, GENE_108

### MITOXANTRONE HYDROCHLORIDE (TOXIC, pred_prob=0.737)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_541, GENE_108, GENE_567, GENE_51

### Tepotinib Hydrochloride (non-toxic, pred_prob=0.730)

**Known toxicophores present:** herg_blocker_basic
**Top genes by attention:** GENE_376, GENE_541, GENE_844, GENE_567, GENE_759

### PASIREOTIDE (TOXIC, pred_prob=0.621)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_844, GENE_541, GENE_567, GENE_51

### LEVOKETOCONAZOLE (TOXIC, pred_prob=0.662)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_541, GENE_844, GENE_567, GENE_759

### NEFAZODONE HYDROCHLORIDE (TOXIC, pred_prob=0.749)

**Known toxicophores present:** herg_blocker_basic
**Top genes by attention:** GENE_376, GENE_541, GENE_844, GENE_567, GENE_51

### IVABRADINE HYDROCHLORIDE (TOXIC, pred_prob=0.730)

**Known toxicophores present:** herg_blocker_basic
**Top genes by attention:** GENE_376, GENE_541, GENE_844, GENE_567, GENE_242

### Phentermine (TOXIC, pred_prob=0.771)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_541, GENE_844, GENE_567, GENE_51

### CARISOPRODOL (TOXIC, pred_prob=0.357)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_541, GENE_844, GENE_567, GENE_51

### CHENODIOL (non-toxic, pred_prob=0.294)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_541, GENE_844, GENE_567, GENE_51

### TEMAZEPAM (TOXIC, pred_prob=0.379)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_541, GENE_567, GENE_108, GENE_844

### ATORVASTATIN (non-toxic, pred_prob=0.355)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_844, GENE_1, GENE_541, GENE_567

### METFORMIN HYDROCHLORIDE (TOXIC, pred_prob=0.752)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_541, GENE_844, GENE_567, GENE_1

### DIAZEPAM (TOXIC, pred_prob=0.511)

**Known toxicophores present:** none detected
**Top genes by attention:** GENE_376, GENE_541, GENE_567, GENE_759, GENE_844

## Aggregate Findings

*(Populated after running full test set analysis)*

## Important Caveat

Even where attention correlates with known toxicophores, correlation is not
causation. These findings support model interpretability but should be validated
with domain experts and wet-lab experiments before clinical use.