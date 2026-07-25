# Cardiotox-Fusion — Complete Project Documentation

> **Working Name:** Cardiotox-Fusion  
> **Full Title:** Multimodal Fusion of Molecular Structure and Transcriptomic Response for Drug-Induced Cardiotoxicity Prediction  
> **Status:** Phase 1 (Mandatory Trace-Through) ✅ Complete → Phase 2 (Full Pipeline Scripts) 🔄 In Progress  
> **Last Updated:** 2026-07-23

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Goal & Scientific Hypothesis](#2-project-goal--scientific-hypothesis)
3. [Architecture](#3-architecture)
4. [Repository Structure (What File Contains What)](#4-repository-structure-what-file-contains-what)
5. [Datasets — Sources, Roles & Status](#5-datasets--sources-roles--status)
6. [How It Was Done — Step-by-Step Pipeline](#6-how-it-was-done--step-by-step-pipeline)
7. [Problems Encountered & How They Were Fixed](#7-problems-encountered--how-they-were-fixed)
8. [Design Decisions (Logged, Not Silent)](#8-design-decisions-logged-not-silent)
9. [Key Numbers & Findings](#9-key-numbers--findings)
10. [Known Risks & Limitations](#10-known-risks--limitations)
11. [Related / Precedent Work](#11-related--precedent-work)
12. [What's Next — Remaining Pipeline Scripts](#12-whats-next--remaining-pipeline-scripts)
13. [Environment Setup (How to Reproduce)](#13-environment-setup-how-to-reproduce)

---

## 1. Project Overview

This project predicts whether a drug compound is likely to be **cardiotoxic** (harmful to the heart) by fusing two independent signals:

| Signal | What It Captures |
|---|---|
| **Molecular structure** | 2D graph of the drug molecule (atoms + bonds), derived from SMILES |
| **Transcriptomic response** | Gene expression changes triggered by the drug in cells (LINCS L1000, 978 landmark genes) |

Instead of relying on structural features alone (the common baseline in prior work) or on gene expression alone, this model **fuses both branches via cross-attention**, learning which structural features co-occur with which biological responses in cardiotoxic compounds.

**Output:** A binary classifier (cardiotoxic = 1 / non-cardiotoxic = 0), with an interpretability layer that highlights which molecular substructures and which genes drove the prediction.

---

## 2. Project Goal & Scientific Hypothesis

**Hypothesis:** Structural features and transcriptomic response carry **complementary** information about cardiotoxic risk — not redundant. Cross-attention fusion should therefore outperform either single-branch model.

**Honest expectation (from prior literature):**  
- Structure-only AUC-ROC ≈ 0.84 (Seal et al. 2023)  
- Biology-only AUC-ROC ≈ 0.76 (Seal et al. 2023)  
- A modest fusion gain is a legitimate, publishable outcome — not guaranteed to be dramatic.

**Target novelty over prior work:** Prior work (Seal et al.) used GO-annotation-derived biological features from LINCS. This project uses a **GNN + Transformer + cross-attention** architecture, which is more expressive and the architectural novelty angle.

---

## 3. Architecture

```
SMILES string
     │
     ▼
[RDKit: mol graph]
     │
     ▼
┌─────────────────────┐
│  GNN Branch         │   ← PyTorch Geometric / DGL
│  (Molecular Graph)  │   → Structure Embedding (d-dim vector)
└──────────┬──────────┘
           │
           ├─────────────────────────────────┐
           │                                 │
           ▼                                 ▼
   [Cross-Attention Fusion Layer]            │
           │                                 │
           │       ┌─────────────────────────┘
           │       │
           │  ┌────┴───────────────────────────┐
           │  │  Transformer Branch             │   ← PyTorch Transformer
           │  │  (LINCS L1000 gene expression) │   → Biology Embedding (d-dim vector)
           │  └────────────────────────────────┘
           │
           ▼
  [Joint Representation]
           │
           ▼
  [Classification Head]
           │
           ▼
  Binary label: 0 = non-cardiotoxic, 1 = cardiotoxic
```

- **GNN branch:** Standard molecular property prediction GNN (e.g., GCN, MPNN) operating on atom/bond features from RDKit.
- **Transformer branch:** Treats the 978 L1000 landmark gene expression values as a sequence/set of features; learns which genes matter.
- **Cross-attention fusion:** Each branch can attend to the other — richer than simple concatenation, allows the model to learn co-occurrences.
- **Interpretability:** GNN attention weights → cross-checked against known cardiotoxicity toxicophores (structural alerts in the literature). NOT asserted as explanation without validation.

---

## 4. Repository Structure (What File Contains What)

```
group_project/
├── cardiotox_fusion.ipynb               ← Main Jupyter notebook (trace-through work)
└── cardiotox-fusion/
    ├── PROJECT_DOCUMENTATION.md         ← THIS FILE — master reference for the project
    │
    ├── data/
    │   ├── raw/
    │   │   ├── dictrank_dataset_508.xlsx     ← FDA DICTrank cardiotoxicity labels
    │   │   │                                    (1,318 rows × 8 cols, 94,888 bytes)
    │   │   │                                    Downloaded from: fda.gov/media/178811/download
    │   │   ├── crediblemeds_dta_list_2026-07-23.pdf
    │   │   │                                 ← CredibleMeds drug-induced arrhythmia list
    │   │   │                                    (secondary label source, PDF)
    │   │   └── lincs/
    │   │       ├── sig_info.txt.gz           ← LINCS L1000 GSE70138 signature metadata
    │   │       │                                (118,050 signatures, ~2.1 MB compressed)
    │   │       └── pert_info.txt.gz          ← LINCS L1000 perturbation/compound metadata
    │   │                                        (2,170 unique compounds, ~82 KB compressed)
    │   │
    │   └── processed/
    │       └── trace_through_log.txt         ← Detailed log of the manual trace-through
    │                                            (Steps 1–3b: DICTrank, SMILES, LINCS)
    │
    ├── scripts/
    │   └── 05_trace_ten_compounds.py         ← MANDATORY pre-pipeline trace script
    │                                            Formalizes the notebook trace-through work;
    │                                            must be run and verified before full pipeline
    │
    ├── models/                               ← (Empty — model weights saved here after training)
    ├── notebooks/                            ← (Working/scratch notebooks)
    └── results/                              ← (Empty — results/reports saved here after evaluation)
```

### Scripts Not Yet Written (Planned)

| Script | Purpose | Depends On |
|---|---|---|
| `scripts/01_fetch_labels.py` | Download DICTrank + CredibleMeds, produce clean labeled CSV | — |
| `scripts/02_resolve_smiles.py` | ChEMBL API → SMILES for all 1,211 compounds, with salt handling | 01 |
| `scripts/03_match_lincs.py` | Name + InChIKey matching against LINCS, download 5 GB GCTX if coverage confirmed | 02 |
| `scripts/04_build_graphs.py` | SMILES → PyG/DGL molecular graphs, log every stripped salt fragment | 02, 03 |
| `scripts/06_train_gnn.py` | Train structure-only GNN baseline | 04 |
| `scripts/07_train_transformer.py` | Train biology-only Transformer baseline | 03 |
| `scripts/08_train_fusion.py` | Train GNN + Transformer + cross-attention fusion model | 04 |
| `scripts/09_evaluate.py` | Evaluate all three models on same LINCS-covered test set, produce `results/baseline_comparison.md` | 06, 07, 08 |
| `scripts/10_interpretability.py` | Extract attention weights, cross-check against known toxicophores | 09 |

---

## 5. Datasets — Sources, Roles & Status

### 5.1 DICTrank (Primary Label Source) ✅ Downloaded

- **Full name:** FDA Drug-Induced Cardiotoxicity Rank
- **Downloaded from:** `https://www.fda.gov/media/178811/download`
- **Local path:** `data/raw/dictrank_dataset_508.xlsx`
- **Size:** 94,888 bytes (confirmed non-trivial — not an error page)
- **Contents:** 1,318 rows × 8 columns
- **Key column:** `DICT_Concern` — severity levels: `most`, `less`, `no`, `ambiguous`
- **Binarization rule:**
  - `no` → label 0 (non-cardiotoxic)
  - `less` + `most` → label 1 (cardiotoxic concern)
  - `ambiguous` → **dropped**
- **After binarization:** 1,211 usable labeled compounds (868 positive, 343 negative)

### 5.2 CredibleMeds ✅ Downloaded (Secondary)

- **Local path:** `data/raw/crediblemeds_dta_list_2026-07-23.pdf`
- **Role:** Alternative/secondary label source — drugs known to cause arrhythmia
- **Format:** PDF list (not a structured dataset)
- **Status:** Downloaded; not yet parsed/merged into the label set

### 5.3 LINCS L1000 — GSE70138 Level 5 (Biology Branch Input) 🔄 Partial

- **Source:** NCBI GEO dataset GSE70138, Phase II collection (~2015–2017)
- **Files downloaded so far:**
  - `data/raw/lincs/sig_info.txt.gz` (118,050 signatures, ~2.1 MB) ✅
  - `data/raw/lincs/pert_info.txt.gz` (2,170 unique compounds, ~82 KB) ✅
  - **Level 5 expression matrix (GCTX, ~5.0 GB):** ❌ NOT yet downloaded (deliberately — confirm match rate first)
- **Perturbation features:** 978 landmark gene expression values per signature
- **Coverage of our labeled set:** 423/1,211 compounds by name-matching (34.9%)
- **Condition used:** `cell_id = HA1E`, highest dose (10 µM), 24h timepoint

### 5.4 ChEMBL (SMILES Source) 🔄 API-Access Only

- **Role:** Resolves drug names → SMILES strings
- **Access:** REST API at `https://www.ebi.ac.uk/chembl/api/data/molecule/search`
- **No local download** — called per-compound during the pipeline
- **Status:** Used successfully in `05_trace_ten_compounds.py` for all 10 sample compounds

### 5.5 FAERS (Possible Auxiliary Signal) ❌ Not Scoped

- **Role:** FDA Adverse Event Reporting System — pharmacovigilance signal
- **Status:** Identified as a candidate data source; not yet integrated or planned in detail

---

## 6. How It Was Done — Step-by-Step Pipeline

### Phase 1: Mandatory Manual Trace-Through (COMPLETE ✅)

**Location:** `cardiotox_fusion.ipynb` (main notebook) + formalized in `scripts/05_trace_ten_compounds.py`

The project brief required manually tracing ~10 real compounds through the full pipeline **before** writing any automated training code — to surface real failure modes early rather than mid-training.

#### Step 1 — DICTrank Label Acquisition
1. Downloaded `dictrank_dataset_508.xlsx` directly from FDA
2. Loaded with `pd.read_excel()` (needed `openpyxl` — added after initial failure)
3. Cleaned column names: stripped whitespace, renamed `DICT _ Concern` → `DICT_Concern`
4. Normalized `DICT_Concern` values: `.str.strip().str.lower()` (found inconsistent casing: `"less"`, `"Less"`, `"less "` were counted as 3 categories)
5. Verified post-cleaning counts: `no=343, less=527, most=341, ambiguous=107` (total 1,318)
6. Noted 1-row discrepancy vs. FDA's stated `less=528, ambiguous=106` — logged, non-blocking
7. Binarized and dropped ambiguous rows → 1,211 usable labeled compounds

#### Step 2 — SMILES Resolution via ChEMBL
1. Sampled 10 compounds from the labeled set (seed=42) — mixed label distribution
2. For each compound:
   - Looked up `Active Ingredient(s)` column; fell back to `Generic/Proper Name(s)` if missing
   - Queried ChEMBL REST API: `GET /molecule/search?q=<name>&format=json`
   - Applied disambiguation rule: prefer exact `pref_name` match over highest `score`
   - Applied salt handling: parsed SMILES with RDKit, took largest fragment by heavy-atom count
3. Full-dataset missing-name check (all 1,211): 27/1,211 (2.2%) missing `Active Ingredient(s)`, but 0/1,211 missing `Generic/Proper Name(s)` — fallback is safe universally
4. **Result: 10/10 resolved successfully**

#### Step 3 — LINCS L1000 Matching
1. Downloaded only metadata files first (`sig_info.txt.gz`, `pert_info.txt.gz`) — not the 5 GB expression matrix
2. Loaded both with pandas; created lowercase/stripped `pert_iname_lower` column
3. For each compound: tried direct name match, then salt-suffix-stripped match, then InChIKey match
4. **10-compound sample result:** 6/10 matched (60%), 4/10 absent
5. **Full 1,211-compound dataset (name-matching only):** 423/1,211 matched (34.9%)

### Phase 2: Full Pipeline Scripts (PLANNED / IN PROGRESS)

To be built using `scripts/01_fetch_labels.py` through `scripts/10_interpretability.py` (see §4).

---

## 7. Problems Encountered & How They Were Fixed

### Problem 1 — Python Not Found on Windows ❌ → ✅ Fixed

**What happened:**  
Running `python` in the terminal launched the Microsoft Store instead of an interpreter.

**Root cause:**  
Windows has "App Execution Aliases" that intercept `python.exe` and `python3.exe` and redirect them to the Store app when no real Python is installed.

**Fix:**  
1. Opened **Windows Settings → Apps → Advanced App Settings → App Execution Aliases**
2. Disabled both `python.exe` and `python3.exe` aliases
3. Installed **Python 3.12.10** from [python.org](https://www.python.org) with **"Add to PATH"** checked during setup
4. Verified: `python --version` returned `Python 3.12.10`

---

### Problem 2 — `pd.read_excel()` Failing (Missing `openpyxl`) ❌ → ✅ Fixed

**What happened:**  
`pd.read_excel("dictrank_dataset_508.xlsx")` raised `ImportError: Missing optional dependency 'openpyxl'`.

**Root cause:**  
pandas can read `.xlsx` files but requires `openpyxl` as a backend, and it is not bundled with pandas by default.

**Fix:**  
```bash
pip install openpyxl
```

---

### Problem 3 — Inconsistent Label Casing in `DICT_Concern` Column ❌ → ✅ Fixed

**What happened:**  
The `DICT_Concern` column had three distinct string variants for the same label: `"less"`, `"Less"`, and `"less "` (trailing space). Naive `.value_counts()` showed them as separate categories.

**Fix:**  
```python
df["DICT_Concern"] = df["DICT_Concern"].str.strip().str.lower()
```
Applied **before** any grouping or binarization.

---

### Problem 4 — Column Name Had Trailing Space ❌ → ✅ Fixed

**What happened:**  
`'Label Section '` (with trailing space) and `'DICT _ Concern'` (inconsistent spacing) caused `KeyError` when accessed by their expected names.

**Fix:**  
```python
df.columns = df.columns.str.strip()
df = df.rename(columns={"DICT _ Concern": "DICT_Concern"})
```

---

### Problem 5 — ChEMBL Returns Multiple Candidates for One Drug Name ❌ → ✅ Fixed

**What happened:**  
Querying `"OLAPARIB"` returned 2 molecules: the actual drug (CHEMBL2107691) and an unrelated analog (AZD2461). Taking `result[0]` naively would have given the correct answer by luck — but not reliably for other compounds.

**Fix (Design Decision):**  
Prefer exact `pref_name` match (case-insensitive) over the highest relevance `score`. Only fall back to highest score when no exact name match exists, and log whenever the fallback path is used.

```python
exact_matches = [m for m in molecules
                 if (m.get("pref_name") or "").lower() == drug_name.lower()]
if exact_matches:
    chosen = exact_matches[0]
    match_method = "exact_pref_name_match"
else:
    chosen = max(molecules, key=lambda m: m.get("score") or 0)
    match_method = "fallback_highest_score"
```

---

### Problem 6 — ChEMBL `pref_name` Field Returns `None` (Not Just Missing) ❌ → ✅ Fixed

**What happened:**  
For some molecules, `pref_name` is present in the JSON response but its value is `None` (Python `NoneType`). A naive `.get("pref_name", "").lower()` call crashed with `AttributeError: 'NoneType' object has no attribute 'lower'`.

**Fix:**  
Guard against `None` explicitly using `or ""`:
```python
(m.get("pref_name") or "").lower()
```

---

### Problem 7 — 30% of Drugs Are Salts (Multi-Component SMILES) ❌ → ✅ Fixed

**What happened:**  
3 of 10 sample compounds returned SMILES with `.` separator — indicating multi-component mixtures (drug + counter-ion or buffer molecule):
- `SONIDEGIB PHOSPHATE` → drug + 2 phosphate ions
- `FLAVOXATE HYDROCHLORIDE` → drug + chloride ion
- `CARBOPROST TROMETHAMINE` → drug + tromethamine buffer

Passing these directly to RDKit's graph builder would have created graphs of the wrong molecule.

**Fix (Design Decision):**  
Extract the largest fragment by heavy-atom count as the "parent" drug:
```python
frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
frags_sorted = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
parent = frags_sorted[0]
```
Every stripped fragment is **logged explicitly** (never silently discarded). Verified correct on all 3 salt cases by manual inspection of the resulting SMILES.

---

### Problem 8 — Salt-Suffix Stripping Not Sufficient for LINCS Matching ❌ → Partially Fixed

**What happened:**  
Stripping `"hydrochloride"` recovered the LINCS match for `FLAVOXATE HYDROCHLORIDE`. But stripping `"phosphate"` from `SONIDEGIB PHOSPHATE` still returned zero LINCS matches. Similarly for `CARBOPROST TROMETHAMINE`.

**Investigation:**  
Performed **InChIKey-based structure matching** (immune to naming differences entirely — computed InChIKey from resolved SMILES, compared against LINCS's own `inchi_key` column). Confirmed that sonidegib and carboprost are **genuinely absent from LINCS Phase II** — not a matching bug.

**Likely reasons:**
- Sonidegib approved 2015 — sits at the edge of the LINCS Phase II collection window (~2015–2017)
- Carboprost is an obstetric drug, unlikely to be in a cancer/toxicology chemical perturbation panel

**Fix:**  
Accept genuine absence. For the full pipeline, use InChIKey matching as the **primary / more robust** matching strategy in `scripts/03_match_lincs.py`, with name-matching as a fast first pass. Accept that coverage will not be 100%.

---

### Problem 9 — Propoxyphene's `Active Ingredient(s)` Field Was Empty ❌ → ✅ Fixed

**What happened:**  
One of the 10 sample compounds (Propoxyphene) had `NaN` in the `Active Ingredient(s)` column — which is used as the primary drug name for ChEMBL lookup and LINCS matching.

**Investigation:**  
Checked this across the **full 1,211-row dataset** (not just the sample):  
- 27/1,211 (2.2%) have missing `Active Ingredient(s)`  
- 0/1,211 have missing `Generic/Proper Name(s)`

**Fix:**  
Universal fallback rule:
```python
name = row["Active Ingredient(s)"]
if pd.isna(name):
    name = row["Generic/Proper Name(s)"]
```
Safe to apply universally — no dead-ends in the full dataset.

---

### Problem 10 — LINCS Coverage Unevenly Distributed Across Labels ⚠️ Logged

**What happened:**  
The LINCS-matched subset (423/1,211 = 34.9%) is **not a random sample** of the labeled dataset — it over-represents cardiotoxic compounds:
- Label 0 (no concern): 96/343 = **28.0%** covered
- Label 1 (concern): 327/868 = **37.7%** covered

**Implication:**  
The fused model will train on a more imbalanced dataset than the full 1,211-compound set. The ~10-point gap must not be silently absorbed.

**Fix (Design Decision):**  
- Use **stratified train/test split** (preserving class balance within the matched subset)
- Report actual class balance in `results/baseline_comparison.md`
- All three models (structure-only, biology-only, fused) must be evaluated on the **same LINCS-covered held-out test set** for a fair comparison

---

## 8. Design Decisions (Logged, Not Silent)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Name resolution fallback:** Use `Active Ingredient(s)`, fall back to `Generic/Proper Name(s)` | 27/1,211 rows missing AI; GPN is never missing. Zero dead-ends confirmed. |
| 2 | **ChEMBL disambiguation:** Prefer exact `pref_name` match over highest score | Score-based ranking can select unrelated analogs when multiple candidates exist (OLAPARIB example). Log when fallback is used. |
| 3 | **Salt handling:** Parent molecule = largest fragment by heavy-atom count | RDKit convention; verified correct on 3 real cases. Every stripped fragment logged — never silently discarded. |
| 4 | **LINCS matching strategy:** Name-match first (incl. salt-suffix stripping); InChIKey structure-match as primary/robust fallback | Name matching is fast and catches most; InChIKey is immune to naming variants. Use both. |
| 5 | **LINCS condition selection:** `cell_id = HA1E`, highest available dose, 24h timepoint | HA1E is present for all matched compounds so far; relatively non-transformed reference line vs. cancer lines. Chosen explicitly — not a silent `.iloc[0]`. |
| 6 | **Fair comparison requirement:** All three models evaluated on the same LINCS-covered test subset | Structure-only could train on all 1,211 but must be evaluated on the same ~423-compound subset as the fusion model for the comparison to be honest. |

---

## 9. Key Numbers & Findings

| Metric | Value |
|---|---|
| Total DICTrank compounds | 1,318 |
| After dropping ambiguous labels | **1,211** |
| Label balance (full set) | 868 positive (concern) / 343 negative (no concern) |
| SMILES resolution success (10-compound sample) | **10/10 (100%)** |
| Compounds needing salt-stripping (sample) | **3/10 (30%)** |
| LINCS name-match coverage (full 1,211 compounds) | **423/1,211 (34.9%)** |
| LINCS coverage — label 0 (no concern) | 96/343 (28.0%) |
| LINCS coverage — label 1 (concern) | 327/868 (37.7%) |
| LINCS coverage imbalance | ~10 percentage points MORE positive coverage |
| Practical fused-model training set size | ~423–450 compounds (after InChIKey matching adds a few more) |
| Structure-only baseline AUC-ROC (prior work) | ~0.84 (Seal et al. 2023) |
| Biology-only baseline AUC-ROC (prior work) | ~0.76 (Seal et al. 2023) |

---

## 10. Known Risks & Limitations

### R1 — LINCS Cell Lines Are Not Cardiomyocytes
The 7 LINCS core cell lines (A375, HA1E, HELA, HT29, MCF7, PC3, YAPC) are **all cancer-derived, none cardiac**. The biology branch will learn a general cross-tissue transcriptional toxicity signature — not a cardiac-specific one. This is a fundamental biological limitation, not an implementation gap. It matches Seal et al.'s own approach and their ~0.76 AUC-ROC ceiling for biology-only. **Must be stated plainly in the final write-up.**

### R2 — Match-Rate Limits Dataset Size
The fused model is capped at ~423–450 compounds. The structure-only baseline can use the full 1,211 — so the comparison requires care (both evaluated on the LINCS-covered subset). Final count will increase slightly once InChIKey-based matching is run on the full dataset.

### R3 — Fusion May Not Dramatically Outperform Single-Branch Baselines
A "fusion helps modestly" result is the most likely honest outcome. The project is designed to handle this: prior work is cited honestly, and a modest but real fusion gain is presented as a legitimate, publishable finding.

### R4 — Attention ≠ Explanation
Raw GNN attention maps are NOT presented as validated explanations. Interpretability claims are validated by cross-checking highlighted substructures against established toxicophores in the cardiotoxicity literature.

### R5 — 1-Row Discrepancy in DICTrank Labels
Our post-cleaning count: `less=527, ambiguous=107`. FDA's stated count: `less=528, ambiguous=106`. Discrepancy is 1 row — likely a data version or whitespace edge case. Logged, non-blocking.

### R6 — Timeline Risk
The original estimate was 10–12 weeks. Data cleaning and matching typically take longer than planned. Treat as optimistic.

---

## 11. Related / Precedent Work

**Seal et al. (2023):** A study on FDA DICTrank classifiers that explored combining chemical structure data with LINCS L1000 gene expression to predict cardiotoxicity.

| Model | AUC-ROC |
|---|---|
| Random baseline | 0.50 |
| Biology-only (LINCS → GO features) | 0.76 |
| Structure-only | **0.84** |
| Random Forest ensemble | — |

**Key takeaways for this project:**
- Structure is the stronger single signal — fusion must beat 0.84 to claim a clear win
- The prior work used GO-annotation features derived from LINCS (indirect). This project uses raw L1000 profiles fed to a Transformer (more direct, more expressive)
- The architectural novelty (GNN + Transformer + cross-attention) is the main contribution over prior work

---

## 12. What's Next — Remaining Pipeline Scripts

In order of execution (each depends on the previous):

### Step 1 → `scripts/01_fetch_labels.py`
- Re-download and clean DICTrank from FDA
- Parse CredibleMeds PDF → structured label list
- Cross-check both sources; produce `data/processed/labeled_compounds.csv`

### Step 2 → `scripts/02_resolve_smiles.py`
- Apply ChEMBL lookup + disambiguation + salt-handling rules to all 1,211 compounds
- Produce `data/processed/compounds_with_smiles.csv`

### Step 3 → `scripts/03_match_lincs.py`
- Apply name-matching + InChIKey-based matching to all resolved compounds
- **If coverage is confirmed adequate:** download the Level 5 GCTX matrix (~5 GB) and parse with `cmapPy`
- Produce `data/processed/lincs_matched_compounds.csv` + `data/processed/expression_matrix.csv`

### Step 4 → `scripts/04_build_graphs.py`
- SMILES → PyTorch Geometric / DGL molecular graphs
- Log every stripped salt fragment per compound
- Save graph objects to `data/processed/graphs/`

### Steps 5–8 → Model Training Scripts
- `06_train_gnn.py` — Structure-only GNN baseline
- `07_train_transformer.py` — Biology-only Transformer baseline
- `08_train_fusion.py` — GNN + Transformer + cross-attention fusion

### Step 9 → `scripts/09_evaluate.py`
- Evaluate all three models on the **same** LINCS-covered held-out test set
- Produce `results/baseline_comparison.md` with AUC-ROC, PR curves, class balance info

### Step 10 → `scripts/10_interpretability.py`
- Extract GNN attention weights
- Cross-check highlighted substructures against known cardiotoxicity toxicophores
- Produce `results/interpretability_validation.md`

---

## 13. Environment Setup (How to Reproduce)

### OS: Windows 10/11

#### Step 1 — Fix Python on Windows
If `python` opens the Microsoft Store:
1. Open **Settings → Apps → Advanced App Settings → App Execution Aliases**
2. Toggle **OFF** both `python.exe` and `python3.exe`
3. Download and install **Python 3.12.10** from [python.org](https://www.python.org/downloads/)
4. ✅ Check **"Add Python to PATH"** during installation

#### Step 2 — Install Dependencies
```bash
pip install jupyter notebook
pip install rdkit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install pandas numpy requests openpyxl
```

> **Note:** PyTorch Geometric (`torch_geometric`) may require additional steps depending on your CUDA version. See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

For future steps (LINCS expression matrix):
```bash
pip install cmapPy  # for reading GCTX files
```

#### Step 3 — Run the Mandatory Trace Script
```bash
# From the group_project directory:
python cardiotox-fusion/scripts/05_trace_ten_compounds.py
```

Expected output: Trace log appended to `data/processed/trace_through_log.txt`.

#### Step 4 — Launch Jupyter for Notebook Work
```bash
jupyter notebook
```
Then open `cardiotox_fusion.ipynb`.

---

*This document is the single source of truth for the Cardiotox-Fusion project. Update it whenever new design decisions are made, new problems are encountered, or new results are obtained.*
