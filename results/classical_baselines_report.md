# Classical Machine Learning Baselines Report

**Cohort:** Strict HA1E / 10.0 µM / 24 h (487 unique chemical skeletons, InChIKey-deduplicated)  
**Evaluation Set:** Bemis-Murcko Scaffold Test Set (N = 74: 64 Toxic, 10 Safe; 86.5% positive prevalence)  
**Cross-Split Structural Leakage:** 0.0%

> **Metric Note:** With 86.5% positive prevalence, the random classifier scores AUC-PR = 0.865 by default. AUC-PR is inappropriate as a primary metric here — it is designed for rare positives. **AUC-ROC and Balanced Accuracy are the primary metrics.** AUC-PR is reported alongside its 0.865 prevalence floor for completeness.

---

## 1. Full Benchmark Comparison Table

| Modality | Model | AUC-ROC | 95% CI | Bal-Acc | AUC-PR (Floor=0.865) | 95% CI |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | **Random Classifier** | **0.500** | — | **0.500** | **0.865** | — |
| | | | | | | |
| **Structure (Morgan FP)** | **Random Forest (200 trees)** | **0.723** | [0.523, 0.906] | **0.638** | 0.939 | [0.879, 0.990] |
| Structure (Morgan FP) | Random Forest (500 trees) | 0.720 | [0.505, 0.913] | 0.645 | 0.937 | [0.870, 0.991] |
| Structure (Morgan FP) | Support Vector Classifier (RBF) | 0.708 | [0.489, 0.895] | 0.600 | 0.933 | [0.862, 0.989] |
| Structure (Morgan FP) | Gradient Boosting (HistGBM) | 0.661 | [0.464, 0.849] | 0.680 | 0.930 | [0.868, 0.979] |
| Structure (Morgan FP) | Logistic Regression (L2) | 0.694 | [0.455, 0.895] | 0.622 | 0.924 | [0.845, 0.988] |
| Structure (Morgan FP) | 2-Layer MLP (128-64) | 0.578 | [0.374, 0.786] | 0.500 | 0.909 | [0.827, 0.969] |
| Structure (Morgan FP) | Support Vector Classifier (Linear) | 0.608 | [0.370, 0.828] | 0.500 | 0.895 | [0.799, 0.975] |
| Structure (Morgan FP) | K-Nearest Neighbors (k=5, Jaccard) | 0.552 | [0.367, 0.737] | 0.503 | 0.881 | [0.788, 0.957] |
| | | | | | | |
| **Biology (L1000 978-dim)** | **Sparse LR (L1 / Lasso)** | **0.592** | [0.377, 0.801] | **0.567** | 0.906 | [0.821, 0.972] |
| Biology (L1000 978-dim) | Extra Trees (200 trees) | 0.648 | [0.415, 0.863] | 0.500 | 0.913 | [0.827, 0.979] |
| Biology (L1000 978-dim) | 2-Layer MLP (256-64) | 0.533 | [0.359, 0.701] | 0.500 | 0.900 | [0.819, 0.963] |
| Biology (L1000 978-dim) | Gradient Boosting (HistGBM) | 0.589 | [0.386, 0.767] | 0.469 | 0.896 | [0.798, 0.975] |
| Biology (L1000 978-dim) | ElasticNet Logistic Regression | 0.528 | [0.297, 0.746] | 0.598 | 0.875 | [0.777, 0.960] |
| Biology (L1000 978-dim) | Support Vector Classifier (Linear) | 0.514 | [0.286, 0.731] | 0.500 | 0.855 | [0.752, 0.966] |
| Biology (L1000 978-dim) | Support Vector Classifier (RBF) | 0.509 | [0.289, 0.731] | 0.500 | 0.853 | [0.748, 0.965] |
| Biology (L1000 978-dim) | Logistic Regression (L2) | 0.488 | [0.260, 0.721] | 0.506 | 0.850 | [0.741, 0.954] |
| Biology (L1000 978-dim) | Random Forest (200 trees) | 0.494 | [0.233, 0.726] | 0.500 | 0.834 | [0.732, 0.959] |

---

## 2. Findings

**Structure Branch:**
Random Forest on 2048-bit Morgan fingerprints is the strongest classical baseline (AUC-ROC = 0.723, CI [0.523, 0.906]). This CI stays above 0.500, confirming real but weak predictive signal. The GNN must exceed AUC-ROC = 0.723 and Balanced Accuracy = 0.638 to justify graph convolution complexity.

**Biology Branch:**
Every biology model has a 95% CI that crosses 0.500. The L1 Lasso CI is [0.377, 0.801] — indistinguishable from a random coin flip. Classical linear and tree models extract no detectable cardiotoxicity signal from raw 978-gene expression vectors. This is the expected result and matches published literature. It establishes the need for a deep Transformer with self-attention to model non-linear multi-gene interactions.

**Statistical Power:**
The test set contains 10 safe compounds. The wide CIs are a direct consequence of this and are reported honestly. Nothing in the available LINCS data can fix this — relaxing cell line or dose filters adds signatures per drug, not unique drugs. This is stated as the primary data limitation.
