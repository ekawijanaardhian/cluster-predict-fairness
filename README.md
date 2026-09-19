# When architecture is not a fairness remedy: a utility-matched evaluation of cluster-then-predict pipelines for diabetes risk prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Scikit-Learn / Fairlearn / LightGBM](https://img.shields.io/badge/frameworks-Scikit--Learn%20%7C%20Fairlearn%20%7C%20LightGBM-orange.svg)](https://fairlearn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official computational reproduction repository for:
> **When architecture is not a fairness remedy: a utility-matched evaluation of cluster-then-predict pipelines for diabetes risk prediction**  
> *Ardhian Ekawijana, Mochamad Teguh Kurniawan, Sugeng Rifqi Mubaroq* (2026)

---

## Overview

Fairness-aware machine learning in population health is conventionally framed as a post hoc or in-processing mitigation problem. An alternative view—the **structural-fairness hypothesis**—posits that disparities stem from architectural rigidity: modelling a heterogeneous population with a single monolithic estimator induces bias, and partitioning the population into clustered sub-groups with dedicated expert models provides a structural remedy.

This study empirically evaluates the structural-fairness hypothesis against a **utility-matched counterfactual** using the CDC Behavioral Risk Factor Surveillance System 2015 cohort (BRFSS 2015; $N = 253{,}680$; 21 features), evaluating household income and educational attainment as protected socioeconomic attributes.

### Key Findings
1. **The Parity Gain is an Artefact of Underfitting:** Hard cluster-then-predict partitioning reduces Demographic Parity Difference ($\text{DPD}$) from $0.2909$ to $0.0324$, but predictive discrimination collapses ($\text{AUC} = 0.7022$, with Expected Calibration Error rising to $\text{ECE} = 0.2068$).
2. **Restoring Utility Dissolves Parity:** When predictive utility is recovered through a global base residual and cluster-wise calibration ($\text{AUC} = 0.8261$), demographic disparity returns ($\text{DPD} = 0.2957$), matching the unmitigated single-model baseline.
3. **Pre-Processing Outperforms Architectural Mitigation:** Pre-processing reweighing achieves $\text{DPD} = 0.1234$ with an AUC reduction of only $0.0101$ in $0.46\text{ s}$, outperforming structural partitioning strategies across both parity and decision-curve clinical net benefit.
4. **Search Space Efficiency:** An informed A\* heuristic tree search synthesizes optimal pipeline configurations across standard ($N=18$) and scaled ($N=105$) candidate spaces with a $97.1\%$ reduction in pipeline evaluations relative to exhaustive brute-force search.

---

## Repository Structure

```
.
├── data/
│   ├── diabetes_binary_health_indicators_BRFSS2015.csv  # Curated CDC BRFSS 2015 dataset (N=253,680)
│   ├── codebook15_llcp.pdf                              # Official CDC BRFSS 2015 survey codebook
│   └── loader.py                                        # Data loading and binarization pipeline
├── src/
│   ├── __init__.py
│   ├── astar_search.py           # Informed A* heuristic search and baseline brute-force search
│   ├── pipeline.py               # Unified Cluster-then-Predict pipeline (Hard vs Hierarchical MoE)
│   ├── clustering.py             # Stage 1 Population Clusterer (K-Means, MiniBatch, GMM, Auto)
│   ├── classifiers.py            # Stage 2 Estimators & GlobalResidualClassifier (MoE)
│   ├── preprocessing.py          # Data standardization and Reweighing sample weight computation
│   ├── postprocessing.py         # Validation threshold calibration and decision policies
│   ├── metrics.py                # AUC, DPD, DPR, EOD, Brier score, ECE, DCA Net Benefit
│   └── evaluation.py             # Decision curve analysis and calibration profiling
├── results/
│   ├── figures/                  # Publication figures (Fig 1 - Fig 6, 600 DPI TIFF/PNG/PDF/EPS)
│   ├── submission_package/       # Complete submission artifacts (Manuscript & Supplementary)
│   ├── benchmark_mitigation_baselines.csv
│   ├── statistical_bootstrap_inference.csv
│   ├── subgroup_calibration_brier_ece.csv
│   ├── full_decision_curve_analysis_dca.csv
│   └── scaled_astar_vs_bruteforce_search.csv
├── run_rigorous_empirical_suite.py       # Main empirical experiment runner (Experiments 1–6)
├── run_standardized_bootstrap_protocol.py# B=2,000 paired stratified bootstrap inference
├── run_astar_vs_bruteforce.py            # Standard 18-pipeline search benchmark
├── run_scaled_astar_vs_bruteforce.py     # Scaled 105-pipeline search benchmark
├── run_q1_empirical_campaign.py          # Lambda sweep, 5-seed stability, and cross-attribute transfer
├── calculate_multitest_corrections.py    # Holm-Bonferroni and Benjamini-Hochberg FDR adjustments
├── generate_complete_submission_package.py # 600 DPI figure exporter and submission builder
├── requirements.txt                      # Environment package dependencies
└── README.md
```

---

## Environment and Requirements

### Prerequisites
- Python `>= 3.10`
- `scikit-learn >= 1.3.0`
- `fairlearn >= 0.9.0`
- `lightgbm >= 4.0.0`
- `xgboost >= 2.0.0`
- `pandas >= 2.0.0`, `numpy >= 1.24.0`, `scipy >= 1.11.0`

### Installation
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Reproduction Guide

### 1. Execute Core Empirical Experiments
Runs the complete experiment suite (Table 1, Table 3, DCA, and Scaled Search) on the canonical CDC BRFSS 2015 partition ($N_{\text{test}} = 50{,}736$):
```bash
python run_rigorous_empirical_suite.py
```

### 2. Paired Stratified Bootstrap Inference ($B = 2{,}000$) and DeLong Tests
Generates 95% non-parametric bootstrap confidence intervals and DeLong ROC comparison tests:
```bash
python run_standardized_bootstrap_protocol.py
```

### 3. Multiple Testing Adjustments
Computes Holm–Bonferroni and Benjamini–Hochberg FDR adjustments across all hypothesis tests:
```bash
python calculate_multitest_corrections.py
```

### 4. Sensitivity Sweep, Multi-Seed Stability, and Cross-Attribute Transfer
Evaluates $\lambda_{\text{fairness}} \in [0.1, 50]$, 5 independent random seeds, and educational attainment transfer:
```bash
python run_q1_empirical_campaign.py
```

### 5. Generate Publication Figures
Generates publication-quality figures (600 DPI TIFF, PDF, EPS, PNG) in `results/figures/`:
```bash
python generate_complete_submission_package.py
```

---

## Citation

```bibtex
@article{ekawijana2026architecture,
  title={When architecture is not a fairness remedy: a utility-matched evaluation of cluster-then-predict pipelines for diabetes risk prediction},
  author={Ekawijana, Ardhian and Kurniawan, Mochamad Teguh and Mubaroq, Sugeng Rifqi},
  journal={Informatics and Health},
  year={2026}
}
```
