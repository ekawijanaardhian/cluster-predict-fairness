# When architecture is not a fairness remedy: a utility-matched evaluation of cluster-then-predict pipelines for diabetes risk prediction

Replication repository for:
> **When architecture is not a fairness remedy: a utility-matched evaluation of cluster-then-predict pipelines for diabetes risk prediction**  
> *Ardhian Ekawijana, Mochamad Teguh Kurniawan, Sugeng Rifqi Mubaroq* (2026)

---

## Overview

This repository provides code and datasets to replicate empirical evaluations of cluster-then-predict architectures against fairness mitigation baselines on the CDC BRFSS 2015 dataset (N = 253,680).

Key components:
- Stage 1: Clustering (K-Means, MiniBatchKMeans, GMM, Adaptive search)
- Stage 2: Downstream classification (LightGBM, XGBoost, Random Forest, Logistic Regression, Hierarchical MoE)
- Fairness baselines: Pre-processing (Reweighing), In-processing (Exponentiated Gradient), Post-processing (Reject-Option / Threshold Optimization)
- Optimization: A* heuristic search over architectural configurations

---

## Directory Structure

```
.
├── data/
│   ├── diabetes_binary_health_indicators_BRFSS2015.csv
│   ├── codebook15_llcp.pdf
│   └── loader.py
├── src/
│   ├── astar_search.py
│   ├── classifiers.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── postprocessing.py
│   └── preprocessing.py
├── results/
│   ├── figures/
│   └── *.csv
├── calculate_multitest_corrections.py
├── generate_q1_publication_figures.py
├── run_astar_vs_bruteforce.py
├── run_q1_empirical_campaign.py
├── run_rigorous_empirical_suite.py
├── run_scaled_astar_vs_bruteforce.py
├── run_standardized_bootstrap_protocol.py
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running Experiments

### 1. Empirical Suite
```bash
python run_rigorous_empirical_suite.py
```

### 2. Bootstrap Statistical Inference
Runs the standardized $B=2,000$ stratified paired bootstrap protocol (fixed random seed = 42 for exact replication), generating `results/bootstrap_estimates.csv` and `results/bootstrap_deltas.csv`:
```bash
python run_standardized_bootstrap_protocol.py
```

### 3. Multiple Testing Corrections
Applies Holm step-down and Benjamini-Hochberg FDR adjustments to the empirical bootstrap and DeLong test p-values:
```bash
python calculate_multitest_corrections.py
```

### 4. Parameter Sweeps and Stability
```bash
python run_q1_empirical_campaign.py
```

### 5. Figures
```bash
python generate_q1_publication_figures.py
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
