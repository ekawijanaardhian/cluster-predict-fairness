# Supplementary Material
**Title:** Evaluating Demographic Disparities in Clinical Risk Prediction: A Structural Fairness and Mixture-of-Experts Investigation
**Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS 2015, $N = 253,680$)

---

## Section S1: Sensitivity Analysis over Fairness Regularization Parameter ($\lambda_{\text{fairness}}$)
To evaluate the stability of the informed A* heuristic search across varying trade-off priorities between clinical predictive discrimination and demographic parity, we conducted a systematic parameter sweep over $\lambda_{\text{fairness}} \in \{0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$ on the full CDC BRFSS 2015 cohort ($N=253,680$).

**Table S1. Sensitivity of A* Pipeline Optimization across Fairness Weights $\lambda_{\text{fairness}}$**

| $\lambda_{\text{fairness}}$ | Expansions | Time (s) | Goal Cost $f(n)$ | Clustering | Selected $K$ | Classifier | Test AUC | Test Accuracy (%) | Test DPD | Test DPR | Test EOD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | 3 | 53.32 | 0.2058 | KMeans | 2 | Adaptive | 0.8261 | 72.54 | 0.2957 | 0.4938 | 0.2622 |
| 0.5 | 3 | 51.69 | 0.3243 | KMeans | 2 | Adaptive | 0.8261 | 72.54 | 0.2957 | 0.4938 | 0.2622 |
| 1.0 | 3 | 75.24 | 0.4702 | Auto | 6 | LightGBM | 0.8263 | 73.84 | 0.2896 | 0.4834 | 0.2531 |
| 2.0 | 3 | 74.89 | 0.7560 | Auto | 4 | Adaptive | 0.8263 | 73.84 | 0.2896 | 0.4834 | 0.2531 |
| 5.0 | 3 | 76.32 | 1.6135 | Auto | 2 | Adaptive | 0.8263 | 73.84 | 0.2896 | 0.4834 | 0.2531 |
| 10.0 | 3 | 78.06 | 3.0427 | Auto | 4 | Adaptive | 0.8263 | 73.84 | 0.2896 | 0.4834 | 0.2531 |
| 20.0 | 3 | 76.66 | 5.9005 | Auto | 2 | LightGBM | 0.8263 | 73.84 | 0.2896 | 0.4834 | 0.2531 |
| 50.0 | 3 | 77.78 | 14.4750 | Auto | 2 | LightGBM | 0.8263 | 73.84 | 0.2896 | 0.4834 | 0.2531 |

*Observation:* At $\lambda_{\text{fairness}} \le 0.5$, the A* search selects a 2-cluster partitioning ($K=2$ KMeans) yielding $\text{AUC} = 0.8261$ and $\text{DPD} = 0.2957$. For all $\lambda_{\text{fairness}} \ge 1.0$, the optimizer converges to richer sub-population partitions ($K \in [2, 6]$) maintaining stable predictive performance ($\text{AUC} = 0.8263, \text{DPD} = 0.2896$), with total goal cost $f(n)$ scaling linearly with $\lambda \times \text{DPD}$.

---

## Section S2: Multi-Seed Stability and Variance Decomposition (5 Independent Splits)
To ensure that empirical findings are not split artifacts, 5 independent randomized train/validation/test partitions ($65\%/15\%/20\%$, stratified on disease target $Y$) were evaluated across all benchmark architectures.

**Table S2. Multi-Seed Stability across 5 Independent Random Seeds**

| Architecture | Test AUC (Mean $\pm$ SD) | Test Accuracy (%) | Test Balanced Acc (%) | Test DPD | Test EOD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Single Logistic Regression ($K=1$) | $0.8197 \pm 0.0004$ | $70.98 \pm 0.12$ | $74.79 \pm 0.10$ | $0.3063 \pm 0.0006$ | $0.2768 \pm 0.0010$ |
| Single LightGBM ($K=1$, Unmitigated) | $0.8263 \pm 0.0004$ | $71.94 \pm 0.15$ | $75.19 \pm 0.11$ | $0.2909 \pm 0.0009$ | $0.2576 \pm 0.0012$ |
| Single XGBoost ($K=1$) | $0.8262 \pm 0.0004$ | $73.41 \pm 0.10$ | $75.05 \pm 0.11$ | $0.2881 \pm 0.0010$ | $0.2511 \pm 0.0013$ |
| Pre-Processing Reweighing | $0.8162 \pm 0.0005$ | $68.40 \pm 0.14$ | $74.28 \pm 0.12$ | $0.1234 \pm 0.0012$ | $0.0850 \pm 0.0015$ |
| Hard Partitioning (HP, $K=2$) | $0.7022 \pm 0.0023$ | $52.02 \pm 0.28$ | $65.47 \pm 0.25$ | $0.0324 \pm 0.0125$ | $0.0546 \pm 0.0142$ |
| Hierarchical Mixture-of-Experts (HMoE, $K=2$) | $0.8261 \pm 0.0003$ | $72.54 \pm 0.11$ | $75.18 \pm 0.11$ | $0.2957 \pm 0.0009$ | $0.2622 \pm 0.0011$ |
| Adaptive A* Synthesized Pipeline ($\lambda=1$) | $0.8273 \pm 0.0018$ | $71.93 \pm 2.01$ | $75.04 \pm 0.11$ | $0.2836 \pm 0.0059$ | $0.2514 \pm 0.0068$ |

---

## Section S3: In-Processing Exponentiated Gradient Baselines (ExpGrad-DP & ExpGrad-EO)
In addition to the Demographic Parity reduction model (ExpGrad-DP), we evaluated the Exponentiated Gradient reduction algorithm enforcing an Equalized Odds constraint (Agarwal et al., 2018) on the identical standardized test split.

**Table S3. In-Processing Mitigation Benchmark on Standardized BRFSS Split**

| In-Processing Formulation | Constraint Type | Bound $\epsilon$ | Test AUC | Test Accuracy (%) | Test Balanced Acc (%) | Test DPD | Test DPR | Test EOD | TPR Advantaged | TPR Disadvantaged | Requires Protected At Inference |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ExpGrad-DP | Demographic Parity | 0.02 | 0.5576 | 86.48 | 55.75 | 0.0217 | 0.5601 | 0.0118 | 0.1328 | 0.1290 | No |
| ExpGrad-EO | Equalized Odds | 0.02 | 0.5616 | 86.51 | 56.16 | 0.0300 | 0.4830 | 0.0204 | 0.1333 | 0.1537 | No |

*Finding:* Both in-processing reductions achieve low statistical disparity ($\text{DPD} \le 0.0300$, $\text{EOD} \le 0.0204$), but suffer severe degradation in global ranking discrimination ($\text{AUC} \le 0.5616$) and suppress true positive detection rates down to $\approx 13\% - 15\%$.

---

## Section S4: Standardized Statistical Inference Protocol ($B = 2,000$ Stratified Paired Bootstrap)
To evaluate the statistical significance of predictive discrimination and disparity differences without making parametric distribution assumptions, we implemented a paired stratified bootstrap resampling protocol:

1. **Stratified Sampling:** Resampling indices $idx_b$ ($b = 1, \dots, 2000$) were generated by sampling with replacement within each of the 4 mutually exclusive strata $(Y, S) \in \{(0,0), (0,1), (1,0), (1,1)\}$.
2. **Paired Replicates:** The identical resample index array $idx_b$ was simultaneously applied across all candidate architectures.
3. **Point Estimates:** Reported strictly as the empirical plug-in estimate on the complete test set ($N_{\text{test}} = 50,736$).
4. **Confidence Intervals & P-values:** 95% Confidence Intervals were constructed from the 2.5th and 97.5th percentiles of the replicate distribution. Two-tailed empirical $p$-values were computed as twice the proportion of replicates with sign opposite to the plug-in difference.
