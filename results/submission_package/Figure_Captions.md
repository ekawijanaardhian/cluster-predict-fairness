# Figure Captions

**Fig. 1.** High-level architectural workflow of the adaptive cluster-then-predict framework for fairness-aware clinical risk prediction across socio-economic strata.

**Fig. 2.** Search efficiency, computation time, and objective comparison between informed A* heuristic tree search and exhaustive brute-force grid search across standard ($N=18$) and scaled ($N=105$) candidate spaces on CDC BRFSS 2015 data: (a) evaluated pipeline configurations (3 vs 18 in standard space, and 3 vs 105 in scaled space); (b) wall-clock computation time (74.73 s vs 125.69 s, and 127.97 s vs 603.84 s); (c) optimization objective cost $f(n)$ comparing heuristic convergence against global enumeration (0.4702 vs 0.4688 in standard space, and 0.4510 vs 0.4418 in scaled space, with identical test-set discrimination $\text{AUC}=0.8263$ and disparity $\text{DPD}=0.2896$).

**Fig. 3.** Stage 1 latent cluster optimization curves across candidate sub-population counts $K \in [2, 10]$: (a) Davies–Bouldin index and composite cost function $f(n)$ showing global cost minimum at $K=2$; (b) Calinski–Harabasz variance ratio criterion exhibiting peak sub-population separation at $K=2$.

**Fig. 4.** Empirical Pareto frontier illustrating clinical utility (AUC-ROC) versus fairness disparity (Demographic Parity Difference, DPD) trade-off across single-model baselines (Logistic Regression, LightGBM, XGBoost), standard mitigation techniques (Reweighing, Threshold Optimizer), hard partitioning (HP), and hierarchical mixture-of-experts (HMoE) on CDC BRFSS 2015 test set ($N=50,736$).

**Fig. 5.** Sensitivity heatmap of predictive and fairness metrics across fairness regularization weights $\lambda_{\text{fairness}} \in [0.1, 50.0]$ in the pipeline objective formulation.

**Fig. 6.** Multi-seed statistical stability across 5 independent stratified random data splits ($N=253,680$ total, test $N=50,736$): (a) AUC-ROC utility distributions; (b) Demographic Parity Difference (DPD) distributions comparing unmitigated single LightGBM, hierarchical mixture-of-experts (HMoE), and hard partitioning (HP).

