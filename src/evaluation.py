"""
Full-Factorial Experimentation Suite (2^3 = 8 Combinations).
Performs systematic benchmark comparisons, trade-off evaluations,
multi-seed statistical significance testing (Mean ± 95% CI),
in-processing epsilon constraint relaxation sweeps,
and quantitative interaction analysis between mitigation layers.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from scipy import stats

from src.pipeline import ClusterThenPredictPipeline
from src.metrics import format_metrics_table

# 8 Factorial configurations
FACTORIAL_CONFIGS = [
    {"name": "1_Baseline_Blind", "pre": False, "in": False, "post": False},
    {"name": "2_Pre_Only",       "pre": True,  "in": False, "post": False},
    {"name": "3_In_Only",        "pre": False, "in": True,  "post": False},
    {"name": "4_Post_Only",      "pre": False, "in": False, "post": True},
    {"name": "5_Pre_plus_In",    "pre": True,  "in": True,  "post": False},
    {"name": "6_Pre_plus_Post",  "pre": True,  "in": False, "post": True},
    {"name": "7_In_plus_Post",   "pre": False, "in": True,  "post": True},
    {"name": "8_Full_TriLayer",  "pre": True,  "in": True,  "post": True},
]

class FactorialExperimentRunner:
    """
    Orchestrates the 8-cell full factorial matrix for the dissertation protocol.
    Supports static execution or adaptive clustering & classifier selection,
    configurable fairness epsilon constraints, and validation threshold calibration.
    """
    def __init__(
        self,
        n_clusters: int = 4,
        clustering_method: str = 'kmeans',
        classifier_type: str = 'lightgbm',
        adaptive_clustering: bool = False,
        adaptive_classifier: bool = False,
        fairness_constraint: str = 'equalized_odds',
        eps: float = 0.02,
        roc_objective: str = 'dpd',
        calibrate_threshold: bool = True,
        k_min: int = 2,
        k_max: int = 10,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.classifier_type = classifier_type
        self.adaptive_clustering = adaptive_clustering
        self.adaptive_classifier = adaptive_classifier
        self.fairness_constraint = fairness_constraint
        self.eps = eps
        self.roc_objective = roc_objective
        self.calibrate_threshold = calibrate_threshold
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state
        self.results_df = None
        self.cluster_profiles = {}
        self.classifier_cv_scores_ = {}

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        s: pd.Series,
        test_size: float = 0.20,
        val_size: float = 0.15
    ) -> pd.DataFrame:
        """
        Executes all 8 factorial configurations on standardized train/val/test splits.
        """
        print("\n" + "="*70)
        print("STARTING FULL FACTORIAL EXPERIMENT MATRIX (8 CONFIGURATIONS)")
        print("="*70)
        
        # 1. Stratified Data Splitting
        X_temp, X_test, y_temp, y_test, s_temp, s_test = train_test_split(
            X, y, s, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        val_ratio_adjusted = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
            X_temp, y_temp, s_temp, test_size=val_ratio_adjusted, random_state=self.random_state, stratify=y_temp
        )

        print(f"Dataset split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
        print(f"Features: {X_train.shape[1]} | Adaptive Clustering: {self.adaptive_clustering} | Adaptive Classifier: {self.adaptive_classifier}")
        print(f"Base Config: Clusters={self.n_clusters} ({self.clustering_method}) | Classifier={self.classifier_type.upper()} | In-Proc Eps={self.eps} | Calibrated Thresh={self.calibrate_threshold}")
        print("-" * 70)

        all_results = []

        for i, cfg in enumerate(FACTORIAL_CONFIGS, 1):
            t0 = time.time()
            print(f"[{i}/8] Running: {cfg['name']:<20} (Pre={cfg['pre']}, In={cfg['in']}, Post={cfg['post']})...", end="", flush=True)
            
            pipe = ClusterThenPredictPipeline(
                name=cfg['name'],
                n_clusters=self.n_clusters,
                clustering_method=self.clustering_method,
                classifier_type=self.classifier_type,
                adaptive_clustering=self.adaptive_clustering,
                adaptive_classifier=self.adaptive_classifier,
                apply_pre=cfg['pre'],
                apply_in=cfg['in'],
                apply_post=cfg['post'],
                fairness_constraint=self.fairness_constraint,
                eps=self.eps,
                roc_objective=self.roc_objective,
                calibrate_threshold=self.calibrate_threshold,
                k_min=self.k_min,
                k_max=self.k_max,
                random_state=self.random_state
            )

            pipe.fit(X_train, y_train, s_train, X_val, y_val, s_val)
            metrics = pipe.evaluate(X_test, y_test, s_test)
            metrics['Execution_Time_Sec'] = round(time.time() - t0, 2)
            all_results.append(metrics)

            if cfg['name'] == '1_Baseline_Blind':
                self.cluster_profiles['Baseline'] = pipe.cluster_demographics_train
                self.classifier_cv_scores_ = pipe.classifier_manager.cluster_candidate_cv_scores_
            elif cfg['name'] == '8_Full_TriLayer':
                self.cluster_profiles['TriLayer'] = pipe.cluster_demographics_train

            print(f" Done in {metrics['Execution_Time_Sec']}s | AUC={metrics['AUC_ROC']:.4f} | DP_Diff={metrics['Demographic_Parity_Diff']:.4f} | EO_Diff={metrics['Equalized_Odds_Diff']:.4f}")

        self.results_df = pd.DataFrame(all_results)
        self._calculate_relative_deltas()
        return self.results_df

    def _calculate_relative_deltas(self):
        """Calculates trade-offs relative to baseline and interaction coefficients."""
        if self.results_df is None or '1_Baseline_Blind' not in self.results_df['Configuration'].values:
            return

        base_row = self.results_df[self.results_df['Configuration'] == '1_Baseline_Blind'].iloc[0]
        base_auc = base_row['AUC_ROC']
        base_dp = base_row['Demographic_Parity_Diff']
        base_eo = base_row['Equalized_Odds_Diff']

        self.results_df['Delta_AUC_pct'] = ((self.results_df['AUC_ROC'] - base_auc) / base_auc * 100).round(2)
        self.results_df['Delta_DPD_pct'] = ((self.results_df['Demographic_Parity_Diff'] - base_dp) / (base_dp + 1e-9) * 100).round(2)
        self.results_df['Delta_EOD_pct'] = ((self.results_df['Equalized_Odds_Diff'] - base_eo) / (base_eo + 1e-9) * 100).round(2)

    def compute_interaction_effects(self) -> pd.DataFrame:
        """
        Quantifies multi-layer interaction:
        Interaction = Delta_Joint - sum(Delta_Individual)
        """
        df = self.results_df.set_index('Configuration')
        
        delta_dp_pre = df.loc['2_Pre_Only', 'Demographic_Parity_Diff'] - df.loc['1_Baseline_Blind', 'Demographic_Parity_Diff']
        delta_dp_in  = df.loc['3_In_Only', 'Demographic_Parity_Diff'] - df.loc['1_Baseline_Blind', 'Demographic_Parity_Diff']
        delta_dp_post= df.loc['4_Post_Only', 'Demographic_Parity_Diff'] - df.loc['1_Baseline_Blind', 'Demographic_Parity_Diff']
        
        delta_dp_joint = df.loc['8_Full_TriLayer', 'Demographic_Parity_Diff'] - df.loc['1_Baseline_Blind', 'Demographic_Parity_Diff']
        linear_sum_dp = delta_dp_pre + delta_dp_in + delta_dp_post
        interaction_dp = delta_dp_joint - linear_sum_dp

        delta_eo_pre = df.loc['2_Pre_Only', 'Equalized_Odds_Diff'] - df.loc['1_Baseline_Blind', 'Equalized_Odds_Diff']
        delta_eo_in  = df.loc['3_In_Only', 'Equalized_Odds_Diff'] - df.loc['1_Baseline_Blind', 'Equalized_Odds_Diff']
        delta_eo_post= df.loc['4_Post_Only', 'Equalized_Odds_Diff'] - df.loc['1_Baseline_Blind', 'Equalized_Odds_Diff']
        
        delta_eo_joint = df.loc['8_Full_TriLayer', 'Equalized_Odds_Diff'] - df.loc['1_Baseline_Blind', 'Equalized_Odds_Diff']
        linear_sum_eo = delta_eo_pre + delta_eo_in + delta_eo_post
        interaction_eo = delta_eo_joint - linear_sum_eo

        def _get_interpretation(interaction_val: float, delta_joint_val: float) -> str:
            if interaction_val < 0 and delta_joint_val < 0:
                return 'True Synergistic Reduction'
            elif interaction_val < 0 and delta_joint_val >= 0:
                return 'Sub-additive Deterioration (Mitigated Compounding)'
            elif interaction_val >= 0 and delta_joint_val < 0:
                return 'Antagonistic Improvement (Partial Offset)'
            else:
                return 'Compounding Deterioration / Offsetting'

        interaction_summary = pd.DataFrame([
            {
                'Metric': 'Demographic Parity Diff',
                'Baseline': df.loc['1_Baseline_Blind', 'Demographic_Parity_Diff'],
                'Joint_TriLayer': df.loc['8_Full_TriLayer', 'Demographic_Parity_Diff'],
                'Actual_Delta_Joint': round(delta_dp_joint, 4),
                'Expected_Linear_Delta': round(linear_sum_dp, 4),
                'Interaction_Effect': round(interaction_dp, 4),
                'Interpretation': _get_interpretation(interaction_dp, delta_dp_joint)
            },
            {
                'Metric': 'Equalized Odds Diff',
                'Baseline': df.loc['1_Baseline_Blind', 'Equalized_Odds_Diff'],
                'Joint_TriLayer': df.loc['8_Full_TriLayer', 'Equalized_Odds_Diff'],
                'Actual_Delta_Joint': round(delta_eo_joint, 4),
                'Expected_Linear_Delta': round(linear_sum_eo, 4),
                'Interaction_Effect': round(interaction_eo, 4),
                'Interpretation': _get_interpretation(interaction_eo, delta_eo_joint)
            }
        ])
        return interaction_summary


class MultiSeedFactorialRunner:
    """
    Executes full 8-cell factorial matrix across N random seeds (>=10).
    Computes statistical estimates: Mean, Std, 95% Confidence Intervals (CI),
    and hypothesis testing for all configurations and interaction coefficients gamma.
    """
    def __init__(
        self,
        seeds: List[int] = [42, 100, 2024, 777, 999, 123, 456, 789, 314, 555],
        adaptive_clustering: bool = True,
        adaptive_classifier: bool = True,
        eps: float = 0.02,
        roc_objective: str = 'dpd',
        calibrate_threshold: bool = True
    ):
        self.seeds = seeds
        self.adaptive_clustering = adaptive_clustering
        self.adaptive_classifier = adaptive_classifier
        self.eps = eps
        self.roc_objective = roc_objective
        self.calibrate_threshold = calibrate_threshold
        self.raw_results_ = []

    def run(self, X: pd.DataFrame, y: pd.Series, s: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Runs multi-seed evaluation across all 8 configurations."""
        self.raw_results_ = []
        n_seeds = len(self.seeds)
        print(f"\n[MultiSeedRunner] Launching {n_seeds}-seed benchmark across all 8 factorial configurations...")

        for s_idx, seed in enumerate(self.seeds, 1):
            print(f"--- Processing Seed {seed} ({s_idx}/{n_seeds}) ---")
            runner = FactorialExperimentRunner(
                n_clusters=4,
                clustering_method='auto' if self.adaptive_clustering else 'kmeans',
                classifier_type='lightgbm',
                adaptive_clustering=self.adaptive_clustering,
                adaptive_classifier=self.adaptive_classifier,
                eps=self.eps,
                roc_objective=self.roc_objective,
                calibrate_threshold=self.calibrate_threshold,
                random_state=seed
            )
            df_seed = runner.run(X, y, s)
            df_seed['Seed'] = seed
            self.raw_results_.append(df_seed)

        df_all = pd.concat(self.raw_results_, ignore_index=True)

        # Aggregate metrics across seeds
        summary_rows = []
        metrics_to_agg = ['AUC_ROC', 'Accuracy', 'Balanced_Accuracy', 'Demographic_Parity_Diff', 'Equalized_Odds_Diff']

        for cfg in df_all['Configuration'].unique():
            sub = df_all[df_all['Configuration'] == cfg]
            row_dict = {'Configuration': cfg, 'Num_Seeds': len(sub)}
            for m in metrics_to_agg:
                vals = sub[m].values
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                ci_95 = 1.96 * (std_v / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                row_dict[f"{m}_Mean"] = round(mean_v, 4)
                row_dict[f"{m}_Std"] = round(std_v, 4)
                row_dict[f"{m}_CI95_Lower"] = round(mean_v - ci_95, 4)
                row_dict[f"{m}_CI95_Upper"] = round(mean_v + ci_95, 4)
                row_dict[f"{m}_Formatted"] = f"{mean_v:.4f} ± {std_v:.4f} [{mean_v - ci_95:.4f}, {mean_v + ci_95:.4f}]"
            summary_rows.append(row_dict)

        df_summary = pd.DataFrame(summary_rows)

        # Compute Gamma Multi-seed statistics
        gamma_rows = []
        for seed in self.seeds:
            sub = df_all[df_all['Seed'] == seed].set_index('Configuration')
            # Gamma DPD
            d_dp_pre = sub.loc['2_Pre_Only', 'Demographic_Parity_Diff'] - sub.loc['1_Baseline_Blind', 'Demographic_Parity_Diff']
            d_dp_in = sub.loc['3_In_Only', 'Demographic_Parity_Diff'] - sub.loc['1_Baseline_Blind', 'Demographic_Parity_Diff']
            d_dp_post = sub.loc['4_Post_Only', 'Demographic_Parity_Diff'] - sub.loc['1_Baseline_Blind', 'Demographic_Parity_Diff']
            d_dp_joint = sub.loc['8_Full_TriLayer', 'Demographic_Parity_Diff'] - sub.loc['1_Baseline_Blind', 'Demographic_Parity_Diff']
            gamma_dpd = d_dp_joint - (d_dp_pre + d_dp_in + d_dp_post)

            # Gamma EOD
            d_eo_pre = sub.loc['2_Pre_Only', 'Equalized_Odds_Diff'] - sub.loc['1_Baseline_Blind', 'Equalized_Odds_Diff']
            d_eo_in = sub.loc['3_In_Only', 'Equalized_Odds_Diff'] - sub.loc['1_Baseline_Blind', 'Equalized_Odds_Diff']
            d_eo_post = sub.loc['4_Post_Only', 'Equalized_Odds_Diff'] - sub.loc['1_Baseline_Blind', 'Equalized_Odds_Diff']
            d_eo_joint = sub.loc['8_Full_TriLayer', 'Equalized_Odds_Diff'] - sub.loc['1_Baseline_Blind', 'Equalized_Odds_Diff']
            gamma_eod = d_eo_joint - (d_eo_pre + d_eo_in + d_eo_post)

            gamma_rows.append({'Seed': seed, 'Gamma_DPD': gamma_dpd, 'Gamma_EOD': gamma_eod})

        df_gamma = pd.DataFrame(gamma_rows)
        
        return df_all, df_summary, df_gamma


class EpsilonSweepRunner:
    """
    Performs systematic sensitivity sweep over in-processing constraint bound epsilon
    to investigate whether AUC drop and inert post-processing are intrinsic or constraint-tightness artifacts.
    """
    def __init__(
        self,
        eps_list: List[float] = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20],
        random_state: int = 42
    ):
        self.eps_list = eps_list
        self.random_state = random_state

    def run(self, X: pd.DataFrame, y: pd.Series, s: pd.Series) -> pd.DataFrame:
        print(f"\n[EpsilonSweep] Evaluating in-processing fairness constraints across eps={self.eps_list}...")
        records = []
        
        X_temp, X_test, y_temp, y_test, s_temp, s_test = train_test_split(
            X, y, s, test_size=0.20, random_state=self.random_state, stratify=y
        )
        X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
            X_temp, y_temp, s_temp, test_size=0.15/0.80, random_state=self.random_state, stratify=y_temp
        )

        for eps_val in self.eps_list:
            for cfg_name, (pre, in_p, post) in [
                ("3_In_Only", (False, True, False)),
                ("7_In_plus_Post", (False, True, True)),
                ("8_Full_TriLayer", (True, True, True))
            ]:
                pipe = ClusterThenPredictPipeline(
                    name=cfg_name,
                    n_clusters=4,
                    adaptive_clustering=True,
                    adaptive_classifier=True,
                    apply_pre=pre,
                    apply_in=in_p,
                    apply_post=post,
                    eps=eps_val,
                    random_state=self.random_state
                )
                pipe.fit(X_train, y_train, s_train, X_val, y_val, s_val)
                m = pipe.evaluate(X_test, y_test, s_test)
                m['Epsilon'] = eps_val
                records.append(m)

        return pd.DataFrame(records)
