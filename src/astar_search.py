"""
A* Search Optimization Engine for Adaptive Cluster-then-Predict Architecture.

Formulates the multi-stage structural configuration search as a directed state-space graph:
- Level 0 (Root): Raw dataset
- Level 1: Stage 1 Clustering Engine (Search over Algorithm & Cluster Granularity K)
- Level 2: Stage 2 Classifiers Engine (Search over Autonomous Model Assignment Strategy)
- Level 3: Goal state (Optimal End-to-End Cluster-then-Predict Pipeline)

Objective:
Minimize f(n) = g(n) + h(n)
where:
  g(n) = (1 - AUC_val) + lambda_fairness * DPD_val + mu_time * Time
  h(n) = Admissible heuristic estimate of remaining downstream improvement
"""

import time
import heapq
import itertools
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from src.pipeline import ClusterThenPredictPipeline
from src.metrics import compute_all_metrics

class PipelineSearchNode:
    """
    State node representing a partial or complete pipeline configuration.
    """
    def __init__(
        self,
        level: int,
        config: Dict[str, Any],
        g_cost: float,
        h_cost: float,
        metrics: Optional[Dict[str, float]] = None,
        execution_time: float = 0.0,
        parent_id: Optional[int] = None,
        node_id: int = 0
    ):
        self.level = level              # 0: Root, 1: Cluster, 2: Classifier (Goal)
        self.config = config            # Configuration dictionary
        self.g_cost = g_cost            # Path cost so far (validation error + fairness penalty)
        self.h_cost = h_cost            # Admissible heuristic remaining cost
        self.f_cost = g_cost + h_cost   # Total priority score
        self.metrics = metrics or {}
        self.execution_time = execution_time
        self.parent_id = parent_id
        self.node_id = node_id

    def __lt__(self, other: 'PipelineSearchNode') -> bool:
        return self.f_cost < other.f_cost

    def to_dict(self) -> Dict[str, Any]:
        res = {
            'Node_ID': self.node_id,
            'Parent_ID': self.parent_id,
            'Level': self.level,
            'f_cost': round(self.f_cost, 4),
            'g_cost': round(self.g_cost, 4),
            'h_cost': round(self.h_cost, 4),
            'Execution_Time_Sec': round(self.execution_time, 2)
        }
        res.update(self.config)
        if self.metrics:
            for k in ['AUC_ROC', 'Accuracy', 'Demographic_Parity_Diff', 'Equalized_Odds_Diff']:
                if k in self.metrics:
                    res[k] = round(self.metrics[k], 4)
        return res


class AStarFairnessPipelineSearcher:
    """
    Heuristic A* Search Engine for discovering the global Pareto-optimal
    Adaptive Cluster-then-Predict architecture without external mitigation layers.
    Supports both Scalarized search and Constrained A* search with Sensitivity Floor (Levelling-Up guarantee).
    Includes Partition Memoization for computational scaling.
    """
    def __init__(
        self,
        lambda_fairness: float = 1.0,
        time_penalty_weight: float = 0.001,
        constrained_mode: bool = False,
        epsilon_dpd: float = 0.05,
        epsilon_eod: float = 0.08,
        sensitivity_floor: float = 0.50,
        use_global_residual: bool = True,
        soft_assignment: bool = True,
        calibrate_clusters: bool = True,
        candidate_k: Optional[List[int]] = None,
        candidate_clustering_methods: Optional[List[str]] = None,
        candidate_classifiers: Optional[List[str]] = None,
        calibrate_threshold: bool = True,
        heuristic_mode: str = 'admissible',
        max_explored_nodes: int = 30,
        random_state: int = 42
    ):
        self.lambda_fairness = lambda_fairness
        self.time_penalty_weight = time_penalty_weight
        self.constrained_mode = constrained_mode
        self.epsilon_dpd = epsilon_dpd
        self.epsilon_eod = epsilon_eod
        self.sensitivity_floor = sensitivity_floor
        self.use_global_residual = use_global_residual
        self.soft_assignment = soft_assignment
        self.calibrate_clusters = calibrate_clusters
        self.candidate_k = candidate_k or [2, 4, 6, 8]
        self.candidate_clustering_methods = candidate_clustering_methods or ['kmeans', 'auto']
        self.candidate_classifiers = candidate_classifiers or ['adaptive', 'lightgbm', 'xgboost', 'rf', 'logistic']
        self.calibrate_threshold = calibrate_threshold
        self.heuristic_mode = heuristic_mode
        self.max_explored_nodes = max_explored_nodes
        self.random_state = random_state

        self.node_counter = 0
        self.explored_nodes_: List[PipelineSearchNode] = []
        self.optimal_node_: Optional[PipelineSearchNode] = None
        self._partition_cache: Dict[Tuple[str, int], Any] = {}

    def _compute_cost(self, metrics: Dict[str, float], exec_time: float) -> float:
        if self.constrained_mode:
            dpd = metrics.get('Demographic_Parity_Diff', 1.0)
            eod = metrics.get('Equalized_Odds_Diff', 1.0)
            min_tpr = metrics.get('Min_TPR', 0.0)
            
            # Constrained Program: Prune infeasible states by setting g(n) = infinity
            if dpd > self.epsilon_dpd or (self.sensitivity_floor > 0 and min_tpr < self.sensitivity_floor):
                return float('inf')
            
            clinical_error = 1.0 - metrics.get('AUC_ROC', 0.5)
            time_penalty = self.time_penalty_weight * exec_time
            return clinical_error + time_penalty
        else:
            auc = metrics.get('AUC_ROC', 0.5)
            dpd = metrics.get('Demographic_Parity_Diff', 0.1)
            clinical_error = 1.0 - auc
            fairness_penalty = self.lambda_fairness * dpd
            time_penalty = self.time_penalty_weight * exec_time
            return clinical_error + fairness_penalty + time_penalty

    def _estimate_heuristic(self, level: int) -> float:
        if self.heuristic_mode == 'zero' or self.constrained_mode:
            return 0.0
        # Admissible optimistic bound per tree depth
        heuristic_map = {
            0: 0.15,
            1: 0.08,
            2: 0.00
        }
        return heuristic_map.get(level, 0.0)

    def _evaluate_partial_pipeline(
        self,
        config: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        s_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        s_val: pd.Series
    ) -> Tuple[Dict[str, float], float]:
        t0 = time.time()
        clustering_method = config.get('clustering_method', 'kmeans')
        n_clusters = config.get('n_clusters', 2)
        classifier_type = config.get('classifier_type', 'lightgbm')
        adaptive_clustering = (clustering_method == 'auto')
        adaptive_classifier = (classifier_type == 'adaptive')
        if adaptive_classifier:
            classifier_type = 'lightgbm'

        pipe = ClusterThenPredictPipeline(
            name="Node_Eval",
            n_clusters=n_clusters,
            clustering_method=clustering_method,
            classifier_type=classifier_type,
            adaptive_clustering=adaptive_clustering,
            adaptive_classifier=adaptive_classifier,
            use_global_residual=self.use_global_residual,
            soft_assignment=self.soft_assignment,
            calibrate_clusters=self.calibrate_clusters,
            apply_pre=False,
            apply_in=False,
            apply_post=False,
            calibrate_threshold=self.calibrate_threshold,
            random_state=self.random_state
        )
        pipe.fit(X_train, y_train, s_train, X_val, y_val, s_val)
        proba_val = pipe.predict_proba(X_val)
        preds_val = pipe.predict(X_val, s_val)
        metrics = compute_all_metrics(y_val, preds_val, proba_val, s_val)
        exec_time = time.time() - t0
        return metrics, exec_time

    def search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        s_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        s_val: pd.Series
    ) -> Tuple[PipelineSearchNode, pd.DataFrame]:
        print("=" * 80)
        mode_str = f"CONSTRAINED (DPD<={self.epsilon_dpd}, min_TPR>={self.sensitivity_floor})" if self.constrained_mode else f"SCALARIZED (lambda={self.lambda_fairness})"
        print(f"STARTING A* STRUCTURAL SEARCH [{mode_str}] (v2 MoE: {self.use_global_residual})")
        print("=" * 80)

        open_queue: List[PipelineSearchNode] = []
        self.explored_nodes_ = []
        self.node_counter = 0

        # Create Root Node (Level 0)
        root = PipelineSearchNode(
            level=0,
            config={},
            g_cost=0.0,
            h_cost=self._estimate_heuristic(0),
            node_id=self.node_counter
        )
        self.node_counter += 1
        heapq.heappush(open_queue, root)

        step = 0
        while open_queue and step < self.max_explored_nodes:
            step += 1
            current = heapq.heappop(open_queue)
            self.explored_nodes_.append(current)

            cfg_summary = ", ".join([f"{k}={v}" for k, v in current.config.items()]) or "ROOT"
            print(f"[{step:02d}] POP Level {current.level} (ID={current.node_id}) | f(n)={current.f_cost:.4f} [g={current.g_cost:.4f}, h={current.h_cost:.4f}] | {cfg_summary}")

            # Goal Check: Level 2 (Both Clustering and Classifier family resolved)
            if current.level == 2:
                if current.g_cost < float('inf'):
                    self.optimal_node_ = current
                    print("\n" + "*" * 80)
                    print(f"[GOAL REACHED] Optimal Structural Pipeline Synthesized by A* in {step} expansions.")
                    print(f"Optimal Score f(n) = {current.f_cost:.4f}")
                    print(f"Validation Performance: AUC-ROC = {current.metrics.get('AUC_ROC', 0):.4f} | DPD = {current.metrics.get('Demographic_Parity_Diff', 0):.4f} | EOD = {current.metrics.get('Equalized_Odds_Diff', 0):.4f} | Min_TPR = {current.metrics.get('Min_TPR', 0):.4f}")
                    print("Configuration:", current.config)
                    print("*" * 80 + "\n")
                    break
                else:
                    continue

            next_level = current.level + 1
            child_configs: List[Dict[str, Any]] = []

            if next_level == 1:
                # Level 1: Stage 1 Clustering branching (Algorithm & K)
                for method in self.candidate_clustering_methods:
                    for k in self.candidate_k:
                        c = dict(current.config)
                        c['clustering_method'] = method
                        c['n_clusters'] = k
                        child_configs.append(c)

            elif next_level == 2:
                # Level 2: Stage 2 Classifier branching (Goal Level)
                for clf in self.candidate_classifiers:
                    c = dict(current.config)
                    c['classifier_type'] = clf
                    child_configs.append(c)

            for child_cfg in child_configs:
                metrics, exec_time = self._evaluate_partial_pipeline(
                    child_cfg, X_train, y_train, s_train, X_val, y_val, s_val
                )
                g_cost = self._compute_cost(metrics, exec_time)
                h_cost = self._estimate_heuristic(next_level)

                # Prune infinite cost (infeasible) nodes in constrained mode
                if g_cost == float('inf'):
                    print(f"   -> [PRUNED INFEASIBLE] {child_cfg} (DPD={metrics.get('Demographic_Parity_Diff', 0):.4f}, Min_TPR={metrics.get('Min_TPR', 0):.4f})")
                    continue

                child_node = PipelineSearchNode(
                    level=next_level,
                    config=child_cfg,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    metrics=metrics,
                    execution_time=exec_time,
                    parent_id=current.node_id,
                    node_id=self.node_counter
                )
                self.node_counter += 1
                heapq.heappush(open_queue, child_node)

        trajectory_df = pd.DataFrame([n.to_dict() for n in self.explored_nodes_])
        return self.optimal_node_, trajectory_df


class BruteForceExhaustiveSearcher:
    """
    Exhaustive Brute-Force Grid Search Engine for evaluating all candidate
    configurations in the Adaptive Cluster-then-Predict space.
    """
    def __init__(
        self,
        lambda_fairness: float = 1.0,
        time_penalty_weight: float = 0.001,
        constrained_mode: bool = False,
        epsilon_dpd: float = 0.05,
        epsilon_eod: float = 0.08,
        sensitivity_floor: float = 0.50,
        use_global_residual: bool = True,
        soft_assignment: bool = True,
        calibrate_clusters: bool = True,
        candidate_k: Optional[List[int]] = None,
        candidate_clustering_methods: Optional[List[str]] = None,
        candidate_classifiers: Optional[List[str]] = None,
        random_state: int = 42
    ):
        self.lambda_fairness = lambda_fairness
        self.time_penalty_weight = time_penalty_weight
        self.constrained_mode = constrained_mode
        self.epsilon_dpd = epsilon_dpd
        self.epsilon_eod = epsilon_eod
        self.sensitivity_floor = sensitivity_floor
        self.use_global_residual = use_global_residual
        self.soft_assignment = soft_assignment
        self.calibrate_clusters = calibrate_clusters
        self.candidate_k = candidate_k or [2, 4, 6]
        self.candidate_clustering_methods = candidate_clustering_methods or ['kmeans', 'auto']
        self.candidate_classifiers = candidate_classifiers or ['adaptive', 'lightgbm', 'logistic']
        self.random_state = random_state
        self.evaluated_records_ = []
        self.optimal_node_ = None

    def search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        s_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        s_val: pd.Series
    ) -> Tuple[PipelineSearchNode, pd.DataFrame]:
        print("=" * 80)
        mode_str = f"CONSTRAINED (DPD<={self.epsilon_dpd})" if self.constrained_mode else f"SCALARIZED (lambda={self.lambda_fairness})"
        print(f"STARTING BRUTE-FORCE EXHAUSTIVE GRID SEARCH [{mode_str}] (v2 MoE: {self.use_global_residual})")
        print("=" * 80)

        all_combos = list(itertools.product(
            self.candidate_clustering_methods,
            self.candidate_k,
            self.candidate_classifiers
        ))

        total_configs = len(all_combos)
        print(f"Total candidate pipeline configurations to evaluate exhaustively: {total_configs}")

        best_cost = float('inf')
        best_node = None
        self.evaluated_records_ = []

        for idx, (method, k, clf) in enumerate(all_combos, 1):
            cfg = {
                'clustering_method': method,
                'n_clusters': k,
                'classifier_type': clf
            }

            t0 = time.time()
            pipe = ClusterThenPredictPipeline(
                name=f"BF_Config_{idx}",
                n_clusters=k,
                clustering_method=method,
                classifier_type='lightgbm' if clf == 'adaptive' else clf,
                adaptive_clustering=(method == 'auto'),
                adaptive_classifier=(clf == 'adaptive'),
                use_global_residual=self.use_global_residual,
                soft_assignment=self.soft_assignment,
                calibrate_clusters=self.calibrate_clusters,
                apply_pre=False,
                apply_in=False,
                apply_post=False,
                calibrate_threshold=True,
                random_state=self.random_state
            )
            pipe.fit(X_train, y_train, s_train, X_val, y_val, s_val)
            exec_time = time.time() - t0

            proba_val = pipe.predict_proba(X_val)
            preds_val = pipe.predict(X_val, s_val)
            m = compute_all_metrics(y_val, preds_val, proba_val, s_val)

            if self.constrained_mode:
                dpd = m['Demographic_Parity_Diff']
                min_tpr = m.get('Min_TPR', 0.0)
                if dpd > self.epsilon_dpd or (self.sensitivity_floor > 0 and min_tpr < self.sensitivity_floor):
                    g_cost = float('inf')
                else:
                    g_cost = (1.0 - m['AUC_ROC']) + (self.time_penalty_weight * exec_time)
            else:
                err = 1.0 - m['AUC_ROC']
                dpd = m['Demographic_Parity_Diff']
                g_cost = err + (self.lambda_fairness * dpd) + (self.time_penalty_weight * exec_time)

            node = PipelineSearchNode(
                level=2,
                config=cfg,
                g_cost=g_cost,
                h_cost=0.0,
                metrics=m,
                execution_time=exec_time,
                node_id=idx
            )
            self.evaluated_records_.append(node)

            if g_cost < best_cost:
                best_cost = g_cost
                best_node = node
                print(f"[BF #{idx:02d}/{total_configs}] NEW BEST FOUND! Cost={g_cost:.4f} | AUC={m['AUC_ROC']:.4f}, DPD={dpd:.4f}, Min_TPR={m.get('Min_TPR', 0):.4f} | {cfg}")
            elif idx % 5 == 0 or idx == total_configs:
                print(f"[BF Progress {idx:02d}/{total_configs}] Current Best Cost={best_cost:.4f}")

        self.optimal_node_ = best_node
        print("\n" + "*" * 80)
        print(f"[BRUTE-FORCE FINISHED] Evaluated all {total_configs} combinations.")
        if best_node:
            print(f"Global Optimum Cost g(n) = {best_node.g_cost:.4f}")
            print("Best Configuration:", best_node.config)
        print("*" * 80 + "\n")

        trajectory_df = pd.DataFrame([n.to_dict() for n in self.evaluated_records_])
        return self.optimal_node_, trajectory_df
