"""
Stage 2: Downstream Classifiers and In-Processing Bias Mitigation.
Supports standard estimators (LightGBM, XGBoost, Logistic Regression, Random Forest),
fairness-constrained loss optimization via Fairlearn reductions (Agarwal et al., 2018),
and Adaptive Classifier Selection per cluster partition.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import lightgbm as lgb
import xgboost as xgb

from fairlearn.reductions import (
    ExponentiatedGradient,
    GridSearch,
    EqualizedOdds,
    DemographicParity,
    ErrorRate
)

def get_base_estimator(
    model_type: str = 'lightgbm', 
    imbalance_ratio: float = 1.0,
    random_state: int = 42
):
    """Factory for standard machine learning classifiers with adaptive imbalance awareness."""
    model_type = model_type.lower()
    
    # Adaptive scale_pos_weight / class_weight calculation
    scale_pos = max(1.0, (1.0 - imbalance_ratio) / (imbalance_ratio + 1e-6)) if imbalance_ratio < 0.2 else 1.0
    
    if model_type == 'lightgbm':
        return lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=scale_pos if scale_pos > 1.5 else 1.0,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
    elif model_type == 'xgboost':
        return xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=scale_pos if scale_pos > 1.5 else 1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
    elif model_type == 'logistic':
        return LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced' if imbalance_ratio < 0.25 else None
        )
    elif model_type == 'rf':
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced' if imbalance_ratio < 0.25 else None,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

class GlobalResidualClassifier:
    """
    Hierarchical Fair Mixture-of-Experts Classifier (v2 Architecture).
    
    Architecture:
    1. Global Base Estimator (p_global): Trained on all training data to establish
       global monotonic risk discrimination (preserving full AUC capacity >= 0.80+).
    2. Global Logit Feature: z_global = logit(clip(p_global(x), 1e-5, 1 - 1e-5))
    3. Cluster-Specific Specialized Estimators:
       p_c(x) = sigma( beta_c0 + beta_c1 * z_global(x) + f_c(x) )
       Trained on cluster partition with z_global as anchor feature.
    4. Cluster Intercept Parity Alignment:
       Adjusts baseline intercept per latent cluster to neutralize demographic disparity
       without observing protected attributes at inference time.
    5. Cluster-Wise Probability Calibration:
       Fits Platt scaling / Logistic calibrator per cluster on validation fold.
    6. Soft Mixture-of-Experts Inference:
       p(x) = sum_{c=1}^K w_c(x) * p_c(x)
       where w_c(x) is soft responsibility from GMM posterior or KMeans RBF kernel.
    """
    def __init__(
        self,
        n_clusters: int = 2,
        base_model_type: str = 'lightgbm',
        adaptive: bool = False,
        calibrate_clusters: bool = True,
        soft_assignment: bool = True,
        align_cluster_intercept: bool = True,
        parity_shift_weight: float = 1.0,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.base_model_type = base_model_type
        self.adaptive = adaptive
        self.calibrate_clusters = calibrate_clusters
        self.soft_assignment = soft_assignment
        self.align_cluster_intercept = align_cluster_intercept
        self.parity_shift_weight = parity_shift_weight
        self.random_state = random_state
        
        self.global_model = None
        self.cluster_models: Dict[int, Any] = {}
        self.cluster_calibrators: Dict[int, Any] = {}
        self.cluster_intercept_shifts: Dict[int, float] = {}
        self.cluster_model_names: Dict[int, str] = {}
        self.is_fitted = False

    def _get_logit(self, proba_pos: np.ndarray) -> np.ndarray:
        p = np.clip(proba_pos, 1e-5, 1.0 - 1e-5)
        return np.log(p / (1.0 - p))

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cluster_labels: np.ndarray,
        s: Optional[pd.Series] = None,
        sample_weights: Optional[np.ndarray] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        val_cluster_labels: Optional[np.ndarray] = None,
        val_cluster_responsibilities: Optional[np.ndarray] = None
    ):
        y_arr = np.asarray(y).astype(int)
        pos_ratio = float(np.mean(y_arr == 1))
        
        # 1. Train Global Base Model on all data
        self.global_model = get_base_estimator(
            self.base_model_type, 
            imbalance_ratio=pos_ratio, 
            random_state=self.random_state
        )
        if sample_weights is not None:
            self.global_model.fit(X, y_arr, sample_weight=sample_weights)
        else:
            self.global_model.fit(X, y_arr)

        if self.n_clusters <= 1:
            self.is_fitted = True
            return self

        # Compute global logit for training data
        p_global_train = self.global_model.predict_proba(X)[:, 1]
        z_global_train = self._get_logit(p_global_train)
        
        # Validation global logits for calibration
        has_val = (X_val is not None and y_val is not None and val_cluster_labels is not None)
        if has_val:
            y_val_arr = np.asarray(y_val).astype(int)
            p_global_val = self.global_model.predict_proba(X_val)[:, 1]
            z_global_val = self._get_logit(p_global_val)
        
        # 2. Train Cluster-Specific Specialized Estimators
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        for k in range(self.n_clusters):
            mask = (cluster_labels == k)
            n_k = np.sum(mask)
            
            if n_k >= 30 and len(np.unique(y_arr[mask])) > 1:
                X_k = X_df[mask].copy()
                y_k = y_arr[mask]
                z_k = z_global_train[mask]
                w_k = sample_weights[mask] if sample_weights is not None else None
                pos_ratio_k = float(np.mean(y_k == 1))
                
                # Intercept parity shift calculation
                if self.align_cluster_intercept:
                    shift = np.log((pos_ratio + 1e-5) / (1.0 - pos_ratio + 1e-5)) - np.log((pos_ratio_k + 1e-5) / (1.0 - pos_ratio_k + 1e-5))
                    self.cluster_intercept_shifts[k] = float(self.parity_shift_weight * shift)
                else:
                    self.cluster_intercept_shifts[k] = 0.0

                # Feature augmentation with z_global
                X_k_aug = X_k.copy()
                X_k_aug['z_global'] = z_k
                
                clf_type = self.base_model_type
                if self.adaptive:
                    clf_type = 'lightgbm'
                
                self.cluster_model_names[k] = clf_type
                c_model = get_base_estimator(clf_type, imbalance_ratio=pos_ratio_k, random_state=self.random_state + k)
                if w_k is not None:
                    c_model.fit(X_k_aug, y_k, sample_weight=w_k)
                else:
                    c_model.fit(X_k_aug, y_k)
                self.cluster_models[k] = c_model
                
                # 3. Fit Platt / Logistic Calibrator per cluster
                if self.calibrate_clusters and has_val:
                    val_mask_k = (val_cluster_labels == k)
                    if np.sum(val_mask_k) >= 20 and len(np.unique(y_val_arr[val_mask_k])) > 1:
                        X_v_k = X_val[val_mask_k].copy() if isinstance(X_val, pd.DataFrame) else pd.DataFrame(X_val)[val_mask_k].copy()
                        X_v_k['z_global'] = z_global_val[val_mask_k]
                        raw_val_p = c_model.predict_proba(X_v_k)[:, 1]
                        
                        # Apply intercept shift to validation logit
                        shift_k = self.cluster_intercept_shifts.get(k, 0.0)
                        shifted_val_logit = self._get_logit(raw_val_p) + shift_k
                        raw_val_logit_2d = shifted_val_logit.reshape(-1, 1)
                        
                        calibrator = LogisticRegression(C=1.0, solver='lbfgs', random_state=self.random_state)
                        calibrator.fit(raw_val_logit_2d, y_val_arr[val_mask_k])
                        self.cluster_calibrators[k] = calibrator
                    else:
                        self.cluster_calibrators[k] = None
                else:
                    self.cluster_calibrators[k] = None
            else:
                self.cluster_model_names[k] = f"global_{self.base_model_type}"
                self.cluster_models[k] = self.global_model
                self.cluster_calibrators[k] = None
                self.cluster_intercept_shifts[k] = 0.0

        self.is_fitted = True
        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
        cluster_labels: Optional[np.ndarray] = None,
        cluster_responsibilities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        n_samples = len(X)
        probas = np.zeros((n_samples, 2), dtype=np.float64)
        
        # 1. Global prediction & logit
        p_global = self.global_model.predict_proba(X)[:, 1]
        
        if self.n_clusters <= 1 or not self.cluster_models:
            probas[:, 1] = p_global
            probas[:, 0] = 1.0 - p_global
            return probas

        z_global = self._get_logit(p_global)
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_aug = X_df.copy()
        X_aug['z_global'] = z_global

        # 2. Compute probabilities for all clusters
        cluster_p1_matrix = np.zeros((n_samples, self.n_clusters), dtype=np.float64)
        for k in range(self.n_clusters):
            c_model = self.cluster_models.get(k, self.global_model)
            if c_model is self.global_model:
                raw_p1 = p_global
            else:
                raw_p1 = c_model.predict_proba(X_aug)[:, 1]
            
            # Apply cluster intercept shift
            shift_k = self.cluster_intercept_shifts.get(k, 0.0)
            if abs(shift_k) > 1e-4:
                raw_logit = self._get_logit(raw_p1) + shift_k
                raw_p1 = 1.0 / (1.0 + np.exp(-raw_logit))

            # Apply cluster-specific calibration if available
            calibrator = self.cluster_calibrators.get(k)
            if calibrator is not None:
                raw_logit = self._get_logit(raw_p1).reshape(-1, 1)
                p1_cal = calibrator.predict_proba(raw_logit)[:, 1]
                cluster_p1_matrix[:, k] = p1_cal
            else:
                cluster_p1_matrix[:, k] = raw_p1

        # 3. Aggregation (Soft Mixture of Experts vs Hard Selection)
        if self.soft_assignment and cluster_responsibilities is not None:
            # p_moe = sum_k w_k * p_k
            p1_final = np.sum(cluster_responsibilities * cluster_p1_matrix, axis=1)
        elif cluster_labels is not None:
            p1_final = np.zeros(n_samples, dtype=np.float64)
            for k in range(self.n_clusters):
                mask = (cluster_labels == k)
                p1_final[mask] = cluster_p1_matrix[mask, k]
        else:
            p1_final = np.mean(cluster_p1_matrix, axis=1)

        probas[:, 1] = np.clip(p1_final, 0.0, 1.0)
        probas[:, 0] = 1.0 - probas[:, 1]
        return probas


class ClusterClassifierManager:
    """
    Manages training and probability estimation of downstream classifiers across clusters.
    Supports in-processing fairness constraints, global residual specialization (v2),
    soft mixture-of-experts aggregation, and adaptive per-cluster model selection.
    """
    def __init__(
        self,
        n_clusters: int = 4,
        model_type: str = 'lightgbm',
        adaptive: bool = False,
        in_processing: bool = False,
        use_global_residual: bool = True,
        soft_assignment: bool = True,
        calibrate_clusters: bool = True,
        fairness_constraint: str = 'equalized_odds', # 'equalized_odds' or 'demographic_parity'
        eps: float = 0.02,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.model_type = model_type
        self.adaptive = adaptive
        self.in_processing = in_processing
        self.use_global_residual = use_global_residual
        self.soft_assignment = soft_assignment
        self.calibrate_clusters = calibrate_clusters
        self.fairness_constraint = fairness_constraint
        self.eps = eps
        self.random_state = random_state
        
        self.cluster_models: Dict[int, Tuple[str, Any]] = {}
        self.cluster_model_names: Dict[int, str] = {}
        self.cluster_candidate_cv_scores_: Dict[int, Dict[str, float]] = {}
        self.global_fallback_model = None
        self.residual_moe_engine: Optional[GlobalResidualClassifier] = None

    def _create_fair_estimator(self, base_model_name: str = 'lightgbm', imbalance_ratio: float = 1.0):
        base_est = get_base_estimator(
            model_type=base_model_name, 
            imbalance_ratio=imbalance_ratio, 
            random_state=self.random_state
        )
        
        if self.fairness_constraint == 'equalized_odds':
            constraint = EqualizedOdds(difference_bound=self.eps)
        else:
            constraint = DemographicParity(difference_bound=self.eps)
            
        mitigator = ExponentiatedGradient(
            estimator=base_est,
            constraints=constraint,
            eps=self.eps,
            max_iter=30
        )
        return mitigator

    def _select_adaptive_model_for_cluster(
        self, 
        X_k: pd.DataFrame, 
        y_k: pd.Series, 
        s_k: Optional[pd.Series],
        w_k: Optional[np.ndarray],
        cluster_id: int
    ) -> str:
        candidates = ['lightgbm', 'xgboost', 'rf', 'logistic']
        pos_ratio = float(np.mean(y_k == 1))
        
        n_cv = min(len(X_k), 8000)
        if len(X_k) > n_cv:
            idx = np.random.RandomState(self.random_state).choice(len(X_k), size=n_cv, replace=False)
            X_eval = X_k.iloc[idx] if isinstance(X_k, pd.DataFrame) else X_k[idx]
            y_eval = np.asarray(y_k)[idx]
            s_eval = np.asarray(s_k)[idx] if s_k is not None else None
            w_eval = w_k[idx] if w_k is not None else None
        else:
            X_eval = X_k
            y_eval = np.asarray(y_k)
            s_eval = np.asarray(s_k) if s_k is not None else None
            w_eval = w_k

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        model_scores = {}

        for m_name in candidates:
            scores = []
            try:
                for train_idx, val_idx in skf.split(X_eval, y_eval):
                    X_tr = X_eval.iloc[train_idx] if isinstance(X_eval, pd.DataFrame) else X_eval[train_idx]
                    X_va = X_eval.iloc[val_idx] if isinstance(X_eval, pd.DataFrame) else X_eval[val_idx]
                    y_tr, y_va = y_eval[train_idx], y_eval[val_idx]
                    w_tr = w_eval[train_idx] if w_eval is not None else None
                    s_tr = s_eval[train_idx] if s_eval is not None else None
                    s_va = s_eval[val_idx] if s_eval is not None else None

                    if self.in_processing and s_tr is not None and len(np.unique(s_tr)) > 1:
                        try:
                            fair_cand = self._create_fair_estimator(base_model_name=m_name, imbalance_ratio=pos_ratio)
                            fair_cand.fit(X_tr, y_tr, sensitive_features=s_tr)
                            if hasattr(fair_cand, '_pmf_predict'):
                                p_val = fair_cand._pmf_predict(X_va)[:, 1]
                            else:
                                p_val = fair_cand.predict(X_va).astype(float)
                            auc = roc_auc_score(y_va, p_val) if len(np.unique(y_va)) > 1 else 0.5
                            
                            if s_va is not None and len(np.unique(s_va)) > 1:
                                preds_va = (p_val >= 0.5).astype(int)
                                r1 = np.mean(preds_va[s_va == 1]) if np.sum(s_va == 1) > 0 else 0.0
                                r0 = np.mean(preds_va[s_va == 0]) if np.sum(s_va == 0) > 0 else 0.0
                                local_dpd = abs(r1 - r0)
                                score = auc - (0.5 * local_dpd)
                            else:
                                score = auc
                        except Exception:
                            est = get_base_estimator(m_name, imbalance_ratio=pos_ratio, random_state=self.random_state)
                            if w_tr is not None:
                                est.fit(X_tr, y_tr, sample_weight=w_tr)
                            else:
                                est.fit(X_tr, y_tr)
                            p_val = est.predict_proba(X_va)[:, 1] if hasattr(est, 'predict_proba') else est.predict(X_va)
                            score = roc_auc_score(y_va, p_val) if len(np.unique(y_va)) > 1 else 0.5
                    else:
                        est = get_base_estimator(m_name, imbalance_ratio=pos_ratio, random_state=self.random_state)
                        if w_tr is not None:
                            est.fit(X_tr, y_tr, sample_weight=w_tr)
                        else:
                            est.fit(X_tr, y_tr)

                        if hasattr(est, 'predict_proba'):
                            p_val = est.predict_proba(X_va)[:, 1]
                            score = roc_auc_score(y_va, p_val) if len(np.unique(y_va)) > 1 else 0.5
                        else:
                            preds = est.predict(X_va)
                            score = balanced_accuracy_score(y_va, preds)
                    scores.append(float(score))
                model_scores[m_name] = round(float(np.mean(scores)), 4)
            except Exception:
                model_scores[m_name] = 0.0

        self.cluster_candidate_cv_scores_[cluster_id] = model_scores
        best_m = max(model_scores, key=model_scores.get)
        return best_m

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cluster_labels: np.ndarray,
        s: Optional[pd.Series] = None,
        sample_weights: Optional[np.ndarray] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        val_cluster_labels: Optional[np.ndarray] = None,
        val_cluster_responsibilities: Optional[np.ndarray] = None
    ):
        """
        Fits classifier manager. If use_global_residual=True, uses GlobalResidualClassifier (v2).
        """
        if self.use_global_residual and not self.in_processing:
            self.residual_moe_engine = GlobalResidualClassifier(
                n_clusters=self.n_clusters,
                base_model_type=self.model_type,
                adaptive=self.adaptive,
                calibrate_clusters=self.calibrate_clusters,
                soft_assignment=self.soft_assignment,
                random_state=self.random_state
            )
            self.residual_moe_engine.fit(
                X=X,
                y=y,
                cluster_labels=cluster_labels,
                s=s,
                sample_weights=sample_weights,
                X_val=X_val,
                y_val=y_val,
                val_cluster_labels=val_cluster_labels,
                val_cluster_responsibilities=val_cluster_responsibilities
            )
            self.cluster_model_names = self.residual_moe_engine.cluster_model_names
            return self

        # Standard independent fitting fallback
        global_pos_ratio = float(np.mean(y == 1))
        self.global_fallback_model = get_base_estimator(self.model_type, imbalance_ratio=global_pos_ratio, random_state=self.random_state)
        if sample_weights is not None:
            self.global_fallback_model.fit(X, y, sample_weight=sample_weights)
        else:
            self.global_fallback_model.fit(X, y)

        for k in range(self.n_clusters):
            mask = (cluster_labels == k)
            n_k = np.sum(mask)
            
            if n_k > 30 and len(np.unique(y[mask])) > 1:
                X_k = X[mask]
                y_k = y[mask]
                s_k = s[mask] if s is not None else None
                w_k = sample_weights[mask] if sample_weights is not None else None
                pos_ratio_k = float(np.mean(y_k == 1))

                if self.adaptive:
                    chosen_model_type = self._select_adaptive_model_for_cluster(X_k, y_k, s_k, w_k, cluster_id=k)
                else:
                    chosen_model_type = self.model_type

                self.cluster_model_names[k] = chosen_model_type

                if self.in_processing and s_k is not None and len(np.unique(s_k)) > 1:
                    try:
                        fair_model = self._create_fair_estimator(
                            base_model_name=chosen_model_type, 
                            imbalance_ratio=pos_ratio_k
                        )
                        fair_model.fit(X_k, y_k, sensitive_features=s_k)
                        self.cluster_models[k] = ('fair', fair_model)
                    except Exception as e:
                        base_m = get_base_estimator(chosen_model_type, imbalance_ratio=pos_ratio_k, random_state=self.random_state)
                        if w_k is not None:
                            base_m.fit(X_k, y_k, sample_weight=w_k)
                        else:
                            base_m.fit(X_k, y_k)
                        self.cluster_models[k] = ('standard', base_m)
                else:
                    base_m = get_base_estimator(chosen_model_type, imbalance_ratio=pos_ratio_k, random_state=self.random_state)
                    if w_k is not None:
                        base_m.fit(X_k, y_k, sample_weight=w_k)
                    else:
                        base_m.fit(X_k, y_k)
                    self.cluster_models[k] = ('standard', base_m)
            else:
                self.cluster_model_names[k] = f"fallback_{self.model_type}"
                self.cluster_models[k] = ('global', self.global_fallback_model)

    def predict_proba(
        self,
        X: pd.DataFrame,
        cluster_labels: Optional[np.ndarray] = None,
        cluster_responsibilities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self.use_global_residual and self.residual_moe_engine is not None:
            return self.residual_moe_engine.predict_proba(
                X=X, 
                cluster_labels=cluster_labels, 
                cluster_responsibilities=cluster_responsibilities
            )

        n_samples = len(X)
        probas = np.zeros((n_samples, 2), dtype=np.float64)

        if cluster_labels is None:
            p = self.global_fallback_model.predict_proba(X)
            return p

        for k in range(self.n_clusters):
            mask = (cluster_labels == k)
            if not np.any(mask):
                continue
            
            X_k = X[mask]
            m_type, model = self.cluster_models.get(k, ('global', self.global_fallback_model))
            
            if m_type == 'fair':
                try:
                    p1 = model._pmf_predict(X_k)[:, 1] if hasattr(model, '_pmf_predict') else model.predict(X_k)
                except Exception:
                    p1 = model.predict(X_k).astype(float)
                p0 = 1.0 - p1
                probas[mask, 0] = p0
                probas[mask, 1] = p1
            else:
                if hasattr(model, 'predict_proba'):
                    p = model.predict_proba(X_k)
                    if p.shape[1] == 2:
                        probas[mask] = p
                    else:
                        probas[mask, 1] = p[:, 0]
                        probas[mask, 0] = 1.0 - p[:, 0]
                else:
                    p1 = model.predict(X_k).astype(float)
                    probas[mask, 1] = p1
                    probas[mask, 0] = 1.0 - p1

        return probas
