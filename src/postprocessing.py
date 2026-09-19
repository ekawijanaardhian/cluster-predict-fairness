"""
Stage 3: Post-Processing Bias Mitigation - Reject Option Classifier (ROC)
Implements Reject-Option Classification (Kamiran et al., 2012),
group-aware threshold optimization, validation-based threshold calibration,
and dual objective optimization (DPD vs EOD vs Balanced).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
from sklearn.metrics import roc_curve

class RejectOptionClassifier:
    """
    Reject-Option Classification (ROC) post-processing (Kamiran et al., 2012).
    Adjusts predictions in the critical decision margin [base_threshold - margin, base_threshold + margin]
    to counteract disparate impact on unprivileged demographic groups.
    Supports validation-calibrated base threshold (e.g. Youden's J) and dual optimization targets.
    """
    def __init__(
        self,
        base_threshold: float = 0.5,
        margin: float = 0.10,
        optimize_margin: bool = True,
        calibrate_base_threshold: bool = True,
        optimization_target: str = 'dpd' # 'dpd', 'eod', 'balanced', or 'pareto'
    ):
        self.base_threshold = base_threshold
        self.margin = margin
        self.optimize_margin = optimize_margin
        self.calibrate_base_threshold = calibrate_base_threshold
        self.optimization_target = optimization_target.lower()
        self.best_margin = margin
        self.best_base_threshold = base_threshold
        self.margin_search_history_: List[Dict[str, float]] = []

    def calibrate_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Calibrates base threshold on validation set using Youden's J statistic (TPR - FPR).
        Crucial for class-imbalanced datasets (e.g., 13.9% disease prevalence).
        """
        p1 = y_proba[:, 1] if (isinstance(y_proba, np.ndarray) and y_proba.ndim > 1) else np.asarray(y_proba)
        y_t = np.asarray(y_true).astype(int)
        
        if len(np.unique(y_t)) < 2:
            return 0.5

        fpr, tpr, thresholds = roc_curve(y_t, p1)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = float(thresholds[best_idx])
        
        # Clamp to reasonable range [0.05, 0.95]
        best_thresh = max(0.05, min(0.95, best_thresh))
        return best_thresh

    def fit(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        s: np.ndarray
    ):
        """
        Calibrates the optimal reject-option boundary margin and threshold using validation data.
        """
        y_t = np.asarray(y_true).astype(int)
        s_arr = np.asarray(s).astype(int)
        p1 = y_proba[:, 1] if (isinstance(y_proba, np.ndarray) and y_proba.ndim > 1) else np.asarray(y_proba)

        if self.calibrate_base_threshold:
            self.best_base_threshold = self.calibrate_threshold(y_t, p1)
        else:
            self.best_base_threshold = self.base_threshold

        if not self.optimize_margin:
            self.best_margin = self.margin
            return self

        best_score = -float('inf')
        best_m = self.margin
        self.margin_search_history_ = []

        # Grid search over candidate margins [0.005, 0.25]
        candidate_margins = np.linspace(0.005, 0.25, 50)
        for m in candidate_margins:
            preds = self._apply_roc(p1, s_arr, m, threshold=self.best_base_threshold)
            
            # Accuracy / Balanced Accuracy
            acc = float(np.mean(preds == y_t))
            
            # Sub-group probabilities and error rates
            mask_s1 = (s_arr == 1)
            mask_s0 = (s_arr == 0)
            
            p_priv = float(np.mean(preds[mask_s1])) if np.sum(mask_s1) > 0 else 0.0
            p_unpriv = float(np.mean(preds[mask_s0])) if np.sum(mask_s0) > 0 else 0.0
            dp_diff = abs(p_priv - p_unpriv)
            
            # Equalized Odds computation (FPR diff and TPR diff)
            tpr_priv = float(np.mean(preds[(mask_s1) & (y_t == 1)] == 1)) if np.sum((mask_s1) & (y_t == 1)) > 0 else 0.0
            tpr_unpriv = float(np.mean(preds[(mask_s0) & (y_t == 1)] == 1)) if np.sum((mask_s0) & (y_t == 1)) > 0 else 0.0
            fpr_priv = float(np.mean(preds[(mask_s1) & (y_t == 0)] == 1)) if np.sum((mask_s1) & (y_t == 0)) > 0 else 0.0
            fpr_unpriv = float(np.mean(preds[(mask_s0) & (y_t == 0)] == 1)) if np.sum((mask_s0) & (y_t == 0)) > 0 else 0.0
            
            eo_diff = max(abs(tpr_priv - tpr_unpriv), abs(fpr_priv - fpr_unpriv))

            # Multi-objective scoring based on chosen target
            if self.optimization_target == 'eod':
                score = acc - 0.5 * eo_diff
            elif self.optimization_target == 'balanced':
                score = acc - 0.25 * dp_diff - 0.25 * eo_diff
            else: # 'dpd' default
                score = acc - 0.5 * dp_diff

            self.margin_search_history_.append({
                'margin': round(float(m), 4),
                'threshold': round(float(self.best_base_threshold), 4),
                'accuracy': round(acc, 4),
                'dp_diff': round(dp_diff, 4),
                'eo_diff': round(eo_diff, 4),
                'score': round(score, 4)
            })

            if score > best_score:
                best_score = score
                best_m = m

        self.best_margin = float(best_m)
        return self

    def _apply_roc(
        self, 
        p1: np.ndarray, 
        s_arr: np.ndarray, 
        margin: float, 
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """Applies ROC decision rule given probabilities, margin, and threshold."""
        t = self.best_base_threshold if threshold is None else threshold
        preds = (p1 >= t).astype(int)
        
        # Region of uncertainty (reject region around threshold)
        lower_bound = max(0.0, t - margin)
        upper_bound = min(1.0, t + margin)
        in_margin = (p1 >= lower_bound) & (p1 <= upper_bound)
        
        # For unprivileged group (s=0): in margin region, assign favorable outcome (1)
        unpriv_mask = (s_arr == 0) & in_margin
        preds[unpriv_mask] = 1
        
        # For privileged group (s=1): in margin region, assign strict outcome (0)
        priv_mask = (s_arr == 1) & in_margin
        preds[priv_mask] = 0
        
        return preds

    def predict(
        self, 
        y_proba: np.ndarray, 
        s: np.ndarray, 
        threshold: Optional[float] = None,
        margin: Optional[float] = None
    ) -> np.ndarray:
        """
        Transforms probability predictions into fair binary predictions.
        """
        p1 = y_proba[:, 1] if (isinstance(y_proba, np.ndarray) and y_proba.ndim > 1) else np.asarray(y_proba)
        s_arr = np.asarray(s).astype(int)
        m = self.best_margin if margin is None else margin
        t = self.best_base_threshold if threshold is None else threshold
        return self._apply_roc(p1, s_arr, m, threshold=t)

