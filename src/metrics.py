"""
Fairness and Clinical Predictive Performance Evaluation Suite.
Implements Demographic Parity, Equalized Odds, AUC-ROC, trade-off ratios,
and multi-layer interaction quantification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, List, Tuple
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix
)
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_positive_rate,
    false_negative_rate
)

def compute_all_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_proba: Union[pd.Series, np.ndarray],
    s: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Computes full spectrum of clinical utility and algorithmic fairness metrics.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)
    s_arr = np.asarray(s).astype(int)
    
    p1 = y_proba[:, 1] if (isinstance(y_proba, np.ndarray) and y_proba.ndim > 1) else np.asarray(y_proba)

    # 1. Clinical Utility Metrics
    acc = accuracy_score(y_t, y_p)
    bacc = balanced_accuracy_score(y_t, y_p)
    f1 = f1_score(y_t, y_p, zero_division=0)
    precision = precision_score(y_t, y_p, zero_division=0)
    recall = recall_score(y_t, y_p, zero_division=0)
    
    try:
        auc = roc_auc_score(y_t, p1)
    except Exception:
        auc = 0.5

    # 2. Sub-group error rates
    # Privileged (s=1)
    mask_s1 = (s_arr == 1)
    # Unprivileged (s=0)
    mask_s0 = (s_arr == 0)

    fpr_s1 = false_positive_rate(y_t[mask_s1], y_p[mask_s1]) if np.sum(mask_s1) > 0 else 0.0
    fpr_s0 = false_positive_rate(y_t[mask_s0], y_p[mask_s0]) if np.sum(mask_s0) > 0 else 0.0
    fnr_s1 = false_negative_rate(y_t[mask_s1], y_p[mask_s1]) if np.sum(mask_s1) > 0 else 0.0
    fnr_s0 = false_negative_rate(y_t[mask_s0], y_p[mask_s0]) if np.sum(mask_s0) > 0 else 0.0

    # True Positive Rate per group (Sensitivity) for Levelling-Up assessment
    tpr_s1 = 1.0 - fnr_s1
    tpr_s0 = 1.0 - fnr_s0
    min_tpr = min(tpr_s1, tpr_s0)

    # 3. Disparity Metrics
    dp_diff = demographic_parity_difference(y_t, y_p, sensitive_features=s_arr)
    dp_ratio = demographic_parity_ratio(y_t, y_p, sensitive_features=s_arr)
    eo_diff = equalized_odds_difference(y_t, y_p, sensitive_features=s_arr)
    eo_ratio = equalized_odds_ratio(y_t, y_p, sensitive_features=s_arr)
    
    fpr_diff = abs(fpr_s1 - fpr_s0)
    fnr_diff = abs(fnr_s1 - fnr_s0)
    tpr_diff = abs(tpr_s1 - tpr_s0)

    return {
        'AUC_ROC': round(float(auc), 4),
        'Accuracy': round(float(acc), 4),
        'Balanced_Accuracy': round(float(bacc), 4),
        'F1_Score': round(float(f1), 4),
        'Precision': round(float(precision), 4),
        'Recall': round(float(recall), 4),
        'Demographic_Parity_Diff': round(float(dp_diff), 4),
        'Demographic_Parity_Ratio': round(float(dp_ratio), 4),
        'Equalized_Odds_Diff': round(float(eo_diff), 4),
        'Equalized_Odds_Ratio': round(float(eo_ratio), 4),
        'TPR_Privileged': round(float(tpr_s1), 4),
        'TPR_Unprivileged': round(float(tpr_s0), 4),
        'Min_TPR': round(float(min_tpr), 4),
        'TPR_Disparity': round(float(tpr_diff), 4),
        'FPR_Disparity': round(float(fpr_diff), 4),
        'FNR_Disparity': round(float(fnr_diff), 4),
        'FPR_Privileged': round(float(fpr_s1), 4),
        'FPR_Unprivileged': round(float(fpr_s0), 4),
        'FNR_Privileged': round(float(fnr_s1), 4),
        'FNR_Unprivileged': round(float(fnr_s0), 4)
    }

def compute_decision_curve_net_benefit(
    y_true: Union[pd.Series, np.ndarray],
    y_proba: Union[pd.Series, np.ndarray],
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Computes Decision Curve Analysis (DCA) Net Benefit across clinical risk thresholds pt.
    Net Benefit (pt) = (TP / N) - (FP / N) * (pt / (1 - pt))
    Also computes 'Treat All' and 'Treat None' reference strategies.
    """
    y_t = np.asarray(y_true).astype(int)
    p1 = y_proba[:, 1] if (isinstance(y_proba, np.ndarray) and y_proba.ndim > 1) else np.asarray(y_proba)
    n = len(y_t)
    prevalence = np.mean(y_t == 1)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 50)

    rows = []
    for pt in thresholds:
        if pt <= 0.0 or pt >= 1.0:
            continue
        weight = pt / (1.0 - pt)
        
        # Model Strategy
        preds = (p1 >= pt).astype(int)
        tp = np.sum((preds == 1) & (y_t == 1))
        fp = np.sum((preds == 1) & (y_t == 0))
        nb_model = (tp / n) - (fp / n) * weight

        # Treat All Strategy: All patients predicted positive
        tp_all = np.sum(y_t == 1)
        fp_all = np.sum(y_t == 0)
        nb_all = (tp_all / n) - (fp_all / n) * weight

        # Treat None Strategy: No patients predicted positive (Net Benefit = 0)
        nb_none = 0.0

        rows.append({
            'Threshold_pt': round(float(pt), 4),
            'Weight': round(float(weight), 4),
            'Net_Benefit_Model': round(float(nb_model), 6),
            'Net_Benefit_Treat_All': round(float(nb_all), 6),
            'Net_Benefit_Treat_None': round(float(nb_none), 6)
        })

    return pd.DataFrame(rows)

def format_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Formats comparison dictionary into a clean pandas DataFrame."""
    df = pd.DataFrame(metrics_dict).T
    return df

