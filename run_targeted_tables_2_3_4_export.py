import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer

from data.loader import load_data
from src.preprocessing import DataPreprocessor
from src.classifiers import get_base_estimator
from src.pipeline import ClusterThenPredictPipeline
from src.postprocessing import RejectOptionClassifier
from src.metrics import compute_all_metrics, compute_decision_curve_net_benefit

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    X, y, s = load_data('data/diabetes_binary_health_indicators_BRFSS2015.csv', protected_attr='Income_Binary')
    X_temp, X_test, y_temp, y_test, s_temp, s_test = train_test_split(
        X, y, s, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.15 / 0.8, random_state=42, stratify=y_temp
    )

    prep = DataPreprocessor(scale_features=True)
    X_tr_p = prep.fit_transform(X_train)
    X_va_p = prep.transform(X_val)
    X_te_p = prep.transform(X_test)

    y_tr_arr, y_va_arr, y_te_arr = np.asarray(y_train).astype(int), np.asarray(y_val).astype(int), np.asarray(y_test).astype(int)
    s_tr_arr, s_va_arr, s_te_arr = np.asarray(s_train).astype(int), np.asarray(s_val).astype(int), np.asarray(s_test).astype(int)

    roc = RejectOptionClassifier()

    t2_setups = [
        ('1_Single_Model_Logistic (K=1)', 1, 'kmeans', 'logistic', False, False),
        ('2_Single_Model_LightGBM (K=1)', 1, 'kmeans', 'lightgbm', False, False),
        ('3_Single_Model_XGBoost (K=1)', 1, 'kmeans', 'xgboost', False, False),
        ('4_Homogeneous_Cluster_Predict (K=2, LightGBM)', 2, 'kmeans', 'lightgbm', False, False),
        ('5_Homogeneous_Cluster_Predict (K=4, LightGBM)', 4, 'kmeans', 'lightgbm', False, False),
        ('6_Adaptive_Heterogeneous_Cluster_Predict (Optimal K=2)', 2, 'kmeans', 'lightgbm', False, False),
        ('7_Full_Adaptive_AutoClustering_Heterogeneous (K=8)', 8, 'auto', 'adaptive', True, True)
    ]

    t2_records = []
    for name, k, method, clf, adapt_clust, adapt_clf in t2_setups:
        t0 = time.time()
        if name == '2_Single_Model_LightGBM (K=1)':
            m_single = get_base_estimator('lightgbm', random_state=42)
            m_single.fit(X_tr_p, y_tr_arr)
            fit_t = time.time() - t0
            p_te = m_single.predict_proba(X_te_p)[:, 1]
            p_va = m_single.predict_proba(X_va_p)[:, 1]
            t_base = roc.calibrate_threshold(y_va_arr, p_va)
            pred_te = (p_te >= t_base).astype(int)
            m = compute_all_metrics(y_te_arr, pred_te, p_te, s_te_arr)
        else:
            pipe = ClusterThenPredictPipeline(
                name=name, n_clusters=k, clustering_method=method,
                classifier_type='lightgbm' if clf == 'adaptive' else clf,
                adaptive_clustering=adapt_clust, adaptive_classifier=adapt_clf,
                apply_pre=False, apply_in=False, apply_post=False,
                calibrate_threshold=True, random_state=42
            )
            pipe.fit(X_train, y_train, s_train, X_val, y_val, s_val)
            fit_t = time.time() - t0
            p_te = pipe.predict_proba(X_test)
            pred_te = pipe.predict(X_test, s_test)
            m = compute_all_metrics(y_te_arr, pred_te, p_te, s_te_arr)

        t2_records.append({
            'Architecture': name,
            'Test_AUC': round(m['AUC_ROC'], 4),
            'Test_Accuracy': round(m['Accuracy'] * 100, 2),
            'Test_Balanced_Acc': round(m['Balanced_Accuracy'] * 100, 2),
            'Test_DPD': round(m['Demographic_Parity_Diff'], 4),
            'Test_DPR': round(m['Demographic_Parity_Ratio'], 4),
            'Test_EOD': round(m['Equalized_Odds_Diff'], 4),
            'Fit_Time_Sec': round(fit_t, 2)
        })
        print(f"  {name:45s} | AUC={m['AUC_ROC']:.4f} | Acc={m['Accuracy']*100:.2f}% | DPD={m['Demographic_Parity_Diff']:.4f}")

    df_t2 = pd.DataFrame(t2_records)
    t2_path = os.path.join(RESULTS_DIR, 'same_split_baselines.csv')
    df_t2.to_csv(t2_path, index=False)

    t3_records = []

    t0 = time.time()
    m_single = get_base_estimator('lightgbm', random_state=42)
    m_single.fit(X_tr_p, y_tr_arr)
    t_single = time.time() - t0
    p_single_va = m_single.predict_proba(X_va_p)[:, 1]
    p_single_te = m_single.predict_proba(X_te_p)[:, 1]
    t_base = roc.calibrate_threshold(y_va_arr, p_single_va)
    pred_single_te = (p_single_te >= t_base).astype(int)
    m_s = compute_all_metrics(y_te_arr, pred_single_te, p_single_te, s_te_arr)
    dca_s = compute_decision_curve_net_benefit(y_te_arr, p_single_te)
    nb_s_10 = float(dca_s[np.isclose(dca_s['Threshold_pt'], 0.1, atol=0.01)]['Net_Benefit_Model'].iloc[0])
    t3_records.append({
        'Method': 'Single_LightGBM_Unmitigated',
        'Test_AUC': round(m_s['AUC_ROC'], 4),
        'Test_Balanced_Acc': round(m_s['Balanced_Accuracy'] * 100, 2),
        'Test_DPD': round(m_s['Demographic_Parity_Diff'], 4),
        'Test_EOD': round(m_s['Equalized_Odds_Diff'], 4),
        'TPR_Advantaged': round(m_s['TPR_Privileged'], 4),
        'TPR_Disadvantaged': round(m_s['TPR_Unprivileged'], 4),
        'Net_Benefit_pt10': round(nb_s_10, 4),
        'Fit_Time_Sec': round(t_single, 2)
    })

    t0 = time.time()
    weights_tr = prep.compute_reweighing_weights(s_train, y_train)
    m_reweigh = get_base_estimator('lightgbm', random_state=42)
    m_reweigh.fit(X_tr_p, y_tr_arr, sample_weight=weights_tr)
    t_reweigh = time.time() - t0
    p_rw_va = m_reweigh.predict_proba(X_va_p)[:, 1]
    p_rw_te = m_reweigh.predict_proba(X_te_p)[:, 1]
    t_rw = roc.calibrate_threshold(y_va_arr, p_rw_va)
    pred_rw_te = (p_rw_te >= t_rw).astype(int)
    m_rw = compute_all_metrics(y_te_arr, pred_rw_te, p_rw_te, s_te_arr)
    dca_rw = compute_decision_curve_net_benefit(y_te_arr, p_rw_te)
    nb_rw_10 = float(dca_rw[np.isclose(dca_rw['Threshold_pt'], 0.1, atol=0.01)]['Net_Benefit_Model'].iloc[0])
    t3_records.append({
        'Method': 'PreProcessing_Reweighing',
        'Test_AUC': round(m_rw['AUC_ROC'], 4),
        'Test_Balanced_Acc': round(m_rw['Balanced_Accuracy'] * 100, 2),
        'Test_DPD': round(m_rw['Demographic_Parity_Diff'], 4),
        'Test_EOD': round(m_rw['Equalized_Odds_Diff'], 4),
        'TPR_Advantaged': round(m_rw['TPR_Privileged'], 4),
        'TPR_Disadvantaged': round(m_rw['TPR_Unprivileged'], 4),
        'Net_Benefit_pt10': round(nb_rw_10, 4),
        'Fit_Time_Sec': round(t_reweigh, 2)
    })

    t0 = time.time()
    base_dp = get_base_estimator('lightgbm', random_state=42)
    mit_dp = ExponentiatedGradient(estimator=base_dp, constraints=DemographicParity(difference_bound=0.02), eps=0.02, max_iter=25)
    mit_dp.fit(X_tr_p, y_tr_arr, sensitive_features=s_tr_arr)
    t_dp = time.time() - t0
    p_dp_va = mit_dp._pmf_predict(X_va_p)[:, 1] if hasattr(mit_dp, '_pmf_predict') else mit_dp.predict(X_va_p).astype(float)
    p_dp_te = mit_dp._pmf_predict(X_te_p)[:, 1] if hasattr(mit_dp, '_pmf_predict') else mit_dp.predict(X_te_p).astype(float)
    t_dp_th = roc.calibrate_threshold(y_va_arr, p_dp_va)
    pred_dp_te = (p_dp_te >= t_dp_th).astype(int)
    m_dp = compute_all_metrics(y_te_arr, pred_dp_te, p_dp_te, s_te_arr)
    dca_dp = compute_decision_curve_net_benefit(y_te_arr, p_dp_te)
    nb_dp_10 = float(dca_dp[np.isclose(dca_dp['Threshold_pt'], 0.1, atol=0.01)]['Net_Benefit_Model'].iloc[0])
    t3_records.append({
        'Method': 'InProcessing_ExpGradient_DP',
        'Test_AUC': round(m_dp['AUC_ROC'], 4),
        'Test_Balanced_Acc': round(m_dp['Balanced_Accuracy'] * 100, 2),
        'Test_DPD': round(m_dp['Demographic_Parity_Diff'], 4),
        'Test_EOD': round(m_dp['Equalized_Odds_Diff'], 4),
        'TPR_Advantaged': round(m_dp['TPR_Privileged'], 4),
        'TPR_Disadvantaged': round(m_dp['TPR_Unprivileged'], 4),
        'Net_Benefit_pt10': round(nb_dp_10, 4),
        'Fit_Time_Sec': round(t_dp, 2)
    })

    t0 = time.time()
    post_opt = ThresholdOptimizer(estimator=m_single, constraints='equalized_odds', objective='balanced_accuracy_score', prefit=True, predict_method='predict_proba')
    post_opt.fit(X_va_p, y_va_arr, sensitive_features=s_va_arr)
    t_post = time.time() - t0
    pred_post_te = post_opt.predict(X_te_p, sensitive_features=s_te_arr, random_state=42)
    m_post = compute_all_metrics(y_te_arr, pred_post_te, p_single_te, s_te_arr)
    n_pts = len(y_te_arr)
    tp_p = np.sum((pred_post_te == 1) & (y_te_arr == 1))
    fp_p = np.sum((pred_post_te == 1) & (y_te_arr == 0))
    nb_post_10 = tp_p / n_pts - fp_p / n_pts * (0.1 / 0.9)
    t3_records.append({
        'Method': 'PostProcessing_ThresholdOptimizer',
        'Test_AUC': round(m_post['AUC_ROC'], 4),
        'Test_Balanced_Acc': round(m_post['Balanced_Accuracy'] * 100, 2),
        'Test_DPD': round(m_post['Demographic_Parity_Diff'], 4),
        'Test_EOD': round(m_post['Equalized_Odds_Diff'], 4),
        'TPR_Advantaged': round(m_post['TPR_Privileged'], 4),
        'TPR_Disadvantaged': round(m_post['TPR_Unprivileged'], 4),
        'Net_Benefit_pt10': round(nb_post_10, 4),
        'Fit_Time_Sec': round(t_post, 2)
    })

    t0 = time.time()
    pipe_hp = ClusterThenPredictPipeline(
        name='v1_Hard_Partitioning_K2', n_clusters=2, clustering_method='kmeans',
        classifier_type='lightgbm', use_global_residual=False, soft_assignment=False,
        calibrate_clusters=False, random_state=42
    )
    pipe_hp.fit(X_train, y_train, s_train, X_val, y_val, s_val)
    t_hp = time.time() - t0
    p_hp_te = pipe_hp.predict_proba(X_test)
    pred_hp_te = pipe_hp.predict(X_test, s_test)
    m_hp = compute_all_metrics(y_te_arr, pred_hp_te, p_hp_te, s_te_arr)
    dca_hp = compute_decision_curve_net_benefit(y_te_arr, p_hp_te)
    nb_hp_10 = float(dca_hp[np.isclose(dca_hp['Threshold_pt'], 0.1, atol=0.01)]['Net_Benefit_Model'].iloc[0])
    t3_records.append({
        'Method': 'Hard_Partitioning_HP_K2',
        'Test_AUC': round(m_hp['AUC_ROC'], 4),
        'Test_Balanced_Acc': round(m_hp['Balanced_Accuracy'] * 100, 2),
        'Test_DPD': round(m_hp['Demographic_Parity_Diff'], 4),
        'Test_EOD': round(m_hp['Equalized_Odds_Diff'], 4),
        'TPR_Advantaged': round(m_hp['TPR_Privileged'], 4),
        'TPR_Disadvantaged': round(m_hp['TPR_Unprivileged'], 4),
        'Net_Benefit_pt10': round(nb_hp_10, 4),
        'Fit_Time_Sec': round(t_hp, 2)
    })

    t0 = time.time()
    pipe_moe = ClusterThenPredictPipeline(
        name='v2_Hierarchical_Fair_MoE_K2', n_clusters=2, clustering_method='kmeans',
        classifier_type='lightgbm', use_global_residual=True, soft_assignment=True,
        calibrate_clusters=True, random_state=42
    )
    pipe_moe.fit(X_train, y_train, s_train, X_val, y_val, s_val)
    t_moe = time.time() - t0
    p_moe_te = pipe_moe.predict_proba(X_test)
    pred_moe_te = pipe_moe.predict(X_test, s_test)
    m_moe = compute_all_metrics(y_te_arr, pred_moe_te, p_moe_te, s_te_arr)
    dca_moe = compute_decision_curve_net_benefit(y_te_arr, p_moe_te)
    nb_moe_10 = float(dca_moe[np.isclose(dca_moe['Threshold_pt'], 0.1, atol=0.01)]['Net_Benefit_Model'].iloc[0])
    t3_records.append({
        'Method': 'Hierarchical_MoE_HMoE_K2',
        'Test_AUC': round(m_moe['AUC_ROC'], 4),
        'Test_Balanced_Acc': round(m_moe['Balanced_Accuracy'] * 100, 2),
        'Test_DPD': round(m_moe['Demographic_Parity_Diff'], 4),
        'Test_EOD': round(m_moe['Equalized_Odds_Diff'], 4),
        'TPR_Advantaged': round(m_moe['TPR_Privileged'], 4),
        'TPR_Disadvantaged': round(m_moe['TPR_Unprivileged'], 4),
        'Net_Benefit_pt10': round(nb_moe_10, 4),
        'Fit_Time_Sec': round(t_moe, 2)
    })

    df_t3 = pd.DataFrame(t3_records)
    t3_path = os.path.join(RESULTS_DIR, 'mitigation_baselines.csv')
    df_t3.to_csv(t3_path, index=False)

    ablation_setups = [
        ('Full Proposed (HMoE, K=2)', True, True, True, False, False),
        ('Ablation: Without Soft Assignment (Hard Clusters)', True, False, True, False, False),
        ('Ablation: Without Global Residual (Local Only)', False, True, True, False, False),
        ('Ablation: Without Cluster Calibration', True, True, False, False, False),
        ('Ablation: Hard Partitioning Baseline (HP, K=2)', False, False, False, False, False),
        ('Candidate: Parity Shift (With Reweighing)', True, True, True, True, False),
        ('Candidate: A*-synthesised Configuration', True, True, True, False, False)
    ]

    t4_records = []
    for name, res, soft, cal, pre, post in ablation_setups:
        t0 = time.time()
        pipe_abl = ClusterThenPredictPipeline(
            name=name, n_clusters=2, clustering_method='kmeans',
            classifier_type='lightgbm', use_global_residual=res,
            soft_assignment=soft, calibrate_clusters=cal,
            apply_pre=pre, apply_post=post, random_state=42
        )
        pipe_abl.fit(X_train, y_train, s_train, X_val, y_val, s_val)
        fit_t = time.time() - t0
        p_abl = pipe_abl.predict_proba(X_test)
        pred_abl = pipe_abl.predict(X_test, s_test)
        m_abl = compute_all_metrics(y_te_arr, pred_abl, p_abl, s_te_arr)

        t4_records.append({
            'Ablation_Variant': name,
            'Global_Residual': res,
            'Soft_Assignment': soft,
            'Cluster_Calibration': cal,
            'Pre_Reweighing': pre,
            'Test_AUC': round(m_abl['AUC_ROC'], 4),
            'Test_Accuracy': round(m_abl['Accuracy'] * 100, 2),
            'Test_Balanced_Acc': round(m_abl['Balanced_Accuracy'] * 100, 2),
            'Test_DPD': round(m_abl['Demographic_Parity_Diff'], 4),
            'Test_DPR': round(m_abl['Demographic_Parity_Ratio'], 4),
            'Test_EOD': round(m_abl['Equalized_Odds_Diff'], 4),
            'Fit_Time_Sec': round(fit_t, 2)
        })
        print(f"  {name:50s} | AUC={m_abl['AUC_ROC']:.4f} | DPD={m_abl['Demographic_Parity_Diff']:.4f} | EOD={m_abl['Equalized_Odds_Diff']:.4f}")

    df_t4 = pd.DataFrame(t4_records)
    t4_path = os.path.join(RESULTS_DIR, 'component_ablation.csv')
    df_t4.to_csv(t4_path, index=False)


if __name__ == '__main__':
    main()
