"""
Multi-Stage Cluster-then-Predict Architecture Pipeline.
Integrates Pre-processing (Re-weighing), Stage 1 Clustering (Static / Adaptive),
Stage 2 Classifier Manager (Static / Adaptive per-cluster with In-processing constraints),
and Stage 3 Post-Processing (Reject-Option Classification).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from src.preprocessing import DataPreprocessor
from src.clustering import PopulationClusterer
from src.classifiers import ClusterClassifierManager
from src.postprocessing import RejectOptionClassifier
from src.metrics import compute_all_metrics

class ClusterThenPredictPipeline:
    """
    Unified experimental pipeline for Fairness-Aware Cluster-then-Predict.
    Allows independent activation of Pre-processing, In-processing, and Post-processing layers,
    as well as Adaptive Clustering and Adaptive Classifier Selection.
    """
    def __init__(
        self,
        name: str = "CTP_Pipeline",
        n_clusters: int = 4,
        clustering_method: str = 'kmeans',
        classifier_type: str = 'lightgbm',
        adaptive_clustering: bool = False,
        adaptive_classifier: bool = False,
        use_global_residual: bool = True,
        soft_assignment: bool = True,
        calibrate_clusters: bool = True,
        apply_pre: bool = False,
        apply_in: bool = False,
        apply_post: bool = False,
        fairness_constraint: str = 'equalized_odds',
        eps: float = 0.02,
        roc_margin: float = 0.10,
        roc_objective: str = 'dpd',
        calibrate_threshold: bool = True,
        k_min: int = 2,
        k_max: int = 10,
        random_state: int = 42
    ):
        self.name = name
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.classifier_type = classifier_type
        self.adaptive_clustering = adaptive_clustering
        self.adaptive_classifier = adaptive_classifier
        self.use_global_residual = use_global_residual
        self.soft_assignment = soft_assignment
        self.calibrate_clusters = calibrate_clusters
        self.apply_pre = apply_pre
        self.apply_in = apply_in
        self.apply_post = apply_post
        self.fairness_constraint = fairness_constraint
        self.eps = eps
        self.roc_margin = roc_margin
        self.roc_objective = roc_objective
        self.calibrate_threshold = calibrate_threshold
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state
        self.calibrated_base_threshold = 0.5

        self.preprocessor = DataPreprocessor(scale_features=True)
        self.clusterer = PopulationClusterer(
            n_clusters=n_clusters, 
            method=clustering_method, 
            adaptive=adaptive_clustering,
            k_min=k_min,
            k_max=k_max,
            random_state=random_state
        )
        self.classifier_manager = ClusterClassifierManager(
            n_clusters=n_clusters,
            model_type=classifier_type,
            adaptive=adaptive_classifier,
            in_processing=apply_in,
            use_global_residual=use_global_residual,
            soft_assignment=soft_assignment,
            calibrate_clusters=calibrate_clusters,
            fairness_constraint=fairness_constraint,
            eps=eps,
            random_state=random_state
        )
        self.postprocessor = RejectOptionClassifier(
            base_threshold=0.5,
            margin=roc_margin,
            optimize_margin=True,
            calibrate_base_threshold=calibrate_threshold,
            optimization_target=roc_objective
        )

        self.cluster_demographics_train = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        s_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        s_val: Optional[pd.Series] = None
    ):
        """
        Fits the multi-stage pipeline on training data.
        """

        X_tr_proc = self.preprocessor.fit_transform(X_train)
        
        sample_weights = None
        if self.apply_pre:
            sample_weights = self.preprocessor.compute_reweighing_weights(s_train, y_train)

        train_clusters = self.clusterer.fit_predict(
            X_tr_proc, 
            s=s_train, 
            sample_weight=sample_weights
        )
        train_resp = self.clusterer.predict_proba(X_tr_proc)
        
        self.cluster_demographics_train = self.clusterer.analyze_cluster_demographics(
            train_clusters, s_train, y_train
        )
        

        actual_k = self.clusterer.n_clusters
        self.classifier_manager.n_clusters = actual_k

        X_v_proc = None
        val_clusters = None
        val_resp = None
        if X_val is not None and y_val is not None:
            X_v_proc = self.preprocessor.transform(X_val)
            val_clusters = self.clusterer.predict(X_v_proc)
            val_resp = self.clusterer.predict_proba(X_v_proc)

        self.classifier_manager.fit(
            X=X_tr_proc,
            y=y_train,
            cluster_labels=train_clusters,
            s=s_train,
            sample_weights=sample_weights,
            X_val=X_v_proc,
            y_val=y_val,
            val_cluster_labels=val_clusters,
            val_cluster_responsibilities=val_resp
        )

        if X_v_proc is not None and y_val is not None:
            val_proba = self.classifier_manager.predict_proba(
                X_v_proc, 
                cluster_labels=val_clusters, 
                cluster_responsibilities=val_resp
            )
            
            if self.calibrate_threshold:
                self.calibrated_base_threshold = self.postprocessor.calibrate_threshold(np.asarray(y_val), val_proba)
            
            if self.apply_post and s_val is not None:
                self.postprocessor.fit(np.asarray(y_val), val_proba, np.asarray(s_val))
        else:
            tr_proba = self.classifier_manager.predict_proba(
                X_tr_proc, 
                cluster_labels=train_clusters, 
                cluster_responsibilities=train_resp
            )
            if self.calibrate_threshold:
                self.calibrated_base_threshold = self.postprocessor.calibrate_threshold(np.asarray(y_train), tr_proba)
            if self.apply_post:
                self.postprocessor.fit(np.asarray(y_train), tr_proba, np.asarray(s_train))

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generates continuous risk probability scores."""
        X_proc = self.preprocessor.transform(X)
        clusters = self.clusterer.predict(X_proc)
        responsibilities = self.clusterer.predict_proba(X_proc)
        return self.classifier_manager.predict_proba(
            X_proc, 
            cluster_labels=clusters, 
            cluster_responsibilities=responsibilities
        )

    def predict(self, X: pd.DataFrame, s: Optional[pd.Series] = None) -> np.ndarray:
        """Generates binary risk predictions."""
        proba = self.predict_proba(X)
        if self.apply_post and s is not None:
            return self.postprocessor.predict(proba, s)
        else:
            p1 = proba[:, 1]
            t = self.calibrated_base_threshold if self.calibrate_threshold else 0.5
            return (p1 >= t).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, s_test: pd.Series) -> Dict[str, float]:
        """Evaluates pipeline on test dataset across all metrics."""
        proba = self.predict_proba(X_test)
        preds = self.predict(X_test, s_test)
        metrics = compute_all_metrics(y_test, preds, proba, s_test)
        metrics['Configuration'] = self.name
        metrics['Pre_Processing'] = self.apply_pre
        metrics['In_Processing'] = self.apply_in
        metrics['Post_Processing'] = self.apply_post
        metrics['Adaptive_Clustering'] = self.adaptive_clustering
        metrics['Adaptive_Classifier'] = self.adaptive_classifier
        return metrics
