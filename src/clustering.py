"""
Stage 1: Clustering Engine and Cluster Disparity Profiling.
Supports K-Means, MiniBatchKMeans, Gaussian Mixture Models, and Adaptive Clustering (optimal K & fairness-balanced selection).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from typing import Dict, Any, Optional, List, Tuple

class PopulationClusterer:
    """
    Stage 1 Clusterer: Groups the population into sub-populations.
    Supports fixed clustering or Adaptive Clustering (automatic K search and fairness-balanced profiling).
    Evaluates candidate K across [k_min, k_max] (default: 2 to 10).
    """
    def __init__(
        self, 
        n_clusters: int = 4, 
        method: str = 'kmeans', 
        adaptive: bool = False,
        k_min: int = 2,
        k_max: int = 10,
        fairness_penalty_weight: float = 0.5,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.method = method.lower()
        self.adaptive = adaptive
        self.k_min = k_min
        self.k_max = k_max
        self.fairness_penalty_weight = fairness_penalty_weight
        self.random_state = random_state
        self.model = None
        self.best_k = n_clusters
        self.best_method = method
        self.search_history_: List[Dict[str, Any]] = []
        
        if not self.adaptive:
            self._init_model(self.n_clusters, self.method)

    def _init_model(self, k: int, method: str):
        if method == 'kmeans':
            self.model = KMeans(
                n_clusters=k, 
                random_state=self.random_state, 
                n_init=10
            )
        elif method == 'minibatch':
            self.model = MiniBatchKMeans(
                n_clusters=k, 
                random_state=self.random_state, 
                batch_size=2048,
                n_init=10
            )
        elif method in ['gmm', 'gaussian_mixture']:
            self.model = GaussianMixture(
                n_components=k, 
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported clustering method: {method}. Choose 'kmeans', 'minibatch', or 'gmm'.")

    def _find_optimal_clustering(
        self, 
        X: pd.DataFrame, 
        s: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[int, str]:
        """
        Evaluates candidate K values and methods to find optimal cluster configuration.
        Balances cluster separation (Davies-Bouldin Index) and demographic disparity across clusters.
        """
        # Subsample for fast evaluation if dataset is very large
        n_eval = min(len(X), 15000)
        if len(X) > n_eval:
            eval_idx = np.random.RandomState(self.random_state).choice(len(X), size=n_eval, replace=False)
            X_eval = X.iloc[eval_idx] if isinstance(X, pd.DataFrame) else X[eval_idx]
            s_eval = np.asarray(s)[eval_idx] if s is not None else None
            w_eval = sample_weight[eval_idx] if sample_weight is not None else None
        else:
            X_eval = X
            s_eval = np.asarray(s) if s is not None else None
            w_eval = sample_weight

        best_score = float('inf')
        selected_k = self.n_clusters
        selected_method = 'kmeans'
        
        # In adaptive mode, evaluate all candidate algorithms unless explicitly locked
        if self.method == 'auto' or self.adaptive:
            candidate_methods = ['kmeans', 'minibatch', 'gmm']
        else:
            candidate_methods = [self.method]
            
        self.search_history_ = []

        for m in candidate_methods:
            for k in range(self.k_min, self.k_max + 1):
                try:
                    if m == 'kmeans':
                        cand_model = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
                    elif m == 'minibatch':
                        cand_model = MiniBatchKMeans(n_clusters=k, random_state=self.random_state, batch_size=2048, n_init=5)
                    else:
                        cand_model = GaussianMixture(n_components=k, random_state=self.random_state)
                    
                    if m in ['kmeans', 'minibatch'] and w_eval is not None:
                        cand_model.fit(X_eval, sample_weight=w_eval)
                    else:
                        cand_model.fit(X_eval)

                    labels = cand_model.predict(X_eval)
                    
                    # Ensure valid cluster distribution
                    if len(np.unique(labels)) < 2:
                        continue

                    # Geometric quality metrics
                    db_score = davies_bouldin_score(X_eval, labels)
                    ch_score = calinski_harabasz_score(X_eval, labels)

                    # Demographic skew across clusters
                    demo_disparity = 0.0
                    if s_eval is not None:
                        cluster_s_means = [np.mean(s_eval[labels == c]) for c in range(k) if np.sum(labels == c) > 0]
                        if len(cluster_s_means) > 1:
                            demo_disparity = float(np.std(cluster_s_means))

                    # Composite criterion: Lower DB score is better, lower demo disparity is better
                    composite_score = db_score + (self.fairness_penalty_weight * demo_disparity)

                    self.search_history_.append({
                        'Method': m,
                        'K': k,
                        'Davies_Bouldin': round(db_score, 4),
                        'Calinski_Harabasz': round(ch_score, 1),
                        'Demographic_Std': round(demo_disparity, 4),
                        'Composite_Score': round(composite_score, 4)
                    })

                    if composite_score < best_score:
                        best_score = composite_score
                        selected_k = k
                        selected_method = m

                except Exception as e:
                    continue

        print(f"[AdaptiveClustering] Auto-selected optimal: Algorithm='{selected_method.upper()}', K={selected_k} (Best Composite Score: {best_score:.4f})")
        return selected_k, selected_method

    def fit_predict(
        self, 
        X: pd.DataFrame, 
        s: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fits clustering algorithm on data X with optional pre-processing sample weights,
        and returns cluster labels.
        """
        if self.adaptive:
            self.best_k, self.best_method = self._find_optimal_clustering(X, s=s, sample_weight=sample_weight)
            self.n_clusters = self.best_k
            self.method = self.best_method
            self._init_model(self.n_clusters, self.method)

        if self.method in ['kmeans', 'minibatch'] and sample_weight is not None:
            self.model.fit(X, sample_weight=sample_weight)
        else:
            self.model.fit(X)
        return self.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Assigns new samples to nearest cluster center."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame, temperature: float = 1.0) -> np.ndarray:
        """
        Computes soft cluster assignment responsibilities w_c(x) = P(c|x).
        - For GMM: returns true posterior probabilities.
        - For KMeans / MiniBatch: computes smooth softmax over negative squared Euclidean distances.
        
        Returns:
            responsibilities (np.ndarray): Shape (n_samples, n_clusters), sums to 1.0 across rows.
        """
        if self.n_clusters <= 1:
            return np.ones((len(X), 1), dtype=np.float64)

        if hasattr(self.model, 'predict_proba') and self.method in ['gmm', 'gaussian_mixture']:
            return self.model.predict_proba(X)
        
        if hasattr(self.model, 'cluster_centers_'):
            # KMeans / MiniBatch: calculate distance to each centroid
            centers = self.model.cluster_centers_
            X_arr = np.asarray(X)
            # Compute squared Euclidean distances: (N, K)
            # (x - c)^2 = x^2 - 2xc + c^2
            dists = np.zeros((len(X_arr), self.n_clusters), dtype=np.float64)
            for k in range(self.n_clusters):
                dists[:, k] = np.sum((X_arr - centers[k]) ** 2, axis=1)
            
            # Dynamic bandwidth sigma based on median distance
            median_dist = np.median(dists) + 1e-6
            gamma = 1.0 / (median_dist * temperature)
            
            # Softmax with numerical stability
            scaled_neg_dist = -gamma * dists
            scaled_neg_dist -= np.max(scaled_neg_dist, axis=1, keepdims=True)
            exp_dists = np.exp(scaled_neg_dist)
            weights = exp_dists / np.sum(exp_dists, axis=1, keepdims=True)
            return weights
        
        # Fallback: one-hot hard assignments
        labels = self.predict(X)
        one_hot = np.zeros((len(X), self.n_clusters), dtype=np.float64)
        for k in range(self.n_clusters):
            one_hot[labels == k, k] = 1.0
        return one_hot

    def analyze_cluster_demographics(
        self, 
        cluster_labels: np.ndarray, 
        s: pd.Series, 
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Analyzes the demographic and disease prevalence profile of each cluster.
        Quantifies socio-economic consolidation and bias propagation (Research Question 1).
        """
        df_analysis = pd.DataFrame({
            'Cluster': cluster_labels,
            'Protected_Privileged': np.asarray(s),
            'Diabetes_Target': np.asarray(y)
        })
        
        overall_priv_rate = df_analysis['Protected_Privileged'].mean()
        overall_target_rate = df_analysis['Diabetes_Target'].mean()
        
        summary = []
        for k in range(self.n_clusters):
            c_data = df_analysis[df_analysis['Cluster'] == k]
            n_k = len(c_data)
            if n_k == 0:
                continue
            priv_rate = c_data['Protected_Privileged'].mean()
            target_rate = c_data['Diabetes_Target'].mean()
            # Demographic representation ratio relative to overall
            rep_ratio = (priv_rate / (overall_priv_rate + 1e-9))
            
            summary.append({
                'Cluster_ID': k,
                'Sample_Count': n_k,
                'Cluster_Pct': n_k / len(df_analysis) * 100,
                'Privileged_Ratio': priv_rate,
                'Unprivileged_Ratio': 1.0 - priv_rate,
                'Demographic_Rep_Ratio': rep_ratio,
                'Diabetes_Prevalence': target_rate
            })
            
        return pd.DataFrame(summary)
