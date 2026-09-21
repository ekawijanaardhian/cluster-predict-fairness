import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class DataPreprocessor:

    def __init__(self, scale_features: bool=True):
        self.scale_features = scale_features
        self.scaler = StandardScaler()
        self.feature_names = None
        self.weights_dict = {}

    def compute_reweighing_weights(self, s: pd.Series, y: pd.Series) -> np.ndarray:
        s_arr = np.asarray(s).astype(int)
        y_arr = np.asarray(y).astype(int)
        n = len(s_arr)
        p_s0 = np.mean(s_arr == 0)
        p_s1 = np.mean(s_arr == 1)
        p_y0 = np.mean(y_arr == 0)
        p_y1 = np.mean(y_arr == 1)
        p_s0_y0 = np.mean((s_arr == 0) & (y_arr == 0)) + 1e-09
        p_s0_y1 = np.mean((s_arr == 0) & (y_arr == 1)) + 1e-09
        p_s1_y0 = np.mean((s_arr == 1) & (y_arr == 0)) + 1e-09
        p_s1_y1 = np.mean((s_arr == 1) & (y_arr == 1)) + 1e-09
        w_s0_y0 = p_s0 * p_y0 / p_s0_y0
        w_s0_y1 = p_s0 * p_y1 / p_s0_y1
        w_s1_y0 = p_s1 * p_y0 / p_s1_y0
        w_s1_y1 = p_s1 * p_y1 / p_s1_y1
        self.weights_dict = {(0, 0): w_s0_y0, (0, 1): w_s0_y1, (1, 0): w_s1_y0, (1, 1): w_s1_y1}
        weights = np.ones(n, dtype=np.float64)
        for s_val in (0, 1):
            for y_val in (0, 1):
                mask = (s_arr == s_val) & (y_arr == y_val)
                weights[mask] = self.weights_dict[s_val, y_val]
        return weights

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.feature_names = X.columns
        if self.scale_features:
            X_scaled = self.scaler.fit_transform(X)
            return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        return X.copy()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scale_features:
            X_scaled = self.scaler.transform(X)
            return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        return X.copy()
