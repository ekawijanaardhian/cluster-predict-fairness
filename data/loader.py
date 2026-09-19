"""
Dataset Loader for BRFSS 2015 Diabetes Risk Prediction.
Supports loading local CSV files for CDC BRFSS 2015 dataset.
"""

import os
import pandas as pd
from typing import Tuple, Optional

BRFSS_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 
    'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 
    'Age', 'Education', 'Income'
]
TARGET_COL = 'Diabetes_binary'
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_binary_health_indicators_BRFSS2015.csv')

def load_data(
    file_path: Optional[str] = None, 
    n_samples: Optional[int] = None, 
    random_state: int = 42,
    protected_attr: str = 'Income_Binary'
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads BRFSS dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file. If None, defaults to diabetes_binary_health_indicators_BRFSS2015.csv in data/
        n_samples: Optional number of samples to subsample from dataset. If None, loads all data.
        random_state: Random seed for sampling reproducibility.
        protected_attr: Name of binary protected attribute ('Income_Binary' or 'Education_Binary').
        
    Returns:
        X (pd.DataFrame): Features dataframe
        y (pd.Series): Binary target (Diabetes)
        s (pd.Series): Binary protected attribute (0: Disadvantaged/Unprivileged, 1: Privileged)
    """
    path = file_path or DEFAULT_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"[DataLoader] Dataset file not found at: {path}")

    print(f"[DataLoader] Loading BRFSS dataset from file: {path}")
    df = pd.read_csv(path)
    
    if n_samples is not None and len(df) > n_samples:
        print(f"[DataLoader] Subsampling {n_samples} samples (from {len(df)} total rows)...")
        df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
    

    if 'Income' in df.columns and 'Income_Binary' not in df.columns:
        df['Income_Binary'] = (df['Income'] >= 5).astype(int)
    

    if 'Education' in df.columns and 'Education_Binary' not in df.columns:
        df['Education_Binary'] = (df['Education'] >= 4).astype(int)
        
    if protected_attr not in df.columns:
        raise ValueError(f"Protected attribute '{protected_attr}' not found in dataset columns: {list(df.columns)}")
        
    y = df[TARGET_COL].astype(int)
    s = df[protected_attr].astype(int)
    
    feature_cols = [c for c in df.columns if c not in [TARGET_COL, 'Income_Binary', 'Education_Binary']]
    X = df[feature_cols]
    
    print(f"[DataLoader] Loaded shape: X={X.shape}, y={y.shape}, Target Positive Rate: {y.mean():.3f}, Privileged Group Ratio (s=1): {s.mean():.3f}")
    return X, y, s

if __name__ == '__main__':

    X, y, s = load_data()
    print("Columns:", list(X.columns))
    print(X.head())
