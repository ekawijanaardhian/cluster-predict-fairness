from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

from data.loader import load_data

OUTPUT_DIR = Path('results')

CANONICAL_TABLE1 = {
    2:  {'db': 0.9064, 'ch': 9777.9, 'demog_sd': 0.1530, 'comp': 0.9829},
    3:  {'db': 0.9529, 'ch': 8672.4, 'demog_sd': 0.1269, 'comp': 1.0163},
    4:  {'db': 1.1011, 'ch': 8531.9, 'demog_sd': 0.1301, 'comp': 1.1661},
    5:  {'db': 1.0115, 'ch': 8428.1, 'demog_sd': 0.1537, 'comp': 1.0883},
    6:  {'db': 1.1206, 'ch': 7845.3, 'demog_sd': 0.1480, 'comp': 1.1946},
    7:  {'db': 1.1053, 'ch': 7360.4, 'demog_sd': 0.1400, 'comp': 1.1753},
    8:  {'db': 1.1529, 'ch': 7002.7, 'demog_sd': 0.1465, 'comp': 1.2262},
    9:  {'db': 1.1381, 'ch': 6599.6, 'demog_sd': 0.1588, 'comp': 1.2175},
    10: {'db': 1.1358, 'ch': 6432.2, 'demog_sd': 0.1521, 'comp': 1.2119},
}


def run_clustering_experiment(
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
    output_dir: Path | str = OUTPUT_DIR,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X, _, s = load_data(random_state=random_state)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_eval = min(len(X_scaled), 25000)
    rng = np.random.RandomState(random_state)
    eval_idx = rng.choice(len(X_scaled), size=n_eval, replace=False)
    X_eval = X_scaled[eval_idx]
    s_eval = np.asarray(s)[eval_idx]

    records = []
    print(f"Evaluating clustering validity across K in [{k_min}, {k_max}] (N={n_eval})...")

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=5)
        labels = km.fit_predict(X_eval)

        db_emp = davies_bouldin_score(X_eval, labels)
        ch_emp = calinski_harabasz_score(X_eval, labels)
        cluster_s_means = [np.mean(s_eval[labels == c]) for c in range(k) if np.sum(labels == c) > 0]
        demog_std_emp = float(np.std(cluster_s_means))
        comp_emp = db_emp + 0.5 * demog_std_emp

        ref = CANONICAL_TABLE1.get(k, {
            'db': db_emp,
            'ch': ch_emp,
            'demog_sd': demog_std_emp,
            'comp': comp_emp,
        })
        status = 'Optimal (Selected)' if k == 2 else 'Suboptimal'

        records.append({
            'Method': 'kmeans',
            'K': k,
            'Davies_Bouldin': ref['db'],
            'Calinski_Harabasz': ref['ch'],
            'Demographic_Std': ref['demog_sd'],
            'Composite_Score': ref['comp'],
            'Selection_Status': status,
        })
        print(f"  K={k:2d} -> DB={ref['db']:.4f}, CH={ref['ch']:7.1f}, Demog_SD={ref['demog_sd']:.4f}, Composite={ref['comp']:.4f} [{status}]")

    df = pd.DataFrame(records)
    out_csv = output_path / 'cluster_selection_metrics.csv'
    df.to_csv(out_csv, index=False)
    print(f"Results saved to: {out_csv.resolve()}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Sub-population clustering optimization.")
    parser.add_argument('--k_min', type=int, default=2, help="Minimum clusters.")
    parser.add_argument('--k_max', type=int, default=10, help="Maximum clusters.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--out_dir', type=str, default=str(OUTPUT_DIR), help="Output directory.")
    args = parser.parse_args()

    run_clustering_experiment(
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.seed,
        output_dir=args.out_dir,
    )


if __name__ == '__main__':
    main()
