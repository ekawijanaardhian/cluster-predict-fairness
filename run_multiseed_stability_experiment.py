from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from data.loader import load_data
from src.pipeline import ClusterThenPredictPipeline
from src.metrics import compute_all_metrics

DEFAULT_SEEDS = (42, 101, 202, 303, 404)
OUTPUT_DIR = Path('results')


def evaluate_split(
    X: pd.DataFrame,
    y: pd.Series,
    s: pd.Series,
    seed: int,
    seed_idx: int,
) -> list[dict]:
    X_train_val, X_test, y_train_val, y_test, s_train_val, s_test = train_test_split(
        X, y, s, test_size=0.20, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
        X_train_val, y_train_val, s_train_val, test_size=0.125, random_state=seed, stratify=y_train_val
    )

    models = {
        'Single_LightGBM': ClusterThenPredictPipeline(
            name='Single_LightGBM',
            n_clusters=1,
            clustering_method='kmeans',
            classifier_type='lightgbm',
            random_state=seed,
        ),
        'Hierarchical_MoE_K2': ClusterThenPredictPipeline(
            name='Hierarchical_MoE_K2',
            n_clusters=2,
            clustering_method='kmeans',
            classifier_type='lightgbm',
            use_global_residual=True,
            soft_assignment=True,
            calibrate_clusters=True,
            random_state=seed,
        ),
        'Hard_Partitioning_K2': ClusterThenPredictPipeline(
            name='Hard_Partitioning_K2',
            n_clusters=2,
            clustering_method='kmeans',
            classifier_type='lightgbm',
            use_global_residual=False,
            soft_assignment=False,
            calibrate_clusters=False,
            random_state=seed,
        ),
    }

    records = []
    y_true = y_test.to_numpy(dtype=int)
    s_attr = s_test.to_numpy(dtype=int)

    for arch_name, pipe in models.items():
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train, s_train, X_val, y_val, s_val)
        y_prob = pipe.predict_proba(X_test)
        y_pred = pipe.predict(X_test, s_test)
        elapsed = time.perf_counter() - t0

        m = compute_all_metrics(y_true, y_pred, y_prob, s_attr)
        auc = round(float(m['AUC_ROC']), 4)
        dpd = round(float(m['Demographic_Parity_Diff']), 4)

        print(f"  [{arch_name:<20}] seed={seed} -> AUC={auc:.4f}, DPD={dpd:.4f} ({elapsed:.1f}s)")
        records.append({
            'Architecture': arch_name,
            'Seed_Index': seed_idx,
            'Test_AUC': auc,
            'Test_DPD': dpd,
        })

    return records


def run_multiseed_stability_benchmark(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_samples: int | None = None,
    output_dir: Path | str = OUTPUT_DIR,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading BRFSS cohort (seeds={list(seeds)}, n_samples={n_samples or 'all'})...")
    X, y, s = load_data(n_samples=n_samples)

    all_records: list[dict] = []
    total_start = time.perf_counter()

    for idx, seed in enumerate(seeds, start=1):
        print(f"\n[Split {idx}/{len(seeds)}] Evaluating models with random seed {seed}...")
        split_records = evaluate_split(X, y, s, seed=seed, seed_idx=idx)
        all_records.extend(split_records)

    df_results = pd.DataFrame(all_records)
    out_csv = output_path / 'q1_multiseed_baselines_raw.csv'
    df_results.to_csv(out_csv, index=False)

    total_time = time.perf_counter() - total_start
    print(f"\nCompleted {len(seeds)} splits in {total_time:.1f}s.")
    print(f"Results saved to: {out_csv.resolve()}\n")
    print(df_results.groupby('Architecture')[['Test_AUC', 'Test_DPD']].agg(['mean', 'std']).round(4))

    return df_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed stability evaluation on BRFSS 2015.")
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS), help="Random seeds to evaluate.")
    parser.add_argument('--n_samples', type=int, default=None, help="Optional sample limit for quick smoke testing.")
    parser.add_argument('--out_dir', type=str, default=str(OUTPUT_DIR), help="Output directory path.")
    args = parser.parse_args()

    run_multiseed_stability_benchmark(
        seeds=args.seeds,
        n_samples=args.n_samples,
        output_dir=args.out_dir,
    )


if __name__ == '__main__':
    main()
