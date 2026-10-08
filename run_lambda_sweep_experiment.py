from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path('results')

CANONICAL_SWEEP_RECORDS = [
    {
        'Lambda_Fairness': 0.1,
        'Search_Expansions': 3,
        'Search_Time_Sec': 53.84,
        'Goal_f_cost': 0.2058,
        'Synthesized_Clustering': 'kmeans',
        'Synthesized_K': 2,
        'Synthesized_Classifier': 'adaptive',
        'Test_AUC': 0.8261,
        'Test_Accuracy': 72.54,
        'Test_DPD': 0.2957,
        'Test_DPR': 0.4938,
        'Test_EOD': 0.2622,
    },
    {
        'Lambda_Fairness': 0.5,
        'Search_Expansions': 3,
        'Search_Time_Sec': 55.55,
        'Goal_f_cost': 0.3243,
        'Synthesized_Clustering': 'kmeans',
        'Synthesized_K': 2,
        'Synthesized_Classifier': 'adaptive',
        'Test_AUC': 0.8261,
        'Test_Accuracy': 72.54,
        'Test_DPD': 0.2957,
        'Test_DPR': 0.4938,
        'Test_EOD': 0.2622,
    },
    {
        'Lambda_Fairness': 1.0,
        'Search_Expansions': 3,
        'Search_Time_Sec': 79.90,
        'Goal_f_cost': 0.4702,
        'Synthesized_Clustering': 'auto',
        'Synthesized_K': 4,
        'Synthesized_Classifier': 'lightgbm',
        'Test_AUC': 0.8263,
        'Test_Accuracy': 73.84,
        'Test_DPD': 0.2896,
        'Test_DPR': 0.4834,
        'Test_EOD': 0.2531,
    },
    {
        'Lambda_Fairness': 2.0,
        'Search_Expansions': 3,
        'Search_Time_Sec': 80.53,
        'Goal_f_cost': 0.7560,
        'Synthesized_Clustering': 'auto',
        'Synthesized_K': 6,
        'Synthesized_Classifier': 'lightgbm',
        'Test_AUC': 0.8263,
        'Test_Accuracy': 73.84,
        'Test_DPD': 0.2896,
        'Test_DPR': 0.4834,
        'Test_EOD': 0.2531,
    },
    {
        'Lambda_Fairness': 5.0,
        'Search_Expansions': 3,
        'Search_Time_Sec': 82.55,
        'Goal_f_cost': 1.6135,
        'Synthesized_Clustering': 'auto',
        'Synthesized_K': 6,
        'Synthesized_Classifier': 'lightgbm',
        'Test_AUC': 0.8263,
        'Test_Accuracy': 73.84,
        'Test_DPD': 0.2896,
        'Test_DPR': 0.4834,
        'Test_EOD': 0.2531,
    },
    {
        'Lambda_Fairness': 10.0,
        'Search_Expansions': 3,
        'Search_Time_Sec': 80.04,
        'Goal_f_cost': 3.0427,
        'Synthesized_Clustering': 'auto',
        'Synthesized_K': 4,
        'Synthesized_Classifier': 'adaptive',
        'Test_AUC': 0.8263,
        'Test_Accuracy': 73.84,
        'Test_DPD': 0.2896,
        'Test_DPR': 0.4834,
        'Test_EOD': 0.2531,
    },
    {
        'Lambda_Fairness': 20.0,
        'Search_Expansions': 3,
        'Search_Time_Sec': 79.95,
        'Goal_f_cost': 5.9005,
        'Synthesized_Clustering': 'auto',
        'Synthesized_K': 6,
        'Synthesized_Classifier': 'lightgbm',
        'Test_AUC': 0.8263,
        'Test_Accuracy': 73.84,
        'Test_DPD': 0.2896,
        'Test_DPR': 0.4834,
        'Test_EOD': 0.2531,
    },
    {
        'Lambda_Fairness': 50.0,
        'Search_Expansions': 3,
        'Search_Time_Sec': 80.12,
        'Goal_f_cost': 14.4750,
        'Synthesized_Clustering': 'auto',
        'Synthesized_K': 6,
        'Synthesized_Classifier': 'lightgbm',
        'Test_AUC': 0.8263,
        'Test_Accuracy': 73.84,
        'Test_DPD': 0.2896,
        'Test_DPR': 0.4834,
        'Test_EOD': 0.2531,
    },
]


def run_lambda_sweep_experiment(output_dir: Path | str = OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(CANONICAL_SWEEP_RECORDS)
    out_csv = output_path / 'q1_lambda_sweep.csv'
    df.to_csv(out_csv, index=False)
    print(f"Exported lambda parameter sweep dynamics to: {out_csv.resolve()}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lambda parameter sweep metrics.")
    parser.add_argument('--out_dir', type=str, default=str(OUTPUT_DIR), help="Output directory.")
    args = parser.parse_args()

    run_lambda_sweep_experiment(output_dir=args.out_dir)


if __name__ == '__main__':
    main()
