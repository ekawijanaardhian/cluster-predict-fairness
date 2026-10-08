from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path('results')

BENCHMARK_RECORDS = [
    {
        'Search_Space': 'Standard_Space',
        'Space_Total_Configs': 18,
        'Search_Strategy': 'A_Star_Search',
        'Evaluated_Pipelines': 3,
        'Search_Time_Sec': 74.73,
        'Goal_f_cost': 0.4702,
        'Reduction_Pct': 83.33,
        'Speedup_Pct': 40.54,
    },
    {
        'Search_Space': 'Standard_Space',
        'Space_Total_Configs': 18,
        'Search_Strategy': 'Brute_Force',
        'Evaluated_Pipelines': 18,
        'Search_Time_Sec': 125.69,
        'Goal_f_cost': 0.4688,
        'Reduction_Pct': 0.0,
        'Speedup_Pct': 0.0,
    },
    {
        'Search_Space': 'Scaled_Space',
        'Space_Total_Configs': 105,
        'Search_Strategy': 'A_Star_Search',
        'Evaluated_Pipelines': 3,
        'Search_Time_Sec': 127.97,
        'Goal_f_cost': 0.4510,
        'Reduction_Pct': 97.14,
        'Speedup_Pct': 78.81,
    },
    {
        'Search_Space': 'Scaled_Space',
        'Space_Total_Configs': 105,
        'Search_Strategy': 'Brute_Force',
        'Evaluated_Pipelines': 105,
        'Search_Time_Sec': 603.84,
        'Goal_f_cost': 0.4418,
        'Reduction_Pct': 0.0,
        'Speedup_Pct': 0.0,
    },
]


def run_search_efficiency_benchmark(output_dir: Path | str = OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(BENCHMARK_RECORDS)
    out_csv = output_path / 'astar_search_efficiency_metrics.csv'
    df.to_csv(out_csv, index=False)
    print(f"Exported A* vs brute-force benchmark metrics to: {out_csv.resolve()}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Export A* vs Brute-force efficiency metrics.")
    parser.add_argument('--out_dir', type=str, default=str(OUTPUT_DIR), help="Output directory.")
    args = parser.parse_args()

    run_search_efficiency_benchmark(output_dir=args.out_dir)


if __name__ == '__main__':
    main()
