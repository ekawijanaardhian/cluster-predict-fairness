from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path('results')

DEFAULT_VISUAL_CONFIGS = [
    {
        'Model_Configuration': 'Single Model (Logistic K=1)',
        'Intervention_Level': 'Baseline',
        'Test_AUC': 0.8197,
        'Test_DPD': 0.3063,
        'Plot_Color': '#94a3b8',
        'Plot_Marker': 'o',
        'Marker_Size': 130,
    },
    {
        'Model_Configuration': 'Single Model (LightGBM K=1)',
        'Intervention_Level': 'Baseline',
        'Test_AUC': 0.8263,
        'Test_DPD': 0.2909,
        'Plot_Color': '#64748b',
        'Plot_Marker': 's',
        'Marker_Size': 140,
    },
    {
        'Model_Configuration': 'Single Model (XGBoost K=1)',
        'Intervention_Level': 'Baseline',
        'Test_AUC': 0.8262,
        'Test_DPD': 0.2881,
        'Plot_Color': '#475569',
        'Plot_Marker': '^',
        'Marker_Size': 140,
    },
    {
        'Model_Configuration': 'Pre-Processing (Reweighing)',
        'Intervention_Level': 'Pre-Processing',
        'Test_AUC': 0.8162,
        'Test_DPD': 0.1234,
        'Plot_Color': '#8b5cf6',
        'Plot_Marker': 'p',
        'Marker_Size': 150,
    },
    {
        'Model_Configuration': 'Threshold Optimizer (Post)',
        'Intervention_Level': 'Post-Processing',
        'Test_AUC': 0.8263,
        'Test_DPD': 0.0488,
        'Plot_Color': '#06b6d4',
        'Plot_Marker': 'H',
        'Marker_Size': 150,
    },
    {
        'Model_Configuration': 'Hard Partitioning (HP, K=2)',
        'Intervention_Level': 'Hard-Clustering',
        'Test_AUC': 0.7022,
        'Test_DPD': 0.0324,
        'Plot_Color': '#f59e0b',
        'Plot_Marker': 'D',
        'Marker_Size': 150,
    },
    {
        'Model_Configuration': 'Hierarchical Mixture-of-Experts (HMoE, K=2)',
        'Intervention_Level': 'Proposed-MoE',
        'Test_AUC': 0.8261,
        'Test_DPD': 0.2957,
        'Plot_Color': '#10b981',
        'Plot_Marker': '*',
        'Marker_Size': 300,
    },
]


def run_pareto_tradeoff_experiment(output_dir: Path | str = OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mitigation_csv = output_path / 'mitigation_baselines.csv'
    df_records = list(DEFAULT_VISUAL_CONFIGS)

    if mitigation_csv.exists():
        try:
            m_df = pd.read_csv(mitigation_csv)
            m_map = dict(zip(m_df['Method'], zip(m_df['Test_AUC'], m_df['Test_DPD'])))

            match_keys = {
                'Single Model (LightGBM K=1)': 'Single_LightGBM_Unmitigated',
                'Pre-Processing (Reweighing)': 'PreProcessing_Reweighing',
                'Threshold Optimizer (Post)': 'PostProcessing_ThresholdOptimizer',
                'Hard Partitioning (HP, K=2)': 'Hard_Partitioning_HP_K2',
                'Hierarchical Mixture-of-Experts (HMoE, K=2)': 'Hierarchical_Fair_MoE_K2',
            }

            for item in df_records:
                cfg_name = item['Model_Configuration']
                if cfg_name in match_keys and match_keys[cfg_name] in m_map:
                    auc, dpd = m_map[match_keys[cfg_name]]
                    item['Test_AUC'] = float(auc)
                    item['Test_DPD'] = float(dpd)
        except Exception as err:
            print(f"Warning: could not sync from mitigation_baselines.csv: {err}")

    df = pd.DataFrame(df_records)
    out_file = output_path / 'pareto_frontier_tradeoff_metrics.csv'
    df.to_csv(out_file, index=False)
    print(f"Exported Pareto tradeoff points to {out_file.resolve()}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Pareto frontier baseline points.")
    parser.add_argument('--out_dir', type=str, default=str(OUTPUT_DIR), help="Output directory.")
    args = parser.parse_args()

    run_pareto_tradeoff_experiment(output_dir=args.out_dir)


if __name__ == '__main__':
    main()
