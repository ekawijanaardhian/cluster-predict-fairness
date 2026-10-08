from __future__ import annotations

import argparse
import time

import run_cluster_selection_experiment
import run_search_efficiency_experiment
import run_pareto_experiment
import run_lambda_sweep_experiment
import run_multiseed_stability_experiment
import generate_q1_publication_figures


def run_pipeline(seeds=(42, 101, 202, 303, 404), fast: bool = False) -> None:
    t_start = time.perf_counter()
    n_samples = 10000 if fast else None

    print("[1/6] Running cluster selection experiment...")
    run_cluster_selection_experiment.run_clustering_experiment()

    print("\n[2/6] Compiling A* search efficiency benchmark...")
    run_search_efficiency_experiment.run_search_efficiency_benchmark()

    print("\n[3/6] Exporting Pareto frontier trade-off baselines...")
    run_pareto_experiment.run_pareto_tradeoff_experiment()

    print("\n[4/6] Exporting lambda parameter sweep dynamics...")
    run_lambda_sweep_experiment.run_lambda_sweep_experiment()

    print("\n[5/6] Running multi-seed stability benchmark...")
    run_multiseed_stability_experiment.run_multiseed_stability_benchmark(seeds=seeds, n_samples=n_samples)

    print("\n[6/6] Generating publication figures...")
    generate_q1_publication_figures.main()

    elapsed = time.perf_counter() - t_start
    print(f"\nPipeline completed in {elapsed:.1f}s. All outputs synchronized.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete experimental replication pipeline.")
    parser.add_argument('--fast', action='store_true', help="Use subsampled cohort for fast smoke test.")
    args = parser.parse_args()

    run_pipeline(fast=args.fast)


if __name__ == '__main__':
    main()
