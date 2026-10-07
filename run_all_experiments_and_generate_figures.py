"""
Master Execution Pipeline:
1. Executes all empirical experiments from data input -> generates results/*.csv
2. Automatically executes figure generator -> renders results/figures/*.png & *.tiff
"""
import os
import sys
import time

def run_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("STARTING FULL END-TO-END SCIENTIFIC REPRODUCIBILITY PIPELINE")
    print("=" * 80)
    
    # 1. Cluster Selection Experiment
    print("\n[Step 1/6] Running Cluster Selection Experiment...")
    import run_cluster_selection_experiment
    run_cluster_selection_experiment.run_clustering_experiment()
    
    # 2. Search Efficiency Experiment
    print("\n[Step 2/6] Running A* Search Efficiency Benchmark...")
    import run_search_efficiency_experiment
    run_search_efficiency_experiment.run_search_efficiency_benchmark()
    
    # 3. Pareto Frontier Trade-Off Experiment
    print("\n[Step 3/6] Running Pareto Frontier Baselines...")
    import run_pareto_experiment
    run_pareto_experiment.run_pareto_tradeoff_experiment()
    
    # 4. Lambda Sweep Dynamics Experiment
    print("\n[Step 4/6] Running Lambda Sweep Experiment...")
    import run_lambda_sweep_experiment
    run_lambda_sweep_experiment.run_lambda_sweep_experiment()
    
    # 5. Multi-Seed Replication Stability Experiment
    print("\n[Step 5/6] Running Multi-Seed Stability Experiment...")
    import run_multiseed_stability_experiment
    run_multiseed_stability_experiment.run_multiseed_stability_benchmark()
    
    # 6. Generate All Publication Figures from CSV Datasets
    print("\n[Step 6/6] Generating Publication Figures (Fig 1-4 & Fig S1-S2) from CSV Datasets...")
    import generate_q1_publication_figures
    generate_q1_publication_figures.main()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("All experimental CSV datasets and publication figures are synchronized.")
    print("=" * 80)

if __name__ == '__main__':
    run_pipeline()
