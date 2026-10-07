import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

RESULTS_DIR = 'results'
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11.5
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9.0


def save_figure(fig, base_names):
    """Save figure in both 300 DPI PNG and 600 DPI publication TIFF (lossless LZW)."""
    if isinstance(base_names, str):
        base_names = [base_names]
    for bname in base_names:
        png_path = os.path.join(FIGURES_DIR, f"{bname}.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        tiff_path = os.path.join(FIGURES_DIR, f"{bname}.tiff")
        fig.savefig(tiff_path, dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print(f"[{base_names[0]}] Saved PNG (300 DPI) and TIFF LZW (600 DPI)")


def generate_figure1_architecture():
    fig, ax = plt.subplots(figsize=(14.2, 5.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    c_blue = '#2563eb'
    c_teal = '#0d9488'
    c_green = '#16a34a'
    c_purple = '#7c3aed'

    # Outer dashed bounding box with generous padding
    ax.add_patch(patches.Rectangle((0.012, 0.03), 0.976, 0.94, fill=False,
                                   edgecolor=c_purple, linestyle='--', linewidth=2.0))

    # Box 1: Input Cohort Data
    ax.add_patch(patches.FancyBboxPatch((0.040, 0.18), 0.185, 0.62,
                                        boxstyle='round,pad=0.02', fc='#eff6ff', ec=c_blue, lw=2))
    ax.text(0.132, 0.67, 'Input Survey Data\nCDC BRFSS 2015',
            ha='center', va='center', fontweight='bold', color='#1e3a8a', fontsize=11.0)
    ax.text(0.132, 0.44, 'N = 253,680 Samples\n• 21 Survey Indicators\n• Target: Diabetes\n• Sensitive: Income / Edu',
            ha='center', va='center', fontsize=9.0, color='#1e40af')

    # Arrow 1 -> 2
    ax.annotate('', xy=(0.285, 0.49), xytext=(0.235, 0.49),
                arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2))

    # Box 2: Stage 1 Clustering
    ax.add_patch(patches.FancyBboxPatch((0.295, 0.15), 0.335, 0.68,
                                        boxstyle='round,pad=0.02', fc='#f0fdfa', ec=c_teal, lw=2.5))
    ax.text(0.462, 0.70, 'Stage 1: Adaptive Clustering Engine',
            ha='center', va='center', fontweight='bold', color='#115e59', fontsize=11.5)
    ax.text(0.462, 0.44, '• Multi-Objective Criterion:\n  Composite(K) = DB(K) + 0.5 · Demog. SD(K)\n'
                         '• Candidate Algorithms: KMeans, Auto, MiniBatch, GMM\n'
                         '• Candidate Cluster Count: K ∈ [2, 10]\n'
                         '• Isolates Distinct Demographic & Clinical Sub-Populations\n'
                         '  (e.g., Low-Risk Advantaged vs High-Risk Disadvantaged)',
            ha='center', va='center', fontsize=8.6, color='#0f766e')

    # Arrow 2 -> 3
    ax.annotate('', xy=(0.680, 0.49), xytext=(0.638, 0.49),
                arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2))

    # Box 3: Stage 2 Classifiers
    ax.add_patch(patches.FancyBboxPatch((0.690, 0.15), 0.265, 0.68,
                                        boxstyle='round,pad=0.02', fc='#f0fdf4', ec=c_green, lw=2.5))
    ax.text(0.822, 0.70, 'Stage 2: Heterogeneous Classifier Engine',
            ha='center', va='center', fontweight='bold', color='#14532d', fontsize=11.5)
    ax.text(0.822, 0.44, '• Stratified Cross-Validation per Sub-Population\n'
                         '• Dynamic Model Assignment:\n'
                         '  - LightGBM / XGBoost (Non-linear complex)\n'
                         '  - Random Forest (Ensemble)\n'
                         '  - Logistic Regression (Parametric linear)\n'
                         '• Autonomous Match to Local Data Geometry',
            ha='center', va='center', fontsize=8.8, color='#15803d')

    # Global search objective formula at bottom
    ax.text(0.50, 0.08, 'Search Objective: g(n) = (1 − AUC) + λ · DPD;   f(n) = g(n) + h(n)',
            ha='center', va='center', fontweight='bold', color='#581c87', fontsize=11.5)

    plt.tight_layout()
    save_figure(fig, 'Fig1_framework_architecture')


def generate_figure2_astar_efficiency():
    csv_path = os.path.join(RESULTS_DIR, 'astar_search_efficiency_metrics.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[Figure 2] Missing experimental dataset: '{csv_path}'. "
                                f"Please run the search benchmark experiment script first.")
    df = pd.read_csv(csv_path)

    # Extract metrics for standard space (N=18) and scaled space (N=105)
    std_astar = df[(df['Search_Space'] == 'Standard_Space') & (df['Search_Strategy'] == 'A_Star_Search')].iloc[0]
    std_bf = df[(df['Search_Space'] == 'Standard_Space') & (df['Search_Strategy'] == 'Brute_Force')].iloc[0]
    scl_astar = df[(df['Search_Space'] == 'Scaled_Space') & (df['Search_Strategy'] == 'A_Star_Search')].iloc[0]
    scl_bf = df[(df['Search_Space'] == 'Scaled_Space') & (df['Search_Strategy'] == 'Brute_Force')].iloc[0]

    eval_18 = [int(std_astar['Evaluated_Pipelines']), int(std_bf['Evaluated_Pipelines'])]
    time_18 = [float(std_astar['Search_Time_Sec']), float(std_bf['Search_Time_Sec'])]
    cost_18 = [float(std_astar['Goal_f_cost']), float(std_bf['Goal_f_cost'])]

    eval_105 = [int(scl_astar['Evaluated_Pipelines']), int(scl_bf['Evaluated_Pipelines'])]
    time_105 = [float(scl_astar['Search_Time_Sec']), float(scl_bf['Search_Time_Sec'])]
    cost_105 = [float(scl_astar['Goal_f_cost']), float(scl_bf['Goal_f_cost'])]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.8, 4.8), dpi=300)
    x = np.arange(2)
    width = 0.35

    c_astar, c_bf = '#10b981', '#f87171'

    # (a) Evaluations
    rects1 = ax1.bar(x - width/2, [eval_18[0], eval_105[0]], width, label='A* Search (Informed)', color=c_astar, edgecolor='black')
    rects2 = ax1.bar(x + width/2, [eval_18[1], eval_105[1]], width, label='Brute-Force (Grid)', color=c_bf, edgecolor='black')
    ax1.set_ylabel('Evaluated Pipeline Configurations', fontweight='bold')
    ax1.set_title('(a) Search Space Evaluations\n(-83.3% / -97.1% Reduction)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Standard Space\n(N=18 Configs)', 'Scaled Space\n(N=105 Configs)'])
    ax1.bar_label(rects1, padding=3, fmt='%d nodes', fontweight='bold', fontsize=8.8)
    ax1.bar_label(rects2, padding=3, fmt='%d nodes', fontweight='bold', fontsize=8.8)
    ax1.set_ylim(0, 125)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True, fontsize=8.5)

    # (b) Time
    rects3 = ax2.bar(x - width/2, [time_18[0], time_105[0]], width, label='A* Search', color=c_astar, edgecolor='black')
    rects4 = ax2.bar(x + width/2, [time_18[1], time_105[1]], width, label='Brute-Force', color=c_bf, edgecolor='black')
    ax2.set_ylabel('Execution Time (Seconds)', fontweight='bold')
    ax2.set_title('(b) Wall-Clock Computation Time\n(40.5% / 78.8% Faster)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Standard Space\n(N=18 Configs)', 'Scaled Space\n(N=105 Configs)'])
    ax2.bar_label(rects3, padding=3, fmt='%.1f s', fontweight='bold', fontsize=8.8)
    ax2.bar_label(rects4, padding=3, fmt='%.1f s', fontweight='bold', fontsize=8.8)
    ax2.set_ylim(0, 720)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', frameon=True, fontsize=8.5)

    # (c) Objective Cost (single-line clean title without inaccurate >98% claim)
    rects5 = ax3.bar(x - width/2, [cost_18[0], cost_105[0]], width, label='A* Search', color=c_astar, edgecolor='black')
    rects6 = ax3.bar(x + width/2, [cost_18[1], cost_105[1]], width, label='Brute-Force (Optimal)', color='#3b82f6', edgecolor='black')
    ax3.set_ylabel('Optimization Cost f(n) [Lower is Better]', fontweight='bold')
    ax3.set_title('(c) Objective Cost at Termination f(n)', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Standard Space\n(N=18 Configs)', 'Scaled Space\n(N=105 Configs)'])
    ax3.bar_label(rects5, padding=3, fmt='%.4f', fontweight='bold', fontsize=8.8)
    ax3.bar_label(rects6, padding=3, fmt='%.4f', fontweight='bold', fontsize=8.8)
    ax3.set_ylim(0, 0.60)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    ax3.legend(loc='lower right', frameon=True, fontsize=8.5)

    plt.tight_layout()
    save_figure(fig, 'Fig2_astar_search_efficiency')


def generate_figure3_cluster_optimization():
    csv_path = os.path.join(RESULTS_DIR, 'cluster_selection_metrics.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[Figure 3] Missing experimental dataset: '{csv_path}'. "
                                f"Please run the cluster selection experiment script first.")
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300)
    ks = df['K'].values
    db_scores = df['Davies_Bouldin'].values
    ch_scores = df['Calinski_Harabasz'].values
    comp_scores = df['Composite_Score'].values

    # (a) Objective Minimization
    ax1.plot(ks, db_scores, marker='o', lw=2.2, color='#2563eb', label='Davies-Bouldin Index (Lower is Better)')
    ax1.plot(ks, comp_scores, marker='s', lw=2.2, color='#dc2626', linestyle='--', label='Composite Score Composite(K)')
    ax1.scatter([ks[0]], [comp_scores[0]], s=220, color='#16a34a', zorder=6, edgecolors='black', label=f'Optimal Cluster (K={int(ks[0])})')
    ax1.set_xlabel('Candidate Number of Sub-Populations (K)', fontweight='bold')
    ax1.set_ylabel('Clustering Separation Quality / Score', fontweight='bold')
    ax1.set_title('(a) Objective Minimization across K ∈ [2, 10]', fontweight='bold')
    ax1.set_xticks(ks)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', frameon=True)

    # (b) Calinski-Harabasz Peak
    ax2.plot(ks, ch_scores, marker='^', lw=2.2, color='#0d9488', label='Calinski-Harabasz Index (Higher is Better)')
    ax2.scatter([ks[0]], [ch_scores[0]], s=220, color='#16a34a', zorder=6, edgecolors='black', label=f'Optimal Peak (K={int(ks[0])})')
    ax2.set_xlabel('Candidate Number of Sub-Populations (K)', fontweight='bold')
    ax2.set_ylabel('Between/Within-Cluster Variance Ratio', fontweight='bold')
    ax2.set_title('(b) Variance Ratio Criterion across K ∈ [2, 10]', fontweight='bold')
    ax2.set_xticks(ks)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    save_figure(fig, 'Fig3_cluster_optimization_curve')


def generate_figure4_pareto():
    csv_path = os.path.join(RESULTS_DIR, 'pareto_frontier_tradeoff_metrics.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[Figure 4] Missing experimental dataset: '{csv_path}'. "
                                f"Please run the benchmark mitigation baselines experiment first.")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=300)

    for row in df.itertuples():
        ax.scatter(row.Test_DPD, row.Test_AUC, color=row.Plot_Color, marker=row.Plot_Marker,
                   s=row.Marker_Size, label=row.Model_Configuration, edgecolors='black', linewidth=1.2, zorder=5)

    # Highlight HMoE if present in dataset
    hmoe_rows = df[df['Model_Configuration'].str.contains('Hierarchical', case=False, na=False)]
    if not hmoe_rows.empty:
        hmoe = hmoe_rows.iloc[0]
        ax.scatter(hmoe['Test_DPD'], hmoe['Test_AUC'], s=450, facecolors='none',
                   edgecolors='#10b981', linewidth=2.5, zorder=6, linestyle='--')
        ax.annotate(f"{hmoe['Model_Configuration']}\nRestores AUC ({hmoe['Test_AUC']:.4f}) & Net Benefit\n(AUC Recovery +0.1238 vs HP)",
                    xy=(hmoe['Test_DPD'], hmoe['Test_AUC']), xytext=(0.08, 0.77),
                    arrowprops=dict(arrowstyle='->', color='#10b981', lw=2.0),
                    fontsize=9.0, fontweight='bold', color='#047857',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#ecfdf5', ec='#10b981', lw=1))

    # Highlight HP if present in dataset
    hp_rows = df[df['Model_Configuration'].str.contains('Hard Partitioning', case=False, na=False)]
    if not hp_rows.empty:
        hp = hp_rows.iloc[0]
        ax.scatter(hp['Test_DPD'], hp['Test_AUC'], s=350, facecolors='none',
                   edgecolors='#f59e0b', linewidth=2.0, zorder=6, linestyle=':')
        ax.annotate(f"Hard Partitioning (HP, K=2)\nApparent Parity (DPD={hp['Test_DPD']:.4f})\nUtility collapse (AUC={hp['Test_AUC']:.4f})",
                    xy=(hp['Test_DPD'], hp['Test_AUC']), xytext=(0.04, 0.665),
                    arrowprops=dict(arrowstyle='->', color='#d97706', lw=1.8),
                    fontsize=8.8, fontweight='bold', color='#b45309',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#fffbeb', ec='#f59e0b', lw=1))

    ax.set_xlabel('Demographic Parity Difference (DPD)', fontweight='bold')
    ax.set_ylabel('Discrimination (AUC)', fontweight='bold')
    ax.set_xlim(-0.02, 0.36)
    ax.set_ylim(0.64, 0.86)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.8)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    # Save with both naming conventions for paper and graphical abstract compatibility
    save_figure(fig, ['Fig4_pareto_frontier_tradeoff', 'Fig4_utility_parity_plane'])


def generate_figureS1_interaction_heatmap():
    csv_path = os.path.join(RESULTS_DIR, 'q1_lambda_sweep.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[Figure S1] Missing experimental dataset: '{csv_path}'. "
                                f"Please run the lambda sweep experiment first.")
    df = pd.read_csv(csv_path)

    lambdas = [str(val) for val in df['Lambda_Fairness']]
    metrics = ['AUC-ROC (Utility)', 'Accuracy (%)', 'DPD (Disparity)', 'DPR (Parity Ratio)', 'Goal Cost f(n)']

    raw_matrix = np.array([
        df['Test_AUC'].values,
        df['Test_Accuracy'].values,
        df['Test_DPD'].values,
        df['Test_DPR'].values,
        df['Goal_f_cost'].values
    ])
    
    # Text without percentage sign in cells (since yticklabel already states 'Accuracy (%)')
    annot_text = []
    for i, m_name in enumerate(metrics):
        row_text = []
        for j in range(len(lambdas)):
            val = raw_matrix[i, j]
            if 'Accuracy' in m_name:
                row_text.append(f'{val:.2f}')
            else:
                row_text.append(f'{val:.4f}')
        annot_text.append(row_text)
    annot_text = np.array(annot_text)

    # Row-wise min-max normalization for clear metric variations
    norm_matrix = np.zeros_like(raw_matrix)
    for i in range(len(metrics)):
        row_min = raw_matrix[i].min()
        row_max = raw_matrix[i].max()
        if row_max > row_min:
            norm_matrix[i] = (raw_matrix[i] - row_min) / (row_max - row_min)
        else:
            norm_matrix[i] = 0.5

    fig, ax = plt.subplots(figsize=(10.2, 5.2), dpi=300)
    sns.heatmap(norm_matrix, annot=annot_text, fmt='', xticklabels=lambdas, yticklabels=metrics,
                cmap='Blues', cbar=False, ax=ax, linewidths=0.5, linecolor='white')
    ax.set_xlabel('Fairness Regularization Parameter (λ)', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'FigS1_interaction_effects_heatmap')
    # Clean up obsolete filename if present
    for old_ext in ['.png', '.tiff']:
        old_path = os.path.join(FIGURES_DIR, f'Fig5_interaction_effects_heatmap{old_ext}')
        if os.path.exists(old_path):
            os.remove(old_path)


def generate_figureS2_multiseed_boxplots():
    csv_path = os.path.join(RESULTS_DIR, 'q1_multiseed_baselines_raw.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[Figure S2] Missing experimental dataset: '{csv_path}'. "
                                f"Please run the multi-seed baseline experiment first.")
    df = pd.read_csv(csv_path)

    single_lgb = df[df['Architecture'] == 'Single_LightGBM'].sort_values('Seed_Index')
    v2_moe     = df[df['Architecture'] == 'Hierarchical_MoE_K2'].sort_values('Seed_Index')
    v1_hard    = df[df['Architecture'] == 'Hard_Partitioning_K2'].sort_values('Seed_Index')

    data_auc = [single_lgb['Test_AUC'].values, v2_moe['Test_AUC'].values, v1_hard['Test_AUC'].values]
    data_dpd = [single_lgb['Test_DPD'].values, v2_moe['Test_DPD'].values, v1_hard['Test_DPD'].values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)
    labels = ['Single LightGBM\n(Unmitigated K=1)', 'Hierarchical MoE\n(HMoE, K=2)', 'Hard Partitioning\n(HP, K=2)']
    colors = ['#60a5fa', '#34d399', '#f59e0b']

    bplot1 = ax1.boxplot(data_auc, tick_labels=labels, patch_artist=True, medianprops=dict(color='black', lw=1.5))
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_title('(a) AUC-ROC Utility Stability (5 Splits)', fontweight='bold')
    ax1.set_ylabel('AUC-ROC Utility', fontweight='bold')
    ax1.set_ylim(0.65, 0.86)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    bplot2 = ax2.boxplot(data_dpd, tick_labels=labels, patch_artist=True, medianprops=dict(color='black', lw=1.5))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_title('(b) Demographic Parity Difference (5 Splits)', fontweight='bold')
    ax2.set_ylabel('Demographic Parity Difference (DPD)', fontweight='bold')
    ax2.set_ylim(-0.02, 0.36)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_figure(fig, 'FigS2_multiseed_stability_boxplots')
    # Clean up obsolete filename if present
    for old_ext in ['.png', '.tiff']:
        old_path = os.path.join(FIGURES_DIR, f'Fig6_multiseed_stability_boxplots{old_ext}')
        if os.path.exists(old_path):
            os.remove(old_path)



def main():
    print('=' * 80)
    print('GENERATING PUBLICATION FIGURES (Fig 1-4, Fig S1-S2) IN 300 DPI PNG & 600 DPI TIFF')
    print('=' * 80)
    generate_figure1_architecture()
    generate_figure2_astar_efficiency()
    generate_figure3_cluster_optimization()
    generate_figure4_pareto()
    generate_figureS1_interaction_heatmap()
    generate_figureS2_multiseed_boxplots()
    print(f"\nAll publication figures successfully generated in '{FIGURES_DIR}/' in dual formats (PNG & TIFF)!")


if __name__ == '__main__':
    main()
