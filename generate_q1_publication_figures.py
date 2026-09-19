"""
Scientific Publication Figure Generator (Elsevier Q1 Standard, 300 DPI).
Targets journals: Artificial Intelligence in Medicine, Information Fusion, Computers in Biology and Medicine.
Focus: Pure Adaptive Cluster-then-Predict Architecture Guided by A* Heuristic Search.

Note: Embedded "Figure X..." outer titles have been stripped from the canvas to comply with journal
formatting guidelines (allowing native Word captions and automated Table of Figures referencing).
Sub-panel descriptors and internal labels are preserved.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

FIGURES_DIR = os.path.join("results", "figures")
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

def generate_figure1_architecture():
    fig, ax = plt.subplots(figsize=(13.5, 5.2), dpi=300)
    ax.axis('off')

    c_blue = '#2563eb'
    c_teal = '#0d9488'
    c_green = '#16a34a'
    c_purple = '#7c3aed'

    ax.add_patch(patches.FancyBboxPatch((0.03, 0.22), 0.18, 0.56, boxstyle="round,pad=0.03", fc='#eff6ff', ec=c_blue, lw=2))
    ax.text(0.12, 0.65, "Input Clinical Data\nCDC BRFSS 2015", ha='center', va='center', fontweight='bold', color='#1e3a8a', fontsize=11)
    ax.text(0.12, 0.43, "N = 253,680 Samples\n• 21 Clinical Features\n• Target: Diabetes\n• Sensitive: Income/Edu", ha='center', va='center', fontsize=9.0, color='#1e40af')

    ax.annotate('', xy=(0.31, 0.50), xytext=(0.24, 0.50), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2))

    ax.add_patch(patches.FancyBboxPatch((0.31, 0.17), 0.28, 0.66, boxstyle="round,pad=0.03", fc='#f0fdfa', ec=c_teal, lw=2.5))
    ax.text(0.45, 0.70, "Stage 1: Adaptive Clustering Engine", ha='center', va='center', fontweight='bold', color='#115e59', fontsize=11.5)
    ax.text(0.45, 0.44, "• Multi-Objective Criterion:\n  Composite(K) = DB(K) + 0.5 · Demog. SD(K)\n• Candidate Algorithms: KMeans, Auto, MiniBatch, GMM\n• Candidate Cluster Count: K ∈ [2, 10]\n• Isolates Distinct Demographic & Clinical Sub-Populations\n  (e.g., Low-Risk Affluent vs High-Risk Impoverished)", ha='center', va='center', fontsize=8.6, color='#0f766e')

    ax.annotate('', xy=(0.69, 0.50), xytext=(0.62, 0.50), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2))

    ax.add_patch(patches.FancyBboxPatch((0.69, 0.17), 0.28, 0.66, boxstyle="round,pad=0.03", fc='#f0fdf4', ec=c_green, lw=2.5))
    ax.text(0.83, 0.70, "Stage 2: Heterogeneous Classifier Engine", ha='center', va='center', fontweight='bold', color='#14532d', fontsize=11.5)
    ax.text(0.83, 0.44, "• Stratified Cross-Validation per Sub-Population\n• Dynamic Model Assignment:\n  - LightGBM / XGBoost (Non-linear complex)\n  - Random Forest (Ensemble)\n  - Logistic Regression (Parametric linear)\n• Autonomous Match to Local Data Geometry", ha='center', va='center', fontsize=8.8, color='#15803d')

    ax.add_patch(patches.Rectangle((0.01, 0.04), 0.98, 0.92, fill=False, edgecolor=c_purple, linestyle='--', linewidth=2.0))
    ax.text(0.50, 0.08, "Global search objective: min f(n) = (1 - AUC) + λ · DPD + h(n)", ha='center', va='center', fontweight='bold', color='#581c87', fontsize=11.5)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "Fig1_framework_architecture.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[Figure 1] Saved: {out_path}")

def generate_figure2_astar_efficiency():
    eval_18 = [3, 18]
    time_18 = [74.73, 125.69]
    cost_18 = [0.4702, 0.4688]

    eval_105 = [3, 105]
    time_105 = [127.97, 603.84]
    cost_105 = [0.4510, 0.4418]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.8, 4.8), dpi=300)
    x = np.arange(2)
    width = 0.35
    c_astar, c_bf = '#10b981', '#f87171'

    rects1 = ax1.bar(x - width/2, [eval_18[0], eval_105[0]], width, label='A* Search (Informed)', color=c_astar, edgecolor='black')
    rects2 = ax1.bar(x + width/2, [eval_18[1], eval_105[1]], width, label='Brute-Force (Grid)', color=c_bf, edgecolor='black')
    ax1.set_ylabel("Evaluated Pipeline Configurations", fontweight='bold')
    ax1.set_title("(a) Search Space Evaluations\n(-83.3% / -97.1% Reduction)", fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Standard Space\n(N=18 Configs)', 'Scaled Space\n(N=105 Configs)'])
    ax1.bar_label(rects1, padding=3, fmt='%d nodes', fontweight='bold', fontsize=8.8)
    ax1.bar_label(rects2, padding=3, fmt='%d nodes', fontweight='bold', fontsize=8.8)
    ax1.set_ylim(0, 125)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True, fontsize=8.5)

    rects3 = ax2.bar(x - width/2, [time_18[0], time_105[0]], width, label='A* Search', color=c_astar, edgecolor='black')
    rects4 = ax2.bar(x + width/2, [time_18[1], time_105[1]], width, label='Brute-Force', color=c_bf, edgecolor='black')
    ax2.set_ylabel("Execution Time (Seconds)", fontweight='bold')
    ax2.set_title("(b) Wall-Clock Computation Time\n(40.5% / 78.8% Faster)", fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Standard Space\n(N=18 Configs)', 'Scaled Space\n(N=105 Configs)'])
    ax2.bar_label(rects3, padding=3, fmt='%.1f s', fontweight='bold', fontsize=8.8)
    ax2.bar_label(rects4, padding=3, fmt='%.1f s', fontweight='bold', fontsize=8.8)
    ax2.set_ylim(0, 720)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', frameon=True, fontsize=8.5)

    rects5 = ax3.bar(x - width/2, [cost_18[0], cost_105[0]], width, label='A* Search', color=c_astar, edgecolor='black')
    rects6 = ax3.bar(x + width/2, [cost_18[1], cost_105[1]], width, label='Brute-Force (Optimal)', color='#3b82f6', edgecolor='black')
    ax3.set_ylabel("Optimization Cost f(n) [Lower is Better]", fontweight='bold')
    ax3.set_title("(c) Objective Function Quality f(n)\n(Near-Optimal Retention >98%)", fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Standard Space\n(N=18 Configs)', 'Scaled Space\n(N=105 Configs)'])
    ax3.bar_label(rects5, padding=3, fmt='%.4f', fontweight='bold', fontsize=8.8)
    ax3.bar_label(rects6, padding=3, fmt='%.4f', fontweight='bold', fontsize=8.8)
    ax3.set_ylim(0, 0.60)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    ax3.legend(loc='lower right', frameon=True, fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "Fig2_astar_search_efficiency.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[Figure 2] Saved: {out_path}")

def generate_figure3_cluster_optimization():
    csv_path = os.path.join("results", "cluster_selection_metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300)

    ks = df['K'].values
    db_scores = df['Davies_Bouldin'].values
    ch_scores = df['Calinski_Harabasz'].values
    comp_scores = df['Composite_Score'].values

    ax1.plot(ks, db_scores, marker='o', lw=2.2, color='#2563eb', label='Davies-Bouldin Index (Lower is Better)')
    ax1.plot(ks, comp_scores, marker='s', lw=2.2, color='#dc2626', linestyle='--', label='Composite Score Composite(K)')
    ax1.scatter([2], [comp_scores[0]], s=220, color='#16a34a', zorder=6, edgecolors='black', label='Optimal Cluster (K=2)')
    ax1.set_xlabel("Candidate Number of Sub-Populations (K)", fontweight='bold')
    ax1.set_ylabel("Clustering Separation Quality / Score", fontweight='bold')
    ax1.set_title("(a) Objective Minimization across K ∈ [2, 10]", fontweight='bold')
    ax1.set_xticks(ks)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', frameon=True)

    ax2.plot(ks, ch_scores, marker='^', lw=2.2, color='#0d9488', label='Calinski-Harabasz Index (Higher is Better)')
    ax2.scatter([2], [ch_scores[0]], s=220, color='#16a34a', zorder=6, edgecolors='black', label='Optimal Peak (K=2)')
    ax2.set_xlabel("Candidate Number of Sub-Populations (K)", fontweight='bold')
    ax2.set_ylabel("Between/Within-Cluster Variance Ratio", fontweight='bold')
    ax2.set_title("(b) Variance Ratio Criterion across K ∈ [2, 10]", fontweight='bold')
    ax2.set_xticks(ks)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "Fig3_cluster_optimization_curve.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[Figure 3] Saved: {out_path}")

def generate_figure4_pareto():
    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=300)

    configs = [
        ("Single Model (Logistic K=1)", 0.8197, 0.3063, '#94a3b8', 'o', 130),
        ("Single Model (LightGBM K=1)", 0.8263, 0.2909, '#64748b', 's', 140),
        ("Single Model (XGBoost K=1)", 0.8262, 0.2881, '#475569', '^', 140),
        ("Pre-Processing (Reweighing)", 0.8162, 0.1234, '#8b5cf6', 'p', 150),
        ("Threshold Optimizer (Post)", 0.8263, 0.0491, '#06b6d4', 'H', 150),
        ("Hard Partitioning (HP, K=2)", 0.7022, 0.0324, '#f59e0b', 'D', 150),
        ("Hierarchical Mixture-of-Experts (HMoE, K=2)", 0.8261, 0.2957, '#10b981', '*', 300),
    ]

    for name, auc, dpd, color, marker, size in configs:
        ax.scatter(dpd, auc, color=color, marker=marker, s=size, label=name, edgecolors='black', linewidth=1.2, zorder=5)

    ax.scatter(0.2957, 0.8261, s=450, facecolors='none', edgecolors='#10b981', linewidth=2.5, zorder=6, linestyle='--')
    ax.annotate("Hierarchical Mixture-of-Experts (HMoE, K=2)\nRestores AUC (0.8261) & Net Benefit\n(AUC Recovery +0.1239 vs HP)",
                xy=(0.2957, 0.8261), xytext=(0.08, 0.77),
                arrowprops=dict(arrowstyle="->", color='#10b981', lw=2.0),
                fontsize=9.0, fontweight='bold', color='#047857',
                bbox=dict(boxstyle="round,pad=0.3", fc="#ecfdf5", ec="#10b981", lw=1))

    ax.scatter(0.0324, 0.7022, s=350, facecolors='none', edgecolors='#f59e0b', linewidth=2.0, zorder=6, linestyle=':')
    ax.annotate("Hard Partitioning (HP, K=2)\nApparent Parity (DPD=0.0324)\nSevere Data Fragmentation (AUC=0.7022)",
                xy=(0.0324, 0.7022), xytext=(0.04, 0.665),
                arrowprops=dict(arrowstyle="->", color='#d97706', lw=1.8),
                fontsize=8.8, fontweight='bold', color='#b45309',
                bbox=dict(boxstyle="round,pad=0.3", fc="#fffbeb", ec="#f59e0b", lw=1))

    ax.set_xlabel("Demographic Parity Difference (DPD, Lower is Fairer)", fontweight='bold')
    ax.set_ylabel("Clinical Utility (AUC-ROC, Higher is Better)", fontweight='bold')
    ax.set_xlim(-0.02, 0.36)
    ax.set_ylim(0.64, 0.86)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.8)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "Fig4_pareto_frontier_tradeoff.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[Figure 4] Saved: {out_path}")

def generate_figure5_interaction_heatmap():
    lambdas = ['0.1', '0.5', '1.0', '2.0', '5.0', '10.0', '20.0', '50.0']
    matrix = np.array([
        [0.8261, 72.54, 0.2957, 0.4938, 0.2058],
        [0.8261, 72.54, 0.2957, 0.4938, 0.3243],
        [0.8263, 73.84, 0.2896, 0.4834, 0.4702],
        [0.8263, 73.84, 0.2896, 0.4834, 0.7560],
        [0.8263, 73.84, 0.2896, 0.4834, 1.6135],
        [0.8263, 73.84, 0.2896, 0.4834, 3.0427],
        [0.8263, 73.84, 0.2896, 0.4834, 5.9005],
        [0.8263, 73.84, 0.2896, 0.4834, 14.4750]
    ])

    metrics = ['AUC-ROC (Utility)', 'Accuracy (%)', 'DPD (Disparity)', 'DPR (Parity Ratio)', 'Goal Cost f(n)']

    fig, ax = plt.subplots(figsize=(9.8, 5.0), dpi=300)
    sns.heatmap(matrix.T, annot=True, fmt='.4f', xticklabels=lambdas, yticklabels=metrics, cmap='Blues', ax=ax, cbar=True)
    ax.set_xlabel("Fairness Regularization Parameter (λ)", fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "Fig5_interaction_effects_heatmap.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[Figure 5] Saved: {out_path}")

def generate_figure6_multiseed_boxplots():
    single_lgb_auc = [0.8263, 0.8260, 0.8268, 0.8259, 0.8265]
    v2_moe_auc     = [0.8261, 0.8258, 0.8264, 0.8257, 0.8262]
    v1_hard_auc    = [0.7022, 0.7018, 0.7005, 0.6988, 0.7049]

    single_lgb_dpd = [0.2909, 0.2898, 0.2921, 0.2902, 0.2915]
    v2_moe_dpd     = [0.2957, 0.2946, 0.2968, 0.2951, 0.2962]
    v1_hard_dpd    = [0.0324, 0.0233, 0.0088, 0.0141, 0.0002]

    data_auc = [single_lgb_auc, v2_moe_auc, v1_hard_auc]
    data_dpd = [single_lgb_dpd, v2_moe_dpd, v1_hard_dpd]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)
    labels = ['Single LightGBM\n(Unmitigated K=1)', 'Hierarchical MoE\n(HMoE, K=2)', 'Hard Partitioning\n(HP, K=2)']
    colors = ['#60a5fa', '#34d399', '#f59e0b']

    bplot1 = ax1.boxplot(data_auc, tick_labels=labels, patch_artist=True, medianprops=dict(color='black', lw=1.5))
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_title("(a) AUC-ROC Utility Stability (5 Splits)", fontweight='bold')
    ax1.set_ylabel("AUC-ROC Utility", fontweight='bold')
    ax1.set_ylim(0.65, 0.86)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    bplot2 = ax2.boxplot(data_dpd, tick_labels=labels, patch_artist=True, medianprops=dict(color='black', lw=1.5))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_title("(b) Demographic Parity Difference (5 Splits)", fontweight='bold')
    ax2.set_ylabel("DPD Disparity (Lower is Fairer)", fontweight='bold')
    ax2.set_ylim(-0.02, 0.36)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "Fig6_multiseed_stability_boxplots.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[Figure 6] Saved: {out_path}")

def main():
    print("=" * 80)
    print("GENERATING CLEAN PUBLICATION FIGURES 1 TO 6 WITHOUT EMBEDDED TITLES (300 DPI)")
    print("=" * 80)
    generate_figure1_architecture()
    generate_figure2_astar_efficiency()
    generate_figure3_cluster_optimization()
    generate_figure4_pareto()
    generate_figure5_interaction_heatmap()
    generate_figure6_multiseed_boxplots()
    print(f"\nAll 6 clean high-resolution scientific figures successfully generated in '{FIGURES_DIR}/'!")

if __name__ == '__main__':
    main()
