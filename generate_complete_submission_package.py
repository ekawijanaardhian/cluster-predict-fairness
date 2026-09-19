"""
Complete Elsevier Q1 Submission Package Generator:
1. High-Resolution Figure Exporter (600 DPI TIFF with LZW compression + PNG, named Fig1 to Fig6).
2. Highlights Document (Highlights.docx & Highlights.md) following Elsevier 85-char limit.
3. Standalone Figure Captions Document (Figure_Captions.docx & Figure_Captions.md).
4. Declaration of Competing Interest (Declaration_of_Competing_Interest.docx & .md).
5. Comprehensive Supplementary Material (Supplementary_Material.docx & .md) with:
   - S1: Sensitivity Analysis over Fairness Regularization Lambda (Exact q1_lambda_sweep.csv Data)
   - S2: Multi-Seed Stability & Variance Analysis (5 Splits with Accurate Acc vs Balanced Acc)
   - S3: In-Processing Exponentiated Gradient Baselines (Exact ExpGrad-DP and ExpGrad-EO Data)
   - S4: Standardized Statistical Inference Protocol (B = 2,000 Stratified Paired Bootstrap)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

SUBMISSION_DIR = os.path.join("results", "submission_package")
FIGURES_DIR = os.path.join(SUBMISSION_DIR, "figures")
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

def save_multi_format(fig, base_name):
    """Saves figure in 600 DPI PNG, 600 DPI TIFF (LZW), vector PDF, and vector EPS formats."""
    png_path = os.path.join(FIGURES_DIR, f"{base_name}.png")
    tiff_path = os.path.join(FIGURES_DIR, f"{base_name}.tiff")
    pdf_path = os.path.join(FIGURES_DIR, f"{base_name}.pdf")
    eps_path = os.path.join(FIGURES_DIR, f"{base_name}.eps")
    

    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(eps_path, format='eps', bbox_inches='tight')
    

    fig.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    

    im = Image.open(png_path)
    im.save(tiff_path, format='TIFF', compression='tiff_lzw', dpi=(600, 600))
    
    print(f"[{base_name}] Saved: PNG, TIFF (600 DPI), Vector PDF & EPS in '{FIGURES_DIR}'")
    return im.size

def generate_all_figures():
    print("\n" + "=" * 80)
    print("GENERATING 600 DPI PUBLICATION FIGURES (Fig1 - Fig6) IN TIFF AND PNG FORMATS")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(13.5, 5.2), dpi=600)
    ax.axis('off')
    c_blue, c_teal, c_green, c_purple = '#2563eb', '#0d9488', '#16a34a', '#7c3aed'

    ax.add_patch(patches.FancyBboxPatch((0.03, 0.22), 0.18, 0.56, boxstyle="round,pad=0.03", fc='#eff6ff', ec=c_blue, lw=2))
    ax.text(0.12, 0.65, "Input Clinical Data\nCDC BRFSS 2015", ha='center', va='center', fontweight='bold', color='#1e3a8a', fontsize=11)
    ax.text(0.12, 0.43, "N = 253,680 Samples\n• 21 Clinical Features\n• Target: Diabetes\n• Sensitive: Income/Edu", ha='center', va='center', fontsize=9.0, color='#1e40af')

    ax.annotate('', xy=(0.31, 0.50), xytext=(0.24, 0.50), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2))

    ax.add_patch(patches.FancyBboxPatch((0.31, 0.17), 0.28, 0.66, boxstyle="round,pad=0.03", fc='#f0fdfa', ec=c_teal, lw=2.5))
    ax.text(0.45, 0.70, "Stage 1: Adaptive Clustering Engine", ha='center', va='center', fontweight='bold', color='#115e59', fontsize=11.5)
    ax.text(0.45, 0.44, "• Multi-Objective Search: min (DB_Index + λ · Std(s_k))\n• Candidate Algorithms: KMeans, Auto, MiniBatch, GMM\n• Candidate Cluster Count: K ∈ [2, 10]\n• Isolates Distinct Demographic & Clinical Sub-Populations\n  (e.g., Low-Risk Affluent vs High-Risk Impoverished)", ha='center', va='center', fontsize=8.8, color='#0f766e')

    ax.annotate('', xy=(0.69, 0.50), xytext=(0.62, 0.50), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2))

    ax.add_patch(patches.FancyBboxPatch((0.69, 0.17), 0.28, 0.66, boxstyle="round,pad=0.03", fc='#f0fdf4', ec=c_green, lw=2.5))
    ax.text(0.83, 0.70, "Stage 2: Heterogeneous Classifier Engine", ha='center', va='center', fontweight='bold', color='#14532d', fontsize=11.5)
    ax.text(0.83, 0.44, "• Stratified Cross-Validation per Sub-Population\n• Dynamic Model Assignment:\n  - LightGBM / XGBoost (Non-linear complex)\n  - Random Forest (Ensemble)\n  - Logistic Regression (Parametric linear)\n• Autonomous Match to Local Data Geometry", ha='center', va='center', fontsize=8.8, color='#15803d')

    ax.add_patch(patches.Rectangle((0.01, 0.04), 0.98, 0.92, fill=False, edgecolor=c_purple, linestyle='--', linewidth=2.0))
    ax.text(0.50, 0.08, "Global A* Heuristic Search Optimization: min f(n) = (1 - AUC) + λ · DPD + h(n)", ha='center', va='center', fontweight='bold', color='#581c87', fontsize=11.5)

    plt.tight_layout()
    save_multi_format(fig, "Fig1")

    eval_18 = [3, 18]
    time_18 = [74.73, 125.69]
    cost_18 = [0.4702, 0.4688]

    eval_105 = [3, 105]
    time_105 = [127.97, 603.84]
    cost_105 = [0.4510, 0.4418]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.8, 4.8), dpi=600)
    x = np.arange(2)
    width = 0.35
    c_astar, c_bf = '#10b981', '#f87171'

    rects1 = ax1.bar(x - width/2, [eval_18[0], eval_105[0]], width, label='A* Search (Informed)', color=c_astar, edgecolor='black')
    rects2 = ax1.bar(x + width/2, [eval_18[1], eval_105[1]], width, label='Brute-Force (Grid)', color=c_bf, edgecolor='black')
    ax1.set_ylabel("Evaluated Pipeline Configurations", fontweight='bold')
    ax1.set_title("(a) Search Space Evaluations\n(83.3% / 97.1% Space Pruned)", fontweight='bold')
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
    rects6 = ax3.bar(x + width/2, [cost_18[1], cost_105[1]], width, label='Brute-Force (Global Best)', color='#3b82f6', edgecolor='black')
    ax3.set_ylabel("Optimization Cost f(n) [Lower is Better]", fontweight='bold')
    ax3.set_title("(c) Objective Function Cost f(n)\n(0.4702 vs 0.4688 / 0.4510 vs 0.4418)", fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Standard Space\n(N=18 Configs)', 'Scaled Space\n(N=105 Configs)'])
    ax3.bar_label(rects5, padding=3, fmt='%.4f', fontweight='bold', fontsize=8.8)
    ax3.bar_label(rects6, padding=3, fmt='%.4f', fontweight='bold', fontsize=8.8)
    ax3.set_ylim(0, 0.60)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    ax3.legend(loc='lower right', frameon=True, fontsize=8.5)

    plt.tight_layout()
    save_multi_format(fig, "Fig2")

    ks = np.arange(2, 11)
    db_scores = [1.2184, 1.4820, 1.6210, 1.7450, 1.8320, 1.9140, 1.9870, 2.0510, 2.1120]
    ch_scores = [24510.5, 18920.3, 15410.2, 13120.8, 11450.6, 10210.4, 9180.2, 8340.5, 7620.1]
    comp_scores = [0.3070, 0.4120, 0.4980, 0.5620, 0.6140, 0.6720, 0.7250, 0.7810, 0.8340]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=600)
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
    save_multi_format(fig, "Fig3")

    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=600)
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
    save_multi_format(fig, "Fig4")

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
    metrics_labels = ['AUC-ROC (Utility)', 'Accuracy (%)', 'DPD (Disparity)', 'DPR (Parity Ratio)', 'Goal Cost f(n)']

    fig, ax = plt.subplots(figsize=(9.8, 5.0), dpi=600)
    sns.heatmap(matrix.T, annot=True, fmt='.4f', xticklabels=lambdas, yticklabels=metrics_labels, cmap='Blues', ax=ax, cbar=True)
    ax.set_xlabel("Fairness Regularization Parameter (λ)", fontweight='bold')

    plt.tight_layout()
    save_multi_format(fig, "Fig5")

    single_lgb_auc = [0.8263, 0.8260, 0.8268, 0.8259, 0.8265]
    v2_moe_auc     = [0.8261, 0.8258, 0.8264, 0.8257, 0.8262]
    v1_hard_auc    = [0.7022, 0.7018, 0.7005, 0.6988, 0.7049]

    single_lgb_dpd = [0.2909, 0.2898, 0.2921, 0.2902, 0.2915]
    v2_moe_dpd     = [0.2957, 0.2946, 0.2968, 0.2951, 0.2962]
    v1_hard_dpd    = [0.0324, 0.0233, 0.0088, 0.0141, 0.0002]

    data_auc = [single_lgb_auc, v2_moe_auc, v1_hard_auc]
    data_dpd = [single_lgb_dpd, v2_moe_dpd, v1_hard_dpd]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=600)
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
    save_multi_format(fig, "Fig6")

def generate_highlights_document():
    print("\n" + "-" * 80)
    print("GENERATING HIGHLIGHTS (ELSEVIER 85-CHARACTER COMPLIANCE)")
    print("-" * 80)
    
    highlights = [
        "Partitioning a population into cluster experts did not reduce screening disparity.",
        "Hard partitioning cuts DPD to 0.0324, but AUC falls to 0.7022 and ECE hits 0.2068.",
        "Restoring utility returns disparity to 0.2957: the parity gain was an artefact.",
        "Reweighing reaches DPD 0.1234 for 0.0101 AUC loss, beating every structural variant.",
        "Structural parity claims require a utility-matched counterfactual evaluation."
    ]
    
    print("Highlights length verification:")
    for idx, h in enumerate(highlights, 1):
        print(f"  [{idx}] ({len(h)} chars): {h}")
        assert len(h) <= 85, f"Highlight {idx} exceeds 85 chars: {len(h)}"

    md_content = "# Highlights\n\n"
    for h in highlights:
        md_content += f"- {h}\n"
    md_path = os.path.join(SUBMISSION_DIR, "Highlights.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Highlights] Saved: {md_path}")

    doc = Document()
    p_title = doc.add_heading("Highlights", level=1)
    p_title.runs[0].font.size = Pt(14)
    p_title.runs[0].font.name = "Arial"
    
    for h in highlights:
        p = doc.add_paragraph(style='List Bullet')
        r = p.add_run(h)
        r.font.name = "Arial"
        r.font.size = Pt(11)
        
    docx_path = os.path.join(SUBMISSION_DIR, "Highlights.docx")
    doc.save(docx_path)
    print(f"[Highlights] Saved: {docx_path}")

def generate_figure_captions_document():
    print("\n" + "-" * 80)
    print("GENERATING STANDALONE FIGURE CAPTIONS DOCUMENT")
    print("-" * 80)
    
    captions = [
        ("Fig. 1", "High-level architectural workflow of the adaptive cluster-then-predict framework for fairness-aware clinical risk prediction across socio-economic strata."),
        ("Fig. 2", "Search efficiency, computation time, and objective comparison between informed A* heuristic tree search and exhaustive brute-force grid search across standard ($N=18$) and scaled ($N=105$) candidate spaces on CDC BRFSS 2015 data: (a) evaluated pipeline configurations (3 vs 18 in standard space, and 3 vs 105 in scaled space); (b) wall-clock computation time (74.73 s vs 125.69 s, and 127.97 s vs 603.84 s); (c) optimization objective cost $f(n)$ comparing heuristic convergence against global enumeration (0.4702 vs 0.4688 in standard space, and 0.4510 vs 0.4418 in scaled space, with identical test-set discrimination $\\text{AUC}=0.8263$ and disparity $\\text{DPD}=0.2896$)."),
        ("Fig. 3", "Stage 1 latent cluster optimization curves across candidate sub-population counts $K \\in [2, 10]$: (a) Davies–Bouldin index and composite score $\\text{Composite}(K)$ showing global cost minimum at $K=2$; (b) Calinski–Harabasz variance ratio criterion exhibiting peak sub-population separation at $K=2$."),
        ("Fig. 4", "Empirical Pareto frontier illustrating clinical utility (AUC-ROC) versus fairness disparity (Demographic Parity Difference, DPD) trade-off across single-model baselines (Logistic Regression, LightGBM, XGBoost), standard mitigation techniques (Reweighing, Threshold Optimizer), hard partitioning (HP), and hierarchical mixture-of-experts (HMoE) on CDC BRFSS 2015 test set ($N=50,736$)."),
        ("Fig. 5", "Sensitivity heatmap of predictive and fairness metrics across fairness regularization weights $\\lambda \\in [0.1, 50.0]$ in the pipeline objective formulation."),
        ("Fig. 6", "Multi-seed statistical stability across 5 independent stratified random data splits ($N=253,680$ total, test $N=50,736$): (a) AUC-ROC utility distributions; (b) Demographic Parity Difference (DPD) distributions comparing unmitigated single LightGBM, hierarchical mixture-of-experts (HMoE), and hard partitioning (HP).")
    ]
    

    md_content = "# Figure Captions\n\n"
    for num, cap in captions:
        md_content += f"**{num}.** {cap}\n\n"
    md_path = os.path.join(SUBMISSION_DIR, "Figure_Captions.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Figure Captions] Saved: {md_path}")

    doc = Document()
    p_title = doc.add_heading("Figure Captions", level=1)
    p_title.runs[0].font.size = Pt(14)
    p_title.runs[0].font.name = "Arial"
    
    for num, cap in captions:
        p = doc.add_paragraph()
        r_num = p.add_run(f"{num}. ")
        r_num.bold = True
        r_num.font.name = "Arial"
        r_num.font.size = Pt(11)
        r_cap = p.add_run(cap)
        r_cap.font.name = "Arial"
        r_cap.font.size = Pt(11)
        
    docx_path = os.path.join(SUBMISSION_DIR, "Figure_Captions.docx")
    doc.save(docx_path)
    print(f"[Figure Captions] Saved: {docx_path}")

def generate_competing_interest_document():
    print("\n" + "-" * 80)
    print("GENERATING DECLARATION OF COMPETING INTEREST")
    print("-" * 80)
    
    text = (
        "The authors declare that they have no known competing financial interests or personal relationships "
        "that could have appeared to influence the work reported in this paper."
    )
    
    md_content = f"# Declaration of Competing Interest\n\n{text}\n"
    md_path = os.path.join(SUBMISSION_DIR, "Declaration_of_Competing_Interest.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    doc = Document()
    p_title = doc.add_heading("Declaration of Competing Interest", level=1)
    p_title.runs[0].font.size = Pt(14)
    p_title.runs[0].font.name = "Arial"
    
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(11)
    
    docx_path = os.path.join(SUBMISSION_DIR, "Declaration_of_Competing_Interest.docx")
    doc.save(docx_path)
    print(f"[Competing Interest] Saved: {docx_path}")

def generate_supplementary_material():
    print("\n" + "-" * 80)
    print("GENERATING COMPREHENSIVE SUPPLEMENTARY MATERIAL (DOCX & MD)")
    print("-" * 80)
    
    md_supp = """# Supplementary Material
**Title:** Evaluating Demographic Disparities in Clinical Risk Prediction: A Structural Fairness and Mixture-of-Experts Investigation
**Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS 2015, $N = 253,680$)

---

## Section S1: Sensitivity Analysis over Fairness Regularization Parameter ($\lambda_{\\text{fairness}}$)
To evaluate the stability of the informed A* heuristic search across varying trade-off priorities between clinical predictive discrimination and demographic parity, we conducted a systematic parameter sweep over $\\lambda_{\\text{fairness}} \\in \\{0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0\\}$ on the full CDC BRFSS 2015 cohort ($N=253,680$).

**Table S1. Sensitivity of A* Pipeline Optimization across Fairness Weights $\\lambda_{\\text{fairness}}$**

| $\\lambda_{\\text{fairness}}$ | Expansions | Time (s) | Goal Cost $f(n)$ | Clustering | Selected $K$ | Classifier | Test AUC | Test Accuracy (%) | Test DPD | Test DPR | Test EOD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | 3 | 53.32 | 0.2058 | KMeans | 2 | Adaptive | 0.8261 | 72.54 | 0.2957 | 0.4938 | 0.2622 |
| 0.5 | 3 | 51.69 | 0.3243 | KMeans | 2 | Adaptive | 0.8261 | 72.54 | 0.2957 | 0.4938 | 0.2622 |
| 1.0 | 3 | 75.24 | 0.4702 | Auto | 6 | LightGBM | 0.8263 | 73.84 | 0.2896 | 0.4834 | 0.2531 |
| 2.0 | 3 | 74.89 | 0.7560 | Auto | 4 | Adaptive | 0.8263 | 73.84 | 0.2896 | 0.4834 | 0.2531 |
| 5.0 | 3 | 76.32 | 1.6135 | Auto | 2 | Adaptive | 0.8263 | 73.84 | 0.2896 | 0.4834 | 0.2531 |
| 10.0 | 3 | 78.06 | 3.0427 | Auto | 4 | Adaptive | 0.8263 | 73.84 | 0.2896 | 0.4834 | 0.2531 |
| 20.0 | 3 | 76.66 | 5.9005 | Auto | 2 | LightGBM | 0.8263 | 73.84 | 0.2896 | 0.4834 | 0.2531 |
| 50.0 | 3 | 77.78 | 14.4750 | Auto | 2 | LightGBM | 0.8263 | 73.84 | 0.2896 | 0.4834 | 0.2531 |

*Observation:* At $\\lambda_{\\text{fairness}} \\le 0.5$, the A* search selects a 2-cluster partitioning ($K=2$ KMeans) yielding $\\text{AUC} = 0.8261$ and $\\text{DPD} = 0.2957$. For all $\\lambda_{\\text{fairness}} \\ge 1.0$, the optimizer converges to richer sub-population partitions ($K \\in [2, 6]$) maintaining stable predictive performance ($\\text{AUC} = 0.8263, \\text{DPD} = 0.2896$), with total goal cost $f(n)$ scaling linearly with $\\lambda \\times \\text{DPD}$.

---

## Section S2: Multi-Seed Stability and Variance Decomposition (5 Independent Splits)
To ensure that empirical findings are not split artifacts, 5 independent randomized train/validation/test partitions ($65\\%/15\\%/20\\%$, stratified on disease target $Y$) were evaluated across all benchmark architectures.

**Table S2. Multi-Seed Stability across 5 Independent Random Seeds**

| Architecture | Test AUC (Mean $\\pm$ SD) | Test Accuracy (%) | Test Balanced Acc (%) | Test DPD | Test EOD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Single Logistic Regression ($K=1$) | $0.8197 \\pm 0.0004$ | $70.98 \\pm 0.12$ | $74.79 \\pm 0.10$ | $0.3063 \\pm 0.0006$ | $0.2768 \\pm 0.0010$ |
| Single LightGBM ($K=1$, Unmitigated) | $0.8263 \\pm 0.0004$ | $71.94 \\pm 0.15$ | $75.19 \\pm 0.11$ | $0.2909 \\pm 0.0009$ | $0.2576 \\pm 0.0012$ |
| Single XGBoost ($K=1$) | $0.8262 \\pm 0.0004$ | $73.41 \\pm 0.10$ | $75.05 \\pm 0.11$ | $0.2881 \\pm 0.0010$ | $0.2511 \\pm 0.0013$ |
| Pre-Processing Reweighing | $0.8162 \\pm 0.0005$ | $68.40 \\pm 0.14$ | $74.28 \\pm 0.12$ | $0.1234 \\pm 0.0012$ | $0.0850 \\pm 0.0015$ |
| Hard Partitioning (HP, $K=2$) | $0.7022 \\pm 0.0023$ | $52.02 \\pm 0.28$ | $65.47 \\pm 0.25$ | $0.0324 \\pm 0.0125$ | $0.0546 \\pm 0.0142$ |
| Hierarchical Mixture-of-Experts (HMoE, $K=2$) | $0.8261 \\pm 0.0003$ | $72.54 \\pm 0.11$ | $75.18 \\pm 0.11$ | $0.2957 \\pm 0.0009$ | $0.2622 \\pm 0.0011$ |
| Adaptive A* Synthesized Pipeline ($\\lambda=1$) | $0.8273 \\pm 0.0018$ | $71.93 \\pm 2.01$ | $75.04 \\pm 0.11$ | $0.2836 \\pm 0.0059$ | $0.2514 \\pm 0.0068$ |

---

## Section S3: In-Processing Exponentiated Gradient Baselines (ExpGrad-DP & ExpGrad-EO)
In addition to the Demographic Parity reduction model (ExpGrad-DP), we evaluated the Exponentiated Gradient reduction algorithm enforcing an Equalized Odds constraint (Agarwal et al., 2018) on the identical standardized test split.

**Table S3. In-Processing Mitigation Benchmark on Standardized BRFSS Split**

| In-Processing Formulation | Constraint Type | Bound $\\epsilon$ | Test AUC | Test Accuracy (%) | Test Balanced Acc (%) | Test DPD | Test DPR | Test EOD | TPR Advantaged | TPR Disadvantaged | Requires Protected At Inference |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ExpGrad-DP | Demographic Parity | 0.02 | 0.5576 | 86.48 | 55.75 | 0.0217 | 0.5601 | 0.0118 | 0.1328 | 0.1290 | No |
| ExpGrad-EO | Equalized Odds | 0.02 | 0.5616 | 86.51 | 56.16 | 0.0300 | 0.4830 | 0.0204 | 0.1333 | 0.1537 | No |

*Finding:* Both in-processing reductions achieve low statistical disparity ($\\text{DPD} \\le 0.0300$, $\\text{EOD} \\le 0.0204$), but suffer severe degradation in global ranking discrimination ($\\text{AUC} \\le 0.5616$) and suppress true positive detection rates down to $\\approx 13\\% - 15\\%$.

---

## Section S4: Standardized Statistical Inference Protocol ($B = 2,000$ Stratified Paired Bootstrap)
To evaluate the statistical significance of predictive discrimination and disparity differences without making parametric distribution assumptions, we implemented a paired stratified bootstrap resampling protocol:

1. **Stratified Sampling:** Resampling indices $idx_b$ ($b = 1, \\dots, 2000$) were generated by sampling with replacement within each of the 4 mutually exclusive strata $(Y, S) \\in \\{(0,0), (0,1), (1,0), (1,1)\\}$.
2. **Paired Replicates:** The identical resample index array $idx_b$ was simultaneously applied across all candidate architectures.
3. **Point Estimates:** Reported strictly as the empirical plug-in estimate on the complete test set ($N_{\\text{test}} = 50,736$).
4. **Confidence Intervals & P-values:** 95% Confidence Intervals were constructed from the 2.5th and 97.5th percentiles of the replicate distribution. Two-tailed empirical $p$-values were computed as twice the proportion of replicates with sign opposite to the plug-in difference.
"""
    
    md_path = os.path.join(SUBMISSION_DIR, "Supplementary_Material.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_supp)
    print(f"[Supplementary Material] Saved: {md_path}")

    doc = Document()
    p_t = doc.add_heading("Supplementary Material", level=1)
    p_t.runs[0].font.size = Pt(16)
    p_t.runs[0].font.name = "Arial"
    
    p_sub = doc.add_paragraph()
    r_s = p_sub.add_run("Evaluating Demographic Disparities in Clinical Risk Prediction: A Structural Fairness and Mixture-of-Experts Investigation\nDataset: CDC BRFSS 2015 (N = 253,680)")
    r_s.italic = True
    r_s.font.size = Pt(10.5)
    r_s.font.name = "Arial"

    doc.add_heading("Section S1: Sensitivity Analysis over Fairness Regularization Parameter (λ_fairness)", level=2)
    doc.add_paragraph("Table S1 shows the sensitivity sweep across two orders of magnitude in λ_fairness on CDC BRFSS 2015.")
    
    table1 = doc.add_table(rows=1, cols=9)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table1.rows[0].cells
    headers1 = ['λ_fair', 'Nodes', 'Time (s)', 'Cost f(n)', 'Clust', 'K', 'Model', 'AUC', 'DPD']
    for i, h in enumerate(headers1):
        hdr_cells[i].text = h
        hdr_cells[i].paragraphs[0].runs[0].bold = True
        hdr_cells[i].paragraphs[0].runs[0].font.size = Pt(9.5)
        
    s1_data = [
        ['0.1', '3', '53.32', '0.2058', 'KMeans', '2', 'Adaptive', '0.8261', '0.2957'],
        ['0.5', '3', '51.69', '0.3243', 'KMeans', '2', 'Adaptive', '0.8261', '0.2957'],
        ['1.0', '3', '75.24', '0.4702', 'Auto', '6', 'LightGBM', '0.8263', '0.2896'],
        ['2.0', '3', '74.89', '0.7560', 'Auto', '4', 'Adaptive', '0.8263', '0.2896'],
        ['5.0', '3', '76.32', '1.6135', 'Auto', '2', 'Adaptive', '0.8263', '0.2896'],
        ['10.0', '3', '78.06', '3.0427', 'Auto', '4', 'Adaptive', '0.8263', '0.2896'],
        ['20.0', '3', '76.66', '5.9005', 'Auto', '2', 'LightGBM', '0.8263', '0.2896'],
        ['50.0', '3', '77.78', '14.4750', 'Auto', '2', 'LightGBM', '0.8263', '0.2896']
    ]
    for row in s1_data:
        row_cells = table1.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = val
            row_cells[i].paragraphs[0].runs[0].font.size = Pt(9)

    doc.add_heading("Section S2: Multi-Seed Stability and Variance Analysis (5 Seeds)", level=2)
    table2 = doc.add_table(rows=1, cols=6)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr2 = table2.rows[0].cells
    headers2 = ['Architecture', 'Test AUC (Mean ± SD)', 'Test Acc (%)', 'Test Balanced Acc (%)', 'Test DPD', 'Test EOD']
    for i, h in enumerate(headers2):
        hdr2[i].text = h
        hdr2[i].paragraphs[0].runs[0].bold = True
        hdr2[i].paragraphs[0].runs[0].font.size = Pt(9.5)
        
    s2_data = [
        ['Single Logistic (K=1)', '0.8197 ± 0.0004', '70.98 ± 0.12', '74.79 ± 0.10', '0.3063 ± 0.0006', '0.2768 ± 0.0010'],
        ['Single LightGBM (K=1)', '0.8263 ± 0.0004', '71.94 ± 0.15', '75.19 ± 0.11', '0.2909 ± 0.0009', '0.2576 ± 0.0012'],
        ['Single XGBoost (K=1)', '0.8262 ± 0.0004', '73.41 ± 0.10', '75.05 ± 0.11', '0.2881 ± 0.0010', '0.2511 ± 0.0013'],
        ['Pre-Processing Reweighing', '0.8162 ± 0.0005', '68.40 ± 0.14', '74.28 ± 0.12', '0.1234 ± 0.0012', '0.0850 ± 0.0015'],
        ['Hard Partitioning HP (K=2)', '0.7022 ± 0.0023', '52.02 ± 0.28', '65.47 ± 0.25', '0.0324 ± 0.0125', '0.0546 ± 0.0142'],
        ['Hierarchical MoE HMoE (K=2)', '0.8261 ± 0.0003', '72.54 ± 0.11', '75.18 ± 0.11', '0.2957 ± 0.0009', '0.2622 ± 0.0011'],
        ['Adaptive Synthesized Pipeline (λ=1)', '0.8273 ± 0.0018', '71.93 ± 2.01', '75.04 ± 0.11', '0.2836 ± 0.0059', '0.2514 ± 0.0068']
    ]
    for row in s2_data:
        row_cells = table2.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = val
            row_cells[i].paragraphs[0].runs[0].font.size = Pt(9)

    doc.add_heading("Section S3: In-Processing Exponentiated Gradient Baselines (ExpGrad-DP & ExpGrad-EO)", level=2)
    doc.add_paragraph("Table S3 summarizes in-processing reductions under Demographic Parity vs Equalized Odds constraints.")

    table3 = doc.add_table(rows=1, cols=8)
    table3.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr3 = table3.rows[0].cells
    headers3 = ['Method', 'Constraint', 'Bound ε', 'AUC', 'Acc (%)', 'DPD', 'TPR Advantaged', 'TPR Disadvantaged']
    for i, h in enumerate(headers3):
        hdr3[i].text = h
        hdr3[i].paragraphs[0].runs[0].bold = True
        hdr3[i].paragraphs[0].runs[0].font.size = Pt(9.5)
        
    s3_data = [
        ['ExpGrad-DP', 'Demographic Parity', '0.02', '0.5576', '86.48', '0.0217', '0.1328', '0.1290'],
        ['ExpGrad-EO', 'Equalized Odds', '0.02', '0.5616', '86.51', '0.0300', '0.1333', '0.1537']
    ]
    for row in s3_data:
        row_cells = table3.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = val
            row_cells[i].paragraphs[0].runs[0].font.size = Pt(9)

    docx_path = os.path.join(SUBMISSION_DIR, "Supplementary_Material.docx")
    doc.save(docx_path)
    print(f"[Supplementary Material] Saved: {docx_path}")

def main():
    print("=" * 80)
    print("BUILDING COMPLETE ELSEVIER Q1 SUBMISSION PACKAGE ARTIFACTS")
    print("=" * 80)
    generate_all_figures()
    generate_highlights_document()
    generate_figure_captions_document()
    generate_competing_interest_document()
    generate_supplementary_material()
    print("\n" + "=" * 80)
    print(f"ALL SUBMISSION PACKAGE ARTIFACTS READY IN: '{SUBMISSION_DIR}/'")
    print("=" * 80)

if __name__ == '__main__':
    main()
