"""
phase1_plot.py -- Create a publication-quality visualization of the
Phase 1 dataset audit and splits to be attached to the Jupyter Notebook/report.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import CFG

def main():
    # Setup aesthetic style
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Serif")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.titlesize": 14,
        "legend.fontsize": 9.5
    })

    # Data
    classes = ["Most Concern", "Less Concern", "No Concern"]
    counts = [199, 246, 117]
    colors_cohort = ["#d32f2f", "#f57c00", "#388e3c"] # Red, Orange, Green

    splits = ["Train", "Validation", "Test"]
    drug_counts = [393, 84, 85]
    scaf_counts = [393, 84, 85]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=300)

    # Panel A: Cohort Distribution
    sns.barplot(x=classes, y=counts, palette=colors_cohort, ax=axes[0], hue=classes, legend=False)
    axes[0].set_title("A. Usable Dataset Cohort by DICTrank Class", pad=12, fontweight="bold")
    axes[0].set_ylabel("Number of Compounds")
    axes[0].set_xlabel("FDA Cardiotoxicity Concern Class")
    
    # Add value labels on top of bars
    for i, v in enumerate(counts):
        axes[0].text(i, v + 5, f"{v}\n({v/sum(counts):.1%})", ha="center", va="bottom", fontsize=9, fontweight="semibold")
    axes[0].set_ylim(0, 280)

    # Panel B: Train/Val/Test Splits
    x_indices = np.arange(len(splits))
    width = 0.35
    
    # Plot bars
    bar1 = axes[1].bar(x_indices - width/2, drug_counts, width, label="Drug-level Split", color="#1a1a2e")
    bar2 = axes[1].bar(x_indices + width/2, scaf_counts, width, label="Scaffold Split", color="#5c6bc0")
    
    axes[1].set_title("B. Target Partition Sizes (70 / 15 / 15)", pad=12, fontweight="bold")
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(splits)
    axes[1].set_ylabel("Number of Compounds")
    axes[1].set_xlabel("Dataset Partitions")
    axes[1].legend(loc="upper right", frameon=True)

    # Add value labels
    for bar in bar1:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{yval}", ha="center", va="bottom", fontsize=8.5)
    for bar in bar2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{yval}", ha="center", va="bottom", fontsize=8.5)
        
    axes[1].set_ylim(0, 460)

    plt.suptitle("Cardiotox-Fusion Phase 1 Usability Audit & Splits Summary", y=0.98, fontweight="bold")
    plt.tight_layout()
    
    # Save path
    out_path = os.path.join(CFG.RESULTS_DIR, "phase1_visualization.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Plot saved successfully to {out_path}")

if __name__ == "__main__":
    import numpy as np
    main()
