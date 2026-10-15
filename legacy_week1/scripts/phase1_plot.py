"""
phase1_plot.py -- Create two separate publication-quality plots:
1. Usable cohort distribution by class (phase1_usable_cohort.png)
2. Partition size comparison (phase1_splits_distribution.png)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import CFG

def main():
    # Setup aesthetic style
    sns.set_theme(style="whitegrid", context="talk", font="DejaVu Serif")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10
    })

    # Data
    classes = ["Most Concern", "Less Concern", "No Concern"]
    counts = [199, 246, 117]
    colors_cohort = ["#d32f2f", "#f57c00", "#388e3c"] # Red, Orange, Green

    # Plot 1: Usable Dataset Cohort by DICTrank Class
    plt.figure(figsize=(6.5, 4.5), dpi=300)
    ax1 = sns.barplot(x=classes, y=counts, palette=colors_cohort, hue=classes, legend=False)
    plt.title("Usable Dataset Cohort by DICTrank Class", pad=15, fontweight="bold", fontsize=13)
    plt.ylabel("Number of Compounds", fontsize=11)
    plt.xlabel("FDA Cardiotoxicity Concern Class", fontsize=11)
    
    # Add value labels
    for i, v in enumerate(counts):
        ax1.text(i, v + 5, f"{v}\n({v/sum(counts):.1%})", ha="center", va="bottom", fontsize=9.5, fontweight="semibold")
    plt.ylim(0, 280)
    
    # Save Plot 1
    plot1_path = os.path.join(CFG.RESULTS_DIR, "phase1_usable_cohort.png")
    plt.savefig(plot1_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Plot 1 saved successfully to {plot1_path}")

    # Plot 2: Dataset Splits Target Partitions
    splits = ["Train", "Validation", "Test"]
    drug_counts = [393, 84, 85]
    scaf_counts = [393, 84, 85]

    plt.figure(figsize=(7.5, 4.5), dpi=300)
    x_indices = np.arange(len(splits))
    width = 0.35
    
    bar1 = plt.bar(x_indices - width/2, drug_counts, width, label="Drug-level Split", color="#1a1a2e")
    bar2 = plt.bar(x_indices + width/2, scaf_counts, width, label="Scaffold Split", color="#5c6bc0")
    
    plt.title("Target Partition Sizes (70 / 15 / 15)", pad=15, fontweight="bold", fontsize=13)
    plt.xticks(x_indices, splits)
    plt.ylabel("Number of Compounds", fontsize=11)
    plt.xlabel("Dataset Partitions", fontsize=11)
    plt.legend(loc="upper right", frameon=True)

    # Add value labels
    for bar in bar1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{yval}", ha="center", va="bottom", fontsize=9)
    for bar in bar2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{yval}", ha="center", va="bottom", fontsize=9)
        
    plt.ylim(0, 460)
    
    # Save Plot 2
    plot2_path = os.path.join(CFG.RESULTS_DIR, "phase1_splits_distribution.png")
    plt.savefig(plot2_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Plot 2 saved successfully to {plot2_path}")

if __name__ == "__main__":
    main()
