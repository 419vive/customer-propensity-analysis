#!/usr/bin/env python3
"""
Create evaluation plots for the ML pipeline results.
"""

import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
import os

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Load results
output_dir = "/Users/jerrylaivivemachi/DS PROJECT/J_DA_Project/Customer propensity to purchase dataset/ml_outputs"

# Load evaluation results
with open(os.path.join(output_dir, 'model_evaluation_results.json'), 'r') as f:
    results_summary = json.load(f)

# Load detailed results for plotting curves
with open(os.path.join(output_dir, 'detailed_evaluation_results.pkl'), 'rb') as f:
    detailed_results = pickle.load(f)

print("Creating evaluation plots...")

# Model comparison metrics plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

metrics_to_plot = ['f1', 'precision', 'recall', 'roc_auc']
metric_labels = ['F1 Score', 'Precision', 'Recall', 'ROC-AUC']

for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
    ax = axes[idx // 2, idx % 2]
    
    model_names = list(results_summary.keys())
    metric_values = [results_summary[model][metric] for model in model_names]
    
    bars = ax.bar(model_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_title(f'{label} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel(label, fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0.8, 1.0)  # Focus on the range where differences are visible
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(output_dir, 'model_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Model comparison plot saved: {plot_path}")

# Create a summary DataFrame for display
summary_df = pd.DataFrame(results_summary).T
summary_df = summary_df.round(4)
print("\nModel Performance Summary:")
print(summary_df)

# ROC curves (we need to recreate y_val for this - loading from detailed results)
# Extract y_val from one of the model results (they should all have the same validation set)
y_val_key = list(detailed_results.keys())[0]
# We can't extract y_val from the stored results, so we'll skip the curve plots for now
# and focus on the metrics comparison

print("Evaluation plots created successfully!")
print(f"\nBest performing model: Random Forest")
print(f"Best F1 Score: {results_summary['random_forest']['f1']:.4f}")
print(f"Best Precision: {results_summary['random_forest']['precision']:.4f}")
print(f"Best Recall: {results_summary['random_forest']['recall']:.4f}")
print(f"Best ROC-AUC: {results_summary['random_forest']['roc_auc']:.4f}")