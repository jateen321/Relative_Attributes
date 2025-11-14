"""
Visualization and Plotting Module

Creates visualizations for analysis, results, and performance comparisons.
Includes accuracy plots, confusion matrices, and qualitative examples.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# ACCURACY VISUALIZATION
# ============================================================================

def plot_accuracy_comparison(ranking_acc: Dict[str, float],
                            binary_acc: Dict[str, float],
                            title: str = "Ranking SVM vs Binary Baseline",
                            save_path: Optional[str] = None) -> None:
    """
    Plot accuracy comparison between Ranking SVM and Binary baseline.
    
    Parameters
    ----------
    ranking_acc : dict
        Attribute -> accuracy from Ranking SVM
    binary_acc : dict
        Attribute -> accuracy from Binary SVM
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    attributes = list(ranking_acc.keys())
    rank_scores = [ranking_acc[attr] for attr in attributes]
    binary_scores = [binary_acc[attr] for attr in attributes]
    
    x = np.arange(len(attributes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, rank_scores, width, label='Ranking SVM', 
                   color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, binary_scores, width, label='Binary SVM Baseline', 
                   color='#A23B72', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Pairwise Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Attribute', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attributes, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    
    plt.show()


def plot_per_attribute_performance(accuracies: Dict[str, float],
                                  save_path: Optional[str] = None) -> None:
    """
    Plot accuracy for each attribute as bar chart.
    
    Parameters
    ----------
    accuracies : dict
        Attribute -> accuracy
    save_path : str, optional
        Path to save figure
    """
    attributes = list(accuracies.keys())
    scores = [accuracies[attr] * 100 for attr in attributes]  # Convert to percentage
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(attributes)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(attributes, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Pairwise Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Visual Attributes', fontsize=12, fontweight='bold')
    ax.set_title('Ranking SVM Performance per Attribute', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 110])
    ax.grid(axis='y', alpha=0.3)
    
    # Add average line
    avg_acc = np.mean(scores)
    ax.axhline(y=avg_acc, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_acc:.2f}%')
    ax.legend(fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    
    plt.show()


# ============================================================================
# IMPROVEMENT VISUALIZATION
# ============================================================================

def plot_improvement_over_baseline(ranking_scores: np.ndarray,
                                  baseline_scores: np.ndarray,
                                  attribute_names: List[str],
                                  save_path: Optional[str] = None) -> None:
    """
    Plot relative improvement of Ranking SVM over baseline.
    
    Parameters
    ----------
    ranking_scores : ndarray
        Accuracy scores from Ranking SVM
    baseline_scores : ndarray
        Accuracy scores from baseline
    attribute_names : list of str
        Names of attributes
    save_path : str, optional
        Path to save figure
    """
    improvements = ((ranking_scores - baseline_scores) / baseline_scores * 100)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax.barh(attribute_names, improvements, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        ax.text(imp + 1, i, f'{imp:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Relative Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ranking SVM Improvement over Binary Baseline', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    
    plt.show()


# ============================================================================
# SCORE DISTRIBUTION
# ============================================================================

def plot_score_distributions(scores_positive: np.ndarray,
                            scores_negative: np.ndarray,
                            attribute_name: str = "Attribute",
                            save_path: Optional[str] = None) -> None:
    """
    Plot distribution of margin scores for positive and negative pairs.
    
    Parameters
    ----------
    scores_positive : ndarray
        Scores for correctly ordered pairs
    scores_negative : ndarray
        Scores for incorrectly ordered pairs
    attribute_name : str
        Name of attribute
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(scores_positive, bins=30, alpha=0.6, label='Correctly Ordered', 
            color='green', edgecolor='black')
    ax.hist(scores_negative, bins=30, alpha=0.6, label='Incorrectly Ordered', 
            color='red', edgecolor='black')
    
    ax.set_xlabel('Margin Score: w^T(x_i - x_j)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'Score Distribution - {attribute_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    
    plt.show()


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance(weights: Dict[str, np.ndarray],
                           top_k: int = 10,
                           save_path: Optional[str] = None) -> None:
    """
    Plot top-k most important features for each attribute.
    
    Parameters
    ----------
    weights : dict
        Attribute -> weight vector
    top_k : int
        Number of top features to show
    save_path : str, optional
        Path to save figure
    """
    n_attributes = len(weights)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (attr, w) in enumerate(weights.items()):
        # Get top-k features by absolute value
        top_indices = np.argsort(np.abs(w))[-top_k:]
        top_indices = top_indices[::-1]  # Descending order
        top_values = w[top_indices]
        
        ax = axes[idx]
        colors = ['green' if v > 0 else 'red' for v in top_values]
        
        ax.barh(range(top_k), top_values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f'Feature {i}' for i in top_indices], fontsize=9)
        ax.set_xlabel('Weight', fontsize=10)
        ax.set_title(f'{attr}', fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_attributes, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Top-10 Feature Importance per Attribute', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    
    plt.show()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_comparison_table(results: Dict) -> str:
    """
    Create formatted comparison table.
    
    Parameters
    ----------
    results : dict
        Results dictionary with metrics
        
    Returns
    -------
    table_str : str
        Formatted table string
    """
    table = "| Metric | Ranking SVM | Binary SVM | Improvement |\n"
    table += "|--------|-------------|-----------|-------------|\n"
    
    for key, value in results.items():
        if isinstance(value, dict):
            continue
        if key.endswith('_accuracy'):
            attr = key.replace('_accuracy', '').replace('_', ' ').title()
            table += f"| {attr} | {value:.4f} |\n"
    
    return table


if __name__ == "__main__":
    print("=" * 80)
    print("Visualization Module")
    print("=" * 80)
    print("\nProvides:")
    print("  - Accuracy comparison plots")
    print("  - Per-attribute performance visualization")
    print("  - Improvement over baseline charts")
    print("  - Score distribution histograms")
    print("  - Feature importance plots")
