"""
Evaluation and Metrics Module

Computes accuracy metrics for ranking models including pairwise accuracy,
zero-shot learning evaluation, and comparative analysis with baselines.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm

# ============================================================================
# PAIRWISE ACCURACY
# ============================================================================

def pairwise_accuracy(w: np.ndarray,
                     X: np.ndarray,
                     pairs: List[Tuple[int, int]]) -> float:
    """
    Compute pairwise accuracy for ranking function.
    
    For each pair (i, j), check if w^T(x_i - x_j) > 0
    (i.e., model predicts x_i ranked higher than x_j)
    
    Parameters
    ----------
    w : ndarray (d,)
        Weight vector of ranking function
    X : ndarray (n_samples, d)
        Feature matrix
    pairs : list of tuples
        Ground truth ordered pairs (i, j) where i > j
        
    Returns
    -------
    accuracy : float
        Fraction of correctly ordered pairs (0 to 1)
        
    Formula
    -------
    accuracy = (# correctly ordered pairs) / (total # pairs)
    """
    if len(pairs) == 0:
        return 0.0
    
    correct = 0
    for i, j in pairs:
        # Compute difference score
        diff_score = (X[i] - X[j]) @ w
        
        # Check if ranking is correct (should be > 0)
        if diff_score > 0:
            correct += 1
    
    accuracy = correct / len(pairs)
    return accuracy


def pairwise_accuracy_batch(w: np.ndarray,
                            X_diff: np.ndarray) -> float:
    """
    Compute pairwise accuracy using pre-computed difference vectors.
    
    More efficient when difference vectors are already computed.
    
    Parameters
    ----------
    w : ndarray (d,)
        Weight vector
    X_diff : ndarray (n_pairs, d)
        Pre-computed difference vectors
        
    Returns
    -------
    accuracy : float
    """
    if X_diff.shape[0] == 0:
        return 0.0
    
    # Compute scores for all pairs at once
    scores = X_diff @ w
    
    # Count how many are positive (correctly ordered)
    correct = np.sum(scores > 0)
    accuracy = correct / len(scores)
    
    return accuracy


# ============================================================================
# MARGIN ANALYSIS
# ============================================================================

def compute_margins(w: np.ndarray,
                   X_diff: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute margin statistics for ranking function.
    
    Margin = w^T(x_i - x_j) indicates confidence in ordering.
    Larger margin = more confident prediction.
    
    Parameters
    ----------
    w : ndarray (d,)
        Weight vector
    X_diff : ndarray (n_pairs, d)
        Difference vectors
        
    Returns
    -------
    min_margin : float
        Minimum margin (worst-case)
    mean_margin : float
        Mean margin (average confidence)
    std_margin : float
        Margin standard deviation
    """
    margins = X_diff @ w
    
    return np.min(margins), np.mean(margins), np.std(margins)


# ============================================================================
# COMPARATIVE EVALUATION
# ============================================================================

def compare_ranking_vs_binary(ranker: object,
                              binary_model: object,
                              X: np.ndarray,
                              test_pairs: List[Tuple[int, int]],
                              category_ordering: List[int]) -> Dict[str, float]:
    """
    Compare Ranking SVM vs Binary SVM baseline.
    
    Parameters
    ----------
    ranker : RankingSVM object
        Trained ranking SVM model
    binary_model : BinarySVM object
        Trained binary SVM model
    X : ndarray (n_samples, d)
        Feature matrix
    test_pairs : list of tuples
        Test pairs for evaluation
    category_ordering : list
        Ground truth category ordering
        
    Returns
    -------
    results : dict
        Accuracy metrics for both methods
    """
    # Ranking SVM accuracy
    ranking_acc = pairwise_accuracy(ranker.w, X, test_pairs)
    
    # Binary SVM accuracy
    binary_correct = 0
    for i, j in test_pairs:
        pred_i = binary_model.predict(X[i:i+1])[0]
        pred_j = binary_model.predict(X[j:j+1])[0]
        
        # Correct if higher-ranked category predicted as 1
        if category_ordering.index(i) > category_ordering.index(j):
            if pred_i > pred_j:
                binary_correct += 1
        else:
            if pred_i <= pred_j:
                binary_correct += 1
    
    binary_acc = binary_correct / len(test_pairs) if len(test_pairs) > 0 else 0
    
    results = {
        'ranking_svm_accuracy': ranking_acc,
        'binary_svm_accuracy': binary_acc,
        'improvement': ranking_acc - binary_acc,
        'relative_improvement': (ranking_acc - binary_acc) / binary_acc * 100 if binary_acc > 0 else 0
    }
    
    return results


# ============================================================================
# ZERO-SHOT LEARNING EVALUATION
# ============================================================================

def zero_shot_accuracy(learned_attributes: Dict[str, np.ndarray],
                       test_categories: List[int],
                       true_category_descriptions: Dict[int, Dict[str, float]],
                       k: int = 1) -> float:
    """
    Evaluate zero-shot learning accuracy.
    
    Uses learned attribute rankers to predict category labels for unseen
    categories based on semantic descriptions.
    
    Parameters
    ----------
    learned_attributes : dict
        Keys: attribute names
        Values: weight vectors from training
    test_categories : list
        Unseen category indices
    true_category_descriptions : dict
        Keys: category indices
        Values: dict of {attribute: strength}
    k : int
        Top-k accuracy (default: 1)
        
    Returns
    -------
    accuracy : float
        Fraction of correctly identified categories
    """
    correct = 0
    
    for category in test_categories:
        # Get true description
        true_desc = true_category_descriptions[category]
        
        # Rank all training categories based on description similarity
        similarities = {}
        for train_cat in range(len(true_category_descriptions)):
            if train_cat == category:
                continue
            
            train_desc = true_category_descriptions[train_cat]
            
            # Compute similarity based on attribute agreement
            similarity = 0
            for attr in learned_attributes:
                if attr in true_desc and attr in train_desc:
                    similarity += abs(true_desc[attr] - train_desc[attr])
            
            similarities[train_cat] = -similarity  # Negative for sorting
        
        # Get top-k most similar categories
        top_k_cats = sorted(similarities.keys(), 
                           key=lambda x: similarities[x])[:k]
        
        if category in top_k_cats:
            correct += 1
    
    accuracy = correct / len(test_categories) if test_categories else 0
    return accuracy


# ============================================================================
# CONFUSION MATRIX AND DETAILED ANALYSIS
# ============================================================================

def compute_per_attribute_accuracy(ranker_w: np.ndarray,
                                   X: np.ndarray,
                                   attribute_pairs: Dict[str, List[Tuple]]) -> Dict[str, float]:
    """
    Compute accuracy separately for each attribute.
    
    Parameters
    ----------
    ranker_w : ndarray (d,)
        Learned weight vector
    X : ndarray (n_samples, d)
        Features
    attribute_pairs : dict
        Keys: attribute names
        Values: list of test pairs
        
    Returns
    -------
    accuracies : dict
        Per-attribute accuracy scores
    """
    accuracies = {}
    
    for attribute, pairs in attribute_pairs.items():
        acc = pairwise_accuracy(ranker_w, X, pairs)
        accuracies[attribute] = acc
    
    return accuracies


def print_evaluation_report(results: Dict, attribute_name: str = "Attribute") -> None:
    """
    Print formatted evaluation report.
    
    Parameters
    ----------
    results : dict
        Evaluation results
    attribute_name : str
        Name of attribute being evaluated
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Report: {attribute_name}")
    print(f"{'='*60}")
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:<30} {value:>10.4f}")
        else:
            print(f"{key:<30} {value}")


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_confidence_intervals(accuracies: np.ndarray,
                                confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for accuracy.
    
    Uses normal approximation (binomial distribution approximation).
    
    Parameters
    ----------
    accuracies : ndarray
        Accuracy scores from multiple runs
    confidence : float
        Confidence level (default: 0.95 = 95%)
        
    Returns
    -------
    lower, upper : floats
        Confidence interval bounds
    """
    from scipy import stats
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    n = len(accuracies)
    
    # Standard error
    se = std_acc / np.sqrt(n)
    
    # Critical value
    z = stats.norm.ppf((1 + confidence) / 2)
    
    margin = z * se
    return mean_acc - margin, mean_acc + margin


def print_summary_statistics(accuracies_dict: Dict[str, List[float]]) -> None:
    """
    Print summary statistics for multiple runs.
    
    Parameters
    ----------
    accuracies_dict : dict
        Keys: attribute names
        Values: list of accuracies from multiple runs
    """
    print(f"\n{'='*70}")
    print(f"{'Attribute':<25} {'Mean':<12} {'Std':<12} {'Min-Max':<20}")
    print(f"{'='*70}")
    
    for attr, accs in accuracies_dict.items():
        mean = np.mean(accs)
        std = np.std(accs)
        min_val = np.min(accs)
        max_val = np.max(accs)
        
        print(f"{attr:<25} {mean:>10.4f}  {std:>10.4f}  [{min_val:.4f}, {max_val:.4f}]")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    print("=" * 80)
    print("Evaluation Module")
    print("=" * 80)
    print("\nProvides:")
    print("  - Pairwise accuracy computation")
    print("  - Ranking vs baseline comparison")
    print("  - Zero-shot learning evaluation")
    print("  - Per-attribute analysis")
    print("  - Statistical summaries")
