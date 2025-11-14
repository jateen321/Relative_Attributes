"""
Data Preparation Module

Constructs training and testing data for Ranking SVM and baseline models.
Handles pairwise data construction, balancing, and sampling.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import defaultdict
import random
from tqdm import tqdm

# ============================================================================
# PAIRWISE DATA CONSTRUCTION
# ============================================================================

def construct_ordered_pairs(category_orderings: Dict[str, List[int]], 
                           max_pairs_per_attribute: int = 3000) -> Dict[str, List[Tuple[int, int]]]:
    """
    Construct ordered pairs from category attribute rankings.
    
    For each attribute, category orderings define relative strength.
    This function creates all pairwise orderings where category i is
    ranked higher than category j.
    
    Parameters
    ----------
    category_orderings : dict
        Keys: attribute names
        Values: ordered list of category indices (higher index = stronger attribute)
        
    max_pairs_per_attribute : int
        Maximum pairs to sample per attribute (for efficiency)
        
    Returns
    -------
    pairs : dict
        Keys: attribute names
        Values: list of (i, j) tuples where category i > category j
        
    Example
    -------
    If category_orderings['natural'] = [0, 2, 1, 3]
    (Category 0 least natural, category 3 most natural)
    
    Ordered pairs generated:
    (3, 0), (3, 2), (3, 1), (2, 0), (1, 0), etc.
    """
    pairs_dict = {}
    
    for attribute, ordering in category_orderings.items():
        ordered_pairs = []
        
        # Generate all pairwise orderings
        for i in range(len(ordering)):
            for j in range(i):
                # ordering[i] is ranked higher than ordering[j]
                # So image from category ordering[i] > image from category ordering[j]
                ordered_pairs.append((ordering[i], ordering[j]))
        
        # Sample if too many pairs
        if len(ordered_pairs) > max_pairs_per_attribute:
            ordered_pairs = random.sample(ordered_pairs, max_pairs_per_attribute)
        
        pairs_dict[attribute] = ordered_pairs
    
    return pairs_dict


def construct_similarity_pairs(category_orderings: Dict[str, List[int]], 
                              epsilon: float = 0.5,
                              max_pairs_per_attribute: int = 1000) -> Dict[str, List[Tuple[int, int]]]:
    """
    Construct similarity pairs (similar attribute strength between categories).
    
    Categories at similar positions in the ranking are considered similar.
    This helps the ranking function assign similar scores to similar categories.
    
    Parameters
    ----------
    category_orderings : dict
        Category orderings for each attribute
    epsilon : float
        Threshold for considering pairs similar (position difference)
    max_pairs_per_attribute : int
        Maximum pairs to sample
        
    Returns
    -------
    pairs : dict
        Similar pairs for each attribute
        
    Rationale
    ---------
    If category A is ranked 3rd and category B is ranked 4th,
    they should receive similar (but not identical) attribute scores.
    This constraint prevents ranking function from making large jumps.
    """
    similar_pairs_dict = {}
    
    for attribute, ordering in category_orderings.items():
        similar_pairs = []
        
        # Adjacent categories in ranking are similar
        for i in range(len(ordering) - 1):
            cat_i = ordering[i]
            cat_j = ordering[i + 1]
            similar_pairs.append((cat_i, cat_j))
        
        # Sample if too many
        if len(similar_pairs) > max_pairs_per_attribute:
            similar_pairs = random.sample(similar_pairs, max_pairs_per_attribute)
        
        similar_pairs_dict[attribute] = similar_pairs
    
    return similar_pairs_dict


# ============================================================================
# IMAGE-LEVEL PAIRWISE DATA
# ============================================================================

def image_level_pairs(ordered_pairs: List[Tuple[int, int]],
                     images_per_category: int = 100) -> List[Tuple[int, int]]:
    """
    Convert category-level pairs to image-level pairs.
    
    For each (category_i, category_j) pair, randomly sample one image
    from each category.
    
    Parameters
    ----------
    ordered_pairs : list of tuples
        (category_i, category_j) tuples where category_i > category_j
    images_per_category : int
        Total images per category in dataset
        
    Returns
    -------
    image_pairs : list of tuples
        (image_i, image_j) tuples ready for training
    """
    image_pairs = []
    
    for cat_i, cat_j in ordered_pairs:
        # Sample random images from each category
        img_i = cat_i * images_per_category + np.random.randint(0, images_per_category)
        img_j = cat_j * images_per_category + np.random.randint(0, images_per_category)
        image_pairs.append((img_i, img_j))
    
    return image_pairs


# ============================================================================
# DATA SPLITTING
# ============================================================================

def split_pairs_train_test(pairs: List[Tuple[int, int]], 
                          train_ratio: float = 0.8) -> Tuple[List, List]:
    """
    Split pairs into training and testing sets.
    
    Parameters
    ----------
    pairs : list of tuples
        Ordered pairs
    train_ratio : float
        Fraction for training (default: 0.8)
        
    Returns
    -------
    train_pairs, test_pairs : lists of tuples
    """
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


# ============================================================================
# BATCH PREPARATION
# ============================================================================

def prepare_batch_data(features: np.ndarray,
                      pairs: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare batch data for pairwise ranking.
    
    Creates difference vectors (x_i - x_j) for each pair.
    These are the actual inputs to the ranking SVM.
    
    Parameters
    ----------
    features : ndarray (n_samples, d)
        Feature matrix
    pairs : list of tuples
        (i, j) pairs
        
    Returns
    -------
    diff_vectors : ndarray (n_pairs, d)
        Difference vectors x_i - x_j
    pair_indices : ndarray (n_pairs, 2)
        Original pair indices
        
    Rationale
    ---------
    Ranking SVM learns function w^T(x_i - x_j) ≥ margin
    So we work with difference vectors instead of individual feature vectors.
    """
    diff_vectors = []
    pair_indices = []
    
    for i, j in pairs:
        diff = features[i] - features[j]
        diff_vectors.append(diff)
        pair_indices.append([i, j])
    
    return np.array(diff_vectors), np.array(pair_indices)


# ============================================================================
# DATA STATISTICS
# ============================================================================

def print_data_statistics(features: np.ndarray,
                         pairs_dict: Dict[str, List],
                         dataset_name: str = "Dataset") -> None:
    """
    Print statistics about dataset and pairs.
    
    Parameters
    ----------
    features : ndarray
        Feature matrix
    pairs_dict : dict
        Dictionary of pairs for each attribute
    dataset_name : str
        Name of dataset
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} Statistics")
    print(f"{'='*60}")
    print(f"Features: {features.shape[0]} samples × {features.shape[1]} dimensions")
    print(f"Memory usage: {features.nbytes / (1024**2):.2f} MB")
    print(f"\nPairs by Attribute:")
    
    total_pairs = 0
    for attr, pairs in pairs_dict.items():
        n_pairs = len(pairs)
        total_pairs += n_pairs
        print(f"  {attr:<20} {n_pairs:>6} pairs")
    
    print(f"  {'TOTAL':<20} {total_pairs:>6} pairs")
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("Data Preparation Module")
    print("=" * 80)
    print("\nProvides:")
    print("  - Pairwise data construction")
    print("  - Similarity pair creation")
    print("  - Train-test splitting")
    print("  - Batch data preparation")
