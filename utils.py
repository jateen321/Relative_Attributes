"""
Utility functions and constants for Relative Attributes project.

Contains common utilities, path configurations, and helper functions
used across the entire project.
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import pickle

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Dataset paths
OSR_PATH = './data/OSR/outdoor/'
LFW_PATH = './data/LFW/lfw/'

# Results directories
RESULTS_DIR = './results/'
FEATURES_DIR = os.path.join(RESULTS_DIR, 'features/')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models/')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots/')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics/')

# Create directories if they don't exist
for directory in [FEATURES_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# CONSTANTS
# ============================================================================

# Feature parameters
GIST_PARAMS = {
    'n_scales': 4,
    'n_orients': 8,
    'gist_size': 512
}

FEATURE_PARAMS = {
    'gist_dim': 512,
    'color_dim': 45,
    'total_dim': 557
}

# Training parameters
TRAINING_PARAMS = {
    'C_values': [0.001, 0.01, 0.1],
    'C_optimal': 0.01,
    'epsilon': 0.5,
    'max_pairs': 3000,
    'similar_pairs_ratio': 0.5
}

# Attributes list
ATTRIBUTES = ['natural', 'open', 'perspective', 'large-objects', 'diagonal-plane', 'close-depth']

# OSR Scene categories
OSR_CATEGORIES = {
    'a': 'tallbuilding',
    'b': 'insidecity', 
    'c': 'street',
    'd': 'highway',
    'e': 'coast',
    'f': 'opencountry',
    'g': 'mountain',
    'h': 'forest'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_dir_exists(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def save_pickle(obj: object, filepath: str) -> None:
    """Save object to pickle file."""
    ensure_dir_exists(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✓ Saved to: {filepath}")


def load_pickle(filepath: str) -> object:
    """Load object from pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj


def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Normalize features using z-score normalization.
    
    Parameters
    ----------
    X : ndarray (n, d)
        Feature matrix
        
    Returns
    -------
    X_normalized : ndarray (n, d)
        Normalized features
    norm_params : dict
        Normalization parameters (mean, std) for later use
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    X_normalized = (X - mean) / std
    
    norm_params = {'mean': mean, 'std': std}
    return X_normalized, norm_params


def print_section(title: str, char: str = "=", width: int = 80) -> None:
    """Print formatted section header."""
    line = char * width
    print(f"\n{line}")
    print(f" {title}")
    print(f"{line}\n")


def print_progress(current: int, total: int, prefix: str = "", width: int = 50) -> None:
    """Print progress bar."""
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r{prefix} [{bar}] {percent:.1%}", end="", flush=True)
    if current == total:
        print()  # New line at end


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_features_shape(X: np.ndarray, expected_dim: int = 557) -> bool:
    """Validate that features have expected shape."""
    if X.ndim != 2:
        raise ValueError(f"Features should be 2D array, got {X.ndim}D")
    if X.shape[1] != expected_dim:
        print(f"Warning: Expected {expected_dim} features, got {X.shape[1]}")
        return False
    return True


def validate_pairs(pairs: List[Tuple[int, int]], n_samples: int) -> bool:
    """Validate that pair indices are valid."""
    pairs = np.array(pairs)
    if np.any(pairs < 0) or np.any(pairs >= n_samples):
        raise ValueError(f"Invalid pair indices for {n_samples} samples")
    return True


# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================

def print_dataset_info(X: np.ndarray, dataset_name: str = "Dataset") -> None:
    """Print dataset statistics."""
    print(f"\n{dataset_name} Statistics:")
    print(f"  Shape: {X.shape}")
    print(f"  Data type: {X.dtype}")
    print(f"  Min: {X.min():.6f}")
    print(f"  Max: {X.max():.6f}")
    print(f"  Mean: {X.mean():.6f}")
    print(f"  Std: {X.std():.6f}")
    print(f"  Memory: {X.nbytes / (1024**2):.2f} MB")


if __name__ == "__main__":
    print_section("UTILITIES MODULE TEST")
    print("✓ All utilities loaded successfully")
    print(f"✓ OSR Path: {OSR_PATH}")
    print(f"✓ Results Dir: {RESULTS_DIR}")
    print(f"✓ Total Attributes: {len(ATTRIBUTES)}")
