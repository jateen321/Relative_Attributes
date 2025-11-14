"""
Ranking SVM Implementation

Implements the Ranking SVM formulation using quadratic programming.
Learns a weight vector w that maximizes margin for correct pairwise orderings.

Mathematical Formulation
------------------------
minimize: (1/2) ||w||^2 + C * (sum(xi_ij) + sum(zeta_ij))

subject to:
  w^T(x_i - x_j) >= 1 - xi_ij         (ordering constraints)
  |w^T(x_i - x_j)| <= epsilon + zeta_ij  (similarity constraints)
  xi_ij, zeta_ij >= 0                 (slack variables)

where:
  w: learned weight vector (d-dimensional)
  C: regularization parameter (controls margin-violation tradeoff)
  xi_ij: slack for violated orderings
  zeta_ij: slack for violated similarity
  epsilon: allowed similarity tolerance
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.optimize import minimize
import cvxopt
from cvxopt import matrix, solvers
from tqdm import tqdm
import time

# Disable verbose CVXOPT output
solvers.options['show_progress'] = False

# ============================================================================
# QP FORMULATION AND SOLVING
# ============================================================================

def formulate_ranking_qp(X_diff: np.ndarray,
                        C: float = 0.01,
                        epsilon: float = 0.5) -> Tuple[np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray]:
    """
    Formulate Ranking SVM as quadratic program (QP).
    
    Converts the ranking SVM problem into standard QP form:
    minimize: (1/2) w^T P w + q^T w
    subject to: G w <= h
               A w = b
    
    Parameters
    ----------
    X_diff : ndarray (n_pairs, d)
        Difference vectors (x_i - x_j) for ordered pairs
    C : float
        Regularization parameter
    epsilon : float
        Similarity tolerance
        
    Returns
    -------
    P, q, G, h : QP matrices
        Ready for CVXOPT solver
        
    Derivation
    ----------
    Original problem has slack variables for both ordering and similarity.
    We reformulate into standard QP by stacking all variables:
    
    u = [w; xi; zeta]
    
    where xi are slack variables for ordered pairs
          zeta are slack variables for similarity pairs
    
    The QP formulation then enforces both ordering and similarity constraints.
    """
    n_pairs = X_diff.shape[0]
    d = X_diff.shape[1]
    
    # Total variables: w (d) + xi (n_pairs) + zeta (n_pairs)
    n_vars = d + 2 * n_pairs
    
    # P matrix: (1/2) w^T P w where P = [I, 0; 0, 0] in full space
    # Only the w part contributes to regularization
    P = np.eye(n_vars)
    P[d:, d:] = 0  # No regularization on slack variables
    
    # q vector: coefficient of linear term
    # For slack variables: coefficient is C
    q = np.zeros(n_vars)
    q[d:] = C
    
    # Inequality constraints: G w <= h
    # We have 2 * n_pairs constraints (ordering and similarity)
    n_constraints = 2 * n_pairs
    
    G = np.zeros((n_constraints, n_vars))
    h = np.zeros(n_constraints)
    
    # Ordering constraints: w^T(x_i - x_j) >= 1 - xi_ij
    # Reformulated as: -w^T(x_i - x_j) + xi_ij <= -1
    for i in range(n_pairs):
        G[i, :d] = -X_diff[i, :]  # Coefficient of w
        G[i, d + i] = 1            # Coefficient of xi_i
        h[i] = -1
    
    # Similarity constraints: |w^T(x_i - x_j)| <= epsilon + zeta_ij
    # Lower bound: w^T(x_i - x_j) >= -(epsilon + zeta_ij)
    # Reformulated as: -w^T(x_i - x_j) - zeta_ij <= -epsilon
    for i in range(n_pairs):
        G[n_pairs + i, :d] = -X_diff[i, :]
        G[n_pairs + i, d + n_pairs + i] = -1
        h[n_pairs + i] = -epsilon
    
    return P, q, G, h


def solve_ranking_qp(X_diff: np.ndarray,
                    C: float = 0.01,
                    epsilon: float = 0.5) -> np.ndarray:
    """
    Solve Ranking SVM using quadratic programming.
    
    Parameters
    ----------
    X_diff : ndarray (n_pairs, d)
        Difference vectors for ordered pairs
    C : float
        Regularization parameter
    epsilon : float
        Similarity tolerance
        
    Returns
    -------
    w : ndarray (d,)
        Learned weight vector
    """
    n_pairs = X_diff.shape[0]
    d = X_diff.shape[1]
    
    # Formulate QP
    P, q, G, h = formulate_ranking_qp(X_diff, C, epsilon)
    
    # Convert to CVXOPT format
    P_cvx = matrix(P, tc='d')
    q_cvx = matrix(q, tc='d')
    G_cvx = matrix(G, tc='d')
    h_cvx = matrix(h, tc='d')
    
    # Solve QP
    try:
        solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
        
        if solution['status'] == 'optimal':
            w = np.array(solution['x']).flatten()[:d]
            return w
        else:
            print(f"Warning: QP solver status: {solution['status']}")
            return np.zeros(d)
            
    except Exception as e:
        print(f"Error in QP solver: {e}")
        return np.zeros(d)


# ============================================================================
# RANKING FUNCTION TRAINING
# ============================================================================

class RankingSVM:
    """
    Ranking SVM model for learning relative attributes.
    
    Learns a linear ranking function w that assigns higher scores to
    categories with stronger attributes.
    """
    
    def __init__(self, C: float = 0.01, epsilon: float = 0.5):
        """
        Initialize Ranking SVM.
        
        Parameters
        ----------
        C : float
            Regularization parameter (default: 0.01)
            - Higher C: less regularization, fits training data better
            - Lower C: more regularization, prevents overfitting
        epsilon : float
            Similarity tolerance (default: 0.5)
        """
        self.C = C
        self.epsilon = epsilon
        self.w = None
        self.fitted = False
        self.training_history = {}
    
    def fit(self, X_diff: np.ndarray) -> None:
        """
        Train ranking SVM.
        
        Parameters
        ----------
        X_diff : ndarray (n_pairs, d)
            Difference vectors x_i - x_j for ordered pairs
        """
        print(f"Training Ranking SVM (C={self.C}, epsilon={self.epsilon})...")
        print(f"  Input: {X_diff.shape[0]} pairs × {X_diff.shape[1]} dimensions")
        
        start_time = time.time()
        
        # Solve QP
        self.w = solve_ranking_qp(X_diff, self.C, self.epsilon)
        
        elapsed = time.time() - start_time
        print(f"✓ Training completed in {elapsed:.2f}s")
        
        self.fitted = True
        
        # Store statistics
        self.training_history = {
            'n_pairs': X_diff.shape[0],
            'n_features': X_diff.shape[1],
            'training_time': elapsed,
            'C': self.C,
            'epsilon': self.epsilon
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict attribute scores for samples.
        
        Parameters
        ----------
        X : ndarray (n_samples, d)
            Feature matrix
            
        Returns
        -------
        scores : ndarray (n_samples,)
            Predicted scores (higher = stronger attribute)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        scores = X @ self.w
        return scores
    
    def predict_pairwise(self, X: np.ndarray, 
                        pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Predict pairwise rankings for given pairs.
        
        Parameters
        ----------
        X : ndarray (n_samples, d)
            Feature matrix
        pairs : list of tuples
            (i, j) pairs to predict
            
        Returns
        -------
        predictions : ndarray (n_pairs,)
            1 if w^T(x_i - x_j) > 0 (i ranked higher)
               0 otherwise
        """
        predictions = []
        for i, j in pairs:
            diff_score = (X[i] - X[j]) @ self.w
            pred = 1 if diff_score > 0 else 0
            predictions.append(pred)
        
        return np.array(predictions)


# ============================================================================
# BASELINE: BINARY SVM
# ============================================================================

class BinarySVM:
    """
    Binary SVM baseline for comparison.
    
    Converts ranking problem to binary classification:
    - Positive class: attribute strong (top half of ranking)
    - Negative class: attribute weak (bottom half of ranking)
    
    This illustrates why ranking formulation is better than
    naive binary classification approach.
    """
    
    def __init__(self, C: float = 1.0):
        self.C = C
        self.w = None
        self.b = None
        self.fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train binary SVM.
        
        Parameters
        ----------
        X : ndarray (n_samples, d)
            Feature matrix
        y : ndarray (n_samples,)
            Binary labels (0 or 1)
        """
        from sklearn.svm import SVC
        
        print(f"Training Binary SVM (C={self.C})...")
        
        svm = SVC(kernel='linear', C=self.C, random_state=42)
        svm.fit(X, y)
        
        self.w = svm.coef_[0]
        self.b = svm.intercept_[0]
        self.fitted = True
        
        print(f"✓ Binary SVM trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class."""
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        
        scores = X @ self.w + self.b
        return (scores > 0).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability scores."""
        scores = X @ self.w + self.b
        return 1 / (1 + np.exp(-scores))


if __name__ == "__main__":
    print("=" * 80)
    print("Ranking SVM Module")
    print("=" * 80)
    print("\nProvides:")
    print("  - Ranking SVM formulation and solving")
    print("  - Binary SVM baseline")
    print("  - Pairwise ranking prediction")
