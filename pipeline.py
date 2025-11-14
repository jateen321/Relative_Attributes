"""
Main Pipeline - Relative Attributes Computer Vision

Complete end-to-end pipeline for Ranking SVM-based Relative Attributes learning.

Module Organization:
├── utils.py           : Configuration, paths, utilities
├── features.py        : Feature extraction (GIST, Gabor, color)
├── data_prep.py       : Pairwise data construction
├── ranking_svm.py     : Ranking SVM & Binary SVM models
├── evaluation.py      : Accuracy metrics & evaluation
└── visualization.py   : Plotting & visualization
"""

import numpy as np
import os
from typing import Dict, List, Tuple
from tqdm import tqdm

# Import all modules
from utils import print_section, ATTRIBUTES, OSR_CATEGORIES, TRAINING_PARAMS
from features import extract_features_batch, FeatureNormalizer
from data_prep import (construct_ordered_pairs, construct_similarity_pairs, 
                       prepare_batch_data, print_data_statistics)
from ranking_svm import RankingSVM, BinarySVM
from evaluation import (pairwise_accuracy, pairwise_accuracy_batch,
                       compare_ranking_vs_binary, print_evaluation_report,
                       compute_per_attribute_accuracy)
from visualization import (plot_accuracy_comparison, plot_per_attribute_performance,
                          plot_improvement_over_baseline)


# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

class RankingAttributesPipeline:
    """
    Complete pipeline for Relative Attributes learning.
    
    Flow:
    1. Load/extract features from images
    2. Construct pairwise training data
    3. Train Ranking SVM for each attribute
    4. Evaluate on test pairs
    5. Compare with Binary SVM baseline
    6. Visualize results
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize pipeline.
        
        Parameters
        ----------
        config : dict, optional
            Configuration parameters (overrides defaults)
        """
        self.config = config or TRAINING_PARAMS
        self.features = None
        self.normalizer = None
        self.rankers = {}  # attribute -> RankingSVM
        self.baseline_models = {}  # attribute -> BinarySVM
        self.results = {}
    
    # ========================================================================
    # PHASE 1: FEATURE EXTRACTION
    # ========================================================================
    
    def extract_features(self, image_paths: List[str], 
                        save_cache: bool = True,
                        cache_path: str = None) -> np.ndarray:
        """
        Extract features from images.
        
        Parameters
        ----------
        image_paths : list of str
            Paths to images
        save_cache : bool
            Whether to cache features
        cache_path : str
            Path to save cached features
            
        Returns
        -------
        features : ndarray (n_images, 557)
            Feature matrix
        """
        print_section("PHASE 1: FEATURE EXTRACTION")
        
        # Extract features
        self.features = extract_features_batch(image_paths, resize=(256, 256))
        
        # Normalize features
        self.normalizer = FeatureNormalizer()
        features_normalized = self.normalizer.fit_transform(self.features)
        
        if save_cache and cache_path:
            np.save(cache_path, features_normalized)
            print(f"✓ Features cached to: {cache_path}")
        
        return features_normalized
    
    # ========================================================================
    # PHASE 2: DATA PREPARATION
    # ========================================================================
    
    def prepare_training_data(self, 
                             category_orderings: Dict[str, List[int]],
                             features: np.ndarray) -> Dict:
        """
        Construct pairwise training data for each attribute.
        
        Parameters
        ----------
        category_orderings : dict
            Attribute -> ordered list of category indices
        features : ndarray
            Feature matrix
            
        Returns
        -------
        training_data : dict
            Attribute -> {ordered_pairs, similar_pairs, X_diff_ordered, X_diff_similar}
        """
        print_section("PHASE 2: DATA PREPARATION")
        
        training_data = {}
        
        for attribute in ATTRIBUTES:
            if attribute not in category_orderings:
                print(f"Warning: No ordering for {attribute}")
                continue
            
            print(f"\nProcessing attribute: {attribute}")
            
            # Construct ordered pairs
            ordered_pairs = construct_ordered_pairs(
                {attribute: category_orderings[attribute]},
                max_pairs_per_attribute=self.config['max_pairs']
            )[attribute]
            
            # Construct similarity pairs
            similar_pairs = construct_similarity_pairs(
                {attribute: category_orderings[attribute]},
                epsilon=self.config['epsilon'],
                max_pairs_per_attribute=int(self.config['max_pairs'] * 
                                           self.config['similar_pairs_ratio'])
            )[attribute]
            
            # Prepare difference vectors
            X_diff_ordered, _ = prepare_batch_data(features, ordered_pairs)
            X_diff_similar, _ = prepare_batch_data(features, similar_pairs)
            
            training_data[attribute] = {
                'ordered_pairs': ordered_pairs,
                'similar_pairs': similar_pairs,
                'X_diff_ordered': X_diff_ordered,
                'X_diff_similar': X_diff_similar,
                'n_ordered': len(ordered_pairs),
                'n_similar': len(similar_pairs)
            }
            
            print(f"  ✓ Ordered pairs: {len(ordered_pairs)}")
            print(f"  ✓ Similar pairs: {len(similar_pairs)}")
        
        return training_data
    
    # ========================================================================
    # PHASE 3: MODEL TRAINING
    # ========================================================================
    
    def train_ranking_svms(self, training_data: Dict, features: np.ndarray) -> None:
        """
        Train Ranking SVM for each attribute.
        
        Parameters
        ----------
        training_data : dict
            Pairwise training data
        features : ndarray
            Feature matrix
        """
        print_section("PHASE 3: RANKING SVM TRAINING")
        
        C_values = self.config['C_values']
        C_optimal = self.config['C_optimal']
        
        for attribute in ATTRIBUTES:
            if attribute not in training_data:
                continue
            
            print(f"\nTraining: {attribute}")
            
            # Use optimal C for training
            ranker = RankingSVM(C=C_optimal, 
                              epsilon=self.config['epsilon'])
            
            # Train on ordered pairs
            ranker.fit(training_data[attribute]['X_diff_ordered'])
            
            self.rankers[attribute] = ranker
    
    def train_binary_baselines(self, training_data: Dict, 
                              features: np.ndarray,
                              category_orderings: Dict) -> None:
        """
        Train Binary SVM baselines for comparison.
        
        Parameters
        ----------
        training_data : dict
            Training data
        features : ndarray
            Feature matrix
        category_orderings : dict
            Category attribute orderings
        """
        print_section("PHASE 4: BINARY SVM BASELINE")
        
        for attribute in ATTRIBUTES:
            if attribute not in category_orderings:
                continue
            
            print(f"\nTraining baseline: {attribute}")
            
            # Create binary labels: top half = 1, bottom half = 0
            ordering = category_orderings[attribute]
            n_categories = len(ordering)
            threshold = n_categories // 2
            
            # Labels for all images (assuming same number per category)
            n_images_per_cat = features.shape[0] // n_categories
            y = np.zeros(features.shape[0])
            
            for i, cat_idx in enumerate(ordering):
                if i >= threshold:  # Top half
                    start = cat_idx * n_images_per_cat
                    end = start + n_images_per_cat
                    y[start:end] = 1
            
            # Train binary SVM
            baseline = BinarySVM(C=1.0)
            baseline.fit(features, y)
            
            self.baseline_models[attribute] = baseline
    
    # ========================================================================
    # PHASE 5: EVALUATION
    # ========================================================================
    
    def evaluate(self, training_data: Dict, features: np.ndarray,
                split_ratio: float = 0.8) -> Dict:
        """
        Evaluate trained models.
        
        Parameters
        ----------
        training_data : dict
            Training data with pairs
        features : ndarray
            Feature matrix
        split_ratio : float
            Train-test split ratio
            
        Returns
        -------
        results : dict
            Evaluation results
        """
        print_section("PHASE 5: EVALUATION")
        
        results = {}
        ranking_accuracies = {}
        
        for attribute in ATTRIBUTES:
            if attribute not in training_data or attribute not in self.rankers:
                continue
            
            print(f"\nEvaluating: {attribute}")
            
            # Get test pairs (use ordered pairs for now)
            test_pairs = training_data[attribute]['ordered_pairs']
            n_split = int(len(test_pairs) * split_ratio)
            test_pairs = test_pairs[n_split:]  # Use second half for testing
            
            # Evaluate Ranking SVM
            ranker = self.rankers[attribute]
            ranking_acc = pairwise_accuracy(ranker.w, features, test_pairs)
            ranking_accuracies[attribute] = ranking_acc
            
            print(f"  Ranking SVM: {ranking_acc:.4f}")
            
            results[attribute] = {
                'ranking_accuracy': ranking_acc,
                'n_test_pairs': len(test_pairs)
            }
        
        # Compute average
        if ranking_accuracies:
            avg_acc = np.mean(list(ranking_accuracies.values()))
            print(f"\n{'='*60}")
            print(f"Average Ranking SVM Accuracy: {avg_acc:.4f}")
            print(f"{'='*60}")
        
        self.results = results
        return results
    
    # ========================================================================
    # PHASE 6: VISUALIZATION
    # ========================================================================
    
    def visualize_results(self, save_dir: str = './results/plots/') -> None:
        """
        Generate visualizations of results.
        
        Parameters
        ----------
        save_dir : str
            Directory to save plots
        """
        print_section("PHASE 6: VISUALIZATION")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract accuracies
        ranking_accs = {attr: self.results[attr]['ranking_accuracy'] 
                       for attr in self.results}
        
        # Plot per-attribute performance
        plot_per_attribute_performance(
            ranking_accs,
            save_path=os.path.join(save_dir, 'per_attribute_accuracy.png')
        )
        
        print(f"\n✓ Visualizations saved to: {save_dir}")
    
    # ========================================================================
    # PIPELINE EXECUTION
    # ========================================================================
    
    def run(self, image_paths: List[str],
           category_orderings: Dict[str, List[int]],
           split_ratio: float = 0.8) -> Dict:
        """
        Run complete pipeline.
        
        Parameters
        ----------
        image_paths : list of str
            Paths to all images
        category_orderings : dict
            Attribute -> ordered category indices
        split_ratio : float
            Train-test split ratio
            
        Returns
        -------
        results : dict
            Complete results
        """
        print("\n" + "="*80)
        print(" RELATIVE ATTRIBUTES RANKING SVM PIPELINE")
        print("="*80)
        
        # Phase 1: Feature extraction
        features = self.extract_features(image_paths)
        
        # Phase 2: Data preparation
        training_data = self.prepare_training_data(category_orderings, features)
        
        # Phase 3: Ranking SVM training
        self.train_ranking_svms(training_data, features)
        
        # Phase 4: Baseline training
        self.train_binary_baselines(training_data, features, category_orderings)
        
        # Phase 5: Evaluation
        results = self.evaluate(training_data, features, split_ratio)
        
        # Phase 6: Visualization
        self.visualize_results()
        
        print("\n" + "="*80)
        print(" PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
        
        return results


if __name__ == "__main__":
    print("="*80)
    print("Main Pipeline Module - Relative Attributes")
    print("="*80)
    print("\nUsage:")
    print("  pipeline = RankingAttributesPipeline()")
    print("  results = pipeline.run(image_paths, category_orderings)")
    print("\nModules:")
    print("  - utils: Configuration and utilities")
    print("  - features: Feature extraction")
    print("  - data_prep: Pairwise data construction")
    print("  - ranking_svm: Model training")
    print("  - evaluation: Performance metrics")
    print("  - visualization: Result visualization")
