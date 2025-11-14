# Modularized Python Modules - Relative Attributes CV

This directory contains the modularized Python implementation extracted from the Jupyter notebook, organized into focused, reusable modules.

## 📁 Module Structure

```
├── utils.py              Configuration, paths, constants, utilities
├── features.py           Feature extraction (GIST, Gabor, Color)
├── data_prep.py          Pairwise data construction
├── ranking_svm.py        Ranking SVM & Binary SVM implementations
├── evaluation.py         Metrics and evaluation functions
├── visualization.py      Plotting and visualization utilities
└── pipeline.py           Main end-to-end pipeline
```

## 🔧 Module Descriptions

### 1. `utils.py` - Utilities & Configuration
**Purpose**: Centralized configuration, paths, and helper functions

**Key Classes/Functions**:
- Path configuration (OSR_PATH, LFW_PATH, RESULTS_DIR)
- Constants (ATTRIBUTES, OSR_CATEGORIES, TRAINING_PARAMS)
- Helper functions (save_pickle, load_pickle, normalize_features)
- Data validation and statistics printing

**Usage**:
```python
from utils import ATTRIBUTES, OSR_CATEGORIES, TRAINING_PARAMS
from utils import save_pickle, load_pickle, normalize_features

# Access configuration
print(ATTRIBUTES)  # ['natural', 'open', 'perspective', ...]
print(TRAINING_PARAMS['C_optimal'])  # 0.01
```

### 2. `features.py` - Feature Extraction
**Purpose**: Extract visual features from images (GIST + Color)

**Key Functions**:
- `build_gabor_filters()`: Create Gabor filter bank (8 orientations × 4 scales)
- `extract_gist_features()`: Extract 512-dim GIST descriptors
- `extract_color_histogram()`: Extract 45-dim Lab color histogram
- `extract_features_batch()`: Batch processing with progress tracking

**Classes**:
- `FeatureNormalizer`: StandardScaler wrapper for consistent normalization

**Total Features**: 557-dimensional (512 GIST + 45 Color)

**Usage**:
```python
from features import extract_features_batch, FeatureNormalizer

# Extract features from images
image_paths = ['img1.jpg', 'img2.jpg', ...]
features = extract_features_batch(image_paths, resize=(256, 256))
# Returns: (n_images, 557) array

# Normalize features
normalizer = FeatureNormalizer()
X_normalized = normalizer.fit_transform(features)
```

### 3. `data_prep.py` - Data Preparation
**Purpose**: Construct pairwise training data for Ranking SVM

**Key Functions**:
- `construct_ordered_pairs()`: Create ordered pairs from category rankings
- `construct_similarity_pairs()`: Create adjacent/similar pairs
- `prepare_batch_data()`: Convert to difference vectors for SVM input
- `split_pairs_train_test()`: Train-test split for pairs

**Usage**:
```python
from data_prep import construct_ordered_pairs, prepare_batch_data

# Define attribute orderings (0 = weakest, 7 = strongest)
orderings = {
    'natural': [5, 3, 1, 0, 7, 2, 4, 6],  # Category ordering
    'open': [2, 1, 5, 6, 0, 3, 4, 7]
}

# Generate pairs
pairs = construct_ordered_pairs(orderings, max_pairs_per_attribute=3000)

# Prepare for SVM (creates difference vectors)
X_diff, pair_indices = prepare_batch_data(features, pairs['natural'])
# X_diff: (n_pairs, 557) - difference vectors
```

### 4. `ranking_svm.py` - Ranking SVM Models
**Purpose**: Implement Ranking SVM and Binary SVM baseline

**Key Classes**:
- `RankingSVM`: Main ranking SVM model
  - Methods: `fit()`, `predict()`, `predict_pairwise()`
  - Uses CVXOPT for QP solving
  
- `BinarySVM`: Binary classification baseline for comparison
  - Methods: `fit()`, `predict()`, `predict_proba()`
  - Uses scikit-learn SVC

**Mathematical Formulation**:
```
minimize: (1/2)||w||² + C(Σξᵢⱼ + Σζᵢⱼ)
subject to:
  wᵀ(xᵢ - xⱼ) ≥ 1 - ξᵢⱼ  (ordering constraint)
  |wᵀ(xᵢ - xⱼ)| ≤ ε + ζᵢⱼ  (similarity constraint)
```

**Usage**:
```python
from ranking_svm import RankingSVM, BinarySVM

# Train Ranking SVM
ranker = RankingSVM(C=0.01, epsilon=0.5)
ranker.fit(X_diff_ordered)  # X_diff: difference vectors

# Predict scores
scores = ranker.predict(features)  # (n_samples,) scores

# Predict pairwise rankings
preds = ranker.predict_pairwise(features, test_pairs)
```

### 5. `evaluation.py` - Evaluation Metrics
**Purpose**: Compute accuracy and comparison metrics

**Key Functions**:
- `pairwise_accuracy()`: Compute % correctly ordered pairs
- `pairwise_accuracy_batch()`: Batch computation using difference vectors
- `compute_margins()`: Get margin statistics (confidence)
- `compare_ranking_vs_binary()`: Compare with baseline
- `zero_shot_accuracy()`: Evaluate zero-shot learning
- `compute_per_attribute_accuracy()`: Per-attribute breakdown

**Usage**:
```python
from evaluation import pairwise_accuracy, compute_per_attribute_accuracy

# Compute accuracy
acc = pairwise_accuracy(w, features, test_pairs)
print(f"Accuracy: {acc:.4f}")  # e.g., 0.9581

# Per-attribute accuracy
accs_per_attr = compute_per_attribute_accuracy(w, features, test_pairs_dict)
for attr, acc in accs_per_attr.items():
    print(f"{attr}: {acc:.4f}")
```

### 6. `visualization.py` - Visualization Utilities
**Purpose**: Create plots and visualizations

**Key Functions**:
- `plot_accuracy_comparison()`: Ranking SVM vs Binary baseline
- `plot_per_attribute_performance()`: Bar chart of per-attribute accuracy
- `plot_improvement_over_baseline()`: Relative improvement percentages
- `plot_score_distributions()`: Histogram of margin scores
- `plot_feature_importance()`: Top features per attribute

**Usage**:
```python
from visualization import plot_accuracy_comparison, plot_per_attribute_performance

# Compare with baseline
plot_accuracy_comparison(
    ranking_acc={'natural': 0.94, 'open': 0.97, ...},
    binary_acc={'natural': 0.37, 'open': 0.17, ...},
    save_path='./plots/comparison.png'
)

# Per-attribute performance
plot_per_attribute_performance(
    accuracies={'natural': 0.9434, 'open': 0.9686, ...},
    save_path='./plots/per_attribute.png'
)
```

### 7. `pipeline.py` - Main Pipeline
**Purpose**: Orchestrate complete workflow end-to-end

**Key Class**: `RankingAttributesPipeline`
- Phase 1: Feature extraction
- Phase 2: Data preparation  
- Phase 3: Ranking SVM training
- Phase 4: Binary SVM baseline
- Phase 5: Evaluation
- Phase 6: Visualization

**Usage**:
```python
from pipeline import RankingAttributesPipeline

# Initialize pipeline
pipeline = RankingAttributesPipeline()

# Run complete pipeline
results = pipeline.run(
    image_paths=image_paths,
    category_orderings=orderings,
    split_ratio=0.8
)

# Results include per-attribute accuracies
print(results)
```

## 📊 Complete Workflow Example

```python
import numpy as np
from pathlib import Path

# Import modules
from features import extract_features_batch, FeatureNormalizer
from data_prep import construct_ordered_pairs, prepare_batch_data
from ranking_svm import RankingSVM
from evaluation import pairwise_accuracy
from visualization import plot_per_attribute_performance

# 1. Extract features
image_paths = list(Path('./data/OSR/outdoor').glob('*/*.jpg'))
features = extract_features_batch(image_paths)

# 2. Normalize
normalizer = FeatureNormalizer()
features = normalizer.fit_transform(features)

# 3. Prepare data
orderings = {'natural': [0, 2, 1, 3, 5, 4, 7, 6]}  # Example
pairs = construct_ordered_pairs(orderings, max_pairs_per_attribute=3000)
X_diff, _ = prepare_batch_data(features, pairs['natural'][:2400])

# 4. Train model
ranker = RankingSVM(C=0.01, epsilon=0.5)
ranker.fit(X_diff)

# 5. Evaluate
test_pairs = pairs['natural'][2400:]
acc = pairwise_accuracy(ranker.w, features, test_pairs)
print(f"Accuracy: {acc:.4f}")

# 6. Visualize
accuracies = {'natural': acc}
plot_per_attribute_performance(accuracies, save_path='./result.png')
```

## 🔄 Modularization Benefits

### Over Monolithic Notebook

| Aspect | Notebook | Modularized |
|--------|----------|------------|
| Reusability | Limited | ✅ Import any function |
| Testing | Difficult | ✅ Unit test each module |
| Collaboration | Hard to merge | ✅ Work on different modules |
| Reproducibility | Cell order matters | ✅ Linear function calls |
| Debugging | Time-consuming | ✅ Isolated components |
| Production | Not suitable | ✅ Ready for deployment |

## 📦 Dependencies

```
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.23.0
scikit-image>=0.17.0
matplotlib>=3.3.0
opencv-python
cvxopt>=1.2.5
tqdm>=4.50.0
seaborn
```

Install all:
```bash
pip install numpy pandas scikit-learn scikit-image matplotlib opencv-python cvxopt tqdm seaborn
```

## 🚀 Quick Start

### As a Python Package

```python
# Set PYTHONPATH
import sys
sys.path.append('/path/to/modules')

from utils import ATTRIBUTES, TRAINING_PARAMS
from features import extract_features_batch
from ranking_svm import RankingSVM
from evaluation import pairwise_accuracy
from visualization import plot_per_attribute_performance

# Use modules...
```

### As Command-Line Scripts

```bash
# Extract features
python features.py --input_dir data/OSR --output result_features.npy

# Train model
python ranking_svm.py --train --config config.json

# Evaluate
python evaluation.py --model model.pkl --test_data test.npy
```

## 📝 Module Dependencies

```
pipeline.py
├── utils.py
├── features.py
├── data_prep.py
├── ranking_svm.py
├── evaluation.py
└── visualization.py

Each module is independent and can be used separately.
```

## 🎯 Key Advantages

1. **Modularity**: Each module handles one responsibility
2. **Reusability**: Import and use any function or class
3. **Testability**: Easy to write unit tests
4. **Documentation**: Each function fully documented
5. **Maintainability**: Easy to update or extend
6. **Performance**: Optimized batch operations
7. **Reproducibility**: Deterministic with saved configurations

## 📚 Extension Points

### Add Custom Feature Extractor
```python
# In features.py, add:
def extract_custom_features(image):
    # Your implementation
    return features
```

### Add New Ranking Method
```python
# In ranking_svm.py, add:
class RankNet:
    def fit(self, X_diff):
        # Neural network implementation
        pass
```

### Add Evaluation Metric
```python
# In evaluation.py, add:
def custom_metric(predictions, ground_truth):
    # Your metric
    return score
```

## 🔗 Converting Notebook to Modules

**Process Used**:
1. Identified distinct functionality sections in notebook
2. Extracted code into logical modules
3. Added proper documentation and type hints
4. Refactored for reusability and clarity
5. Removed notebook-specific cells (like mount_drive)
6. Centralized configuration and constants

**From 50 notebook cells → 7 focused Python modules**

---

**Status**: ✅ Production-Ready Modular Implementation
**Version**: 1.0.0
**Last Updated**: November 2025
