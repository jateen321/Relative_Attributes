# Modularized Python Code - Complete Guide

## 📊 Quick Overview

Your Jupyter notebook has been fragmented into **7 focused Python modules** with ~2,550 lines of well-documented code.

```
Notebook (50 cells, 3.9 MB)  →  FRAGMENTED INTO  →  7 Modules (~2,550 lines)

✅ All functionality preserved
✅ Code modularized for reusability
✅ Full documentation with type hints
✅ Ready for production deployment
```

## 📦 Module Breakdown

### 1. **utils.py** (~200 lines)
**What it does**: Configuration and utilities

```python
from utils import ATTRIBUTES, TRAINING_PARAMS, save_pickle, load_pickle

# Configuration
print(ATTRIBUTES)  # ['natural', 'open', 'perspective', ...]
print(TRAINING_PARAMS['C_optimal'])  # 0.01

# Save/load models
save_pickle(model, './models/ranker.pkl')
loaded_model = load_pickle('./models/ranker.pkl')
```

**Contains**:
- Dataset paths (OSR_PATH, LFW_PATH)
- Constants (ATTRIBUTES, OSR_CATEGORIES)
- Utility functions (normalize_features, validate_features)
- Pretty printing functions

---

### 2. **features.py** (~400 lines)
**What it does**: Extract 557-dimensional features from images

```python
from features import extract_features_batch, FeatureNormalizer, build_gabor_filters

# Extract features (GIST + Color)
image_paths = ['img1.jpg', 'img2.jpg', ...]
features = extract_features_batch(image_paths)  # (n_images, 557)

# Normalize
normalizer = FeatureNormalizer()
X_norm = normalizer.fit_transform(features)

# Components:
# - 512 GIST descriptors (4 scales × 8 orientations × 4 blocks)
# - 45 Color histograms (3 channels × 15 bins)
```

**Functions**:
- `build_gabor_filters()` - Create Gabor filter bank
- `extract_gist_features()` - Extract GIST descriptors
- `extract_color_histogram()` - Extract Lab color features
- `extract_features_batch()` - Batch process images

**Class**:
- `FeatureNormalizer` - Standardize features

---

### 3. **data_prep.py** (~300 lines)
**What it does**: Prepare pairwise training data

```python
from data_prep import construct_ordered_pairs, construct_similarity_pairs, prepare_batch_data

# Define attribute orderings (index = strength, lower = weaker, higher = stronger)
orderings = {
    'natural': [5, 3, 1, 0, 7, 2, 4, 6],
    'open': [2, 1, 5, 6, 0, 3, 4, 7]
}

# Generate training pairs
ordered_pairs = construct_ordered_pairs(orderings, max_pairs_per_attribute=3000)
similar_pairs = construct_similarity_pairs(orderings, epsilon=0.5, max_pairs_per_attribute=1000)

# Convert to SVM format (difference vectors)
X_diff, pair_indices = prepare_batch_data(features, ordered_pairs['natural'])
# X_diff shape: (n_pairs, 557)
```

**Functions**:
- `construct_ordered_pairs()` - Create category-level orderings
- `construct_similarity_pairs()` - Create adjacent pairs
- `image_level_pairs()` - Convert to image-level pairs
- `prepare_batch_data()` - Create difference vectors

---

### 4. **ranking_svm.py** (~400 lines)
**What it does**: Train Ranking SVM and Binary SVM baseline

```python
from ranking_svm import RankingSVM, BinarySVM

# Train Ranking SVM
ranker = RankingSVM(C=0.01, epsilon=0.5)
ranker.fit(X_diff_ordered)  # X_diff: (n_pairs, 557)

# Predict
scores = ranker.predict(features)  # (n_samples,)
predictions = ranker.predict_pairwise(features, test_pairs)

# Binary baseline
baseline = BinarySVM(C=1.0)
baseline.fit(features, binary_labels)
```

**Classes**:
- `RankingSVM` - Main ranking model
  - `fit(X_diff)` - Train on difference vectors
  - `predict(X)` - Get attribute scores
  - `predict_pairwise(X, pairs)` - Rank pairs
  
- `BinarySVM` - Baseline binary classifier
  - `fit(X, y)` - Train binary model
  - `predict(X)` - Binary predictions
  - `predict_proba(X)` - Confidence scores

---

### 5. **evaluation.py** (~400 lines)
**What it does**: Compute accuracy and evaluation metrics

```python
from evaluation import pairwise_accuracy, compute_per_attribute_accuracy

# Pairwise accuracy
accuracy = pairwise_accuracy(w, features, test_pairs)  # 0.9581 (95.81%)

# Per-attribute breakdown
accs = compute_per_attribute_accuracy(w, features, test_pairs_dict)
# {'natural': 0.9434, 'open': 0.9686, ...}

# Margin analysis
min_margin, mean_margin, std_margin = compute_margins(w, X_diff)

# Zero-shot learning
zero_shot_acc = zero_shot_accuracy(learned_attrs, test_cats, descriptions, k=1)

# Comparison with baseline
results = compare_ranking_vs_binary(ranker, baseline, features, test_pairs, ordering)
# Returns: {'ranking_svm_accuracy': 0.95, 'binary_svm_accuracy': 0.52, ...}
```

**Functions**:
- `pairwise_accuracy()` - Fraction of correctly ordered pairs
- `pairwise_accuracy_batch()` - Vectorized computation
- `compute_margins()` - Get margin statistics
- `compute_per_attribute_accuracy()` - Per-attribute breakdown
- `zero_shot_accuracy()` - Unseen category recognition
- `compare_ranking_vs_binary()` - Baseline comparison

---

### 6. **visualization.py** (~350 lines)
**What it does**: Create publication-quality plots

```python
from visualization import (plot_accuracy_comparison, 
                          plot_per_attribute_performance,
                          plot_improvement_over_baseline)

# Compare Ranking SVM vs Binary
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

# Relative improvement
plot_improvement_over_baseline(ranking_scores, baseline_scores, attributes)

# Score distributions
plot_score_distributions(correct_scores, incorrect_scores, 'natural')

# Feature importance
plot_feature_importance({'natural': w_natural, 'open': w_open, ...}, top_k=10)
```

**Functions**:
- `plot_accuracy_comparison()` - Bar charts
- `plot_per_attribute_performance()` - Attribute-wise accuracy
- `plot_improvement_over_baseline()` - Relative improvements
- `plot_score_distributions()` - Margin histograms
- `plot_feature_importance()` - Top features per attribute

---

### 7. **pipeline.py** (~500 lines)
**What it does**: End-to-end orchestration

```python
from pipeline import RankingAttributesPipeline

# Initialize
pipeline = RankingAttributesPipeline()

# Run complete pipeline
results = pipeline.run(
    image_paths=image_paths,
    category_orderings=orderings,
    split_ratio=0.8
)

# Phases:
# 1. Extract features
# 2. Prepare pairwise data
# 3. Train Ranking SVM
# 4. Train Binary baseline
# 5. Evaluate models
# 6. Generate visualizations
```

**Class**: `RankingAttributesPipeline`
- `extract_features()` - Phase 1
- `prepare_training_data()` - Phase 2
- `train_ranking_svms()` - Phase 3
- `train_binary_baselines()` - Phase 4
- `evaluate()` - Phase 5
- `visualize_results()` - Phase 6
- `run()` - Execute all phases

---

## 🔄 Usage Patterns

### Pattern 1: Use Full Pipeline
```python
from pipeline import RankingAttributesPipeline

pipeline = RankingAttributesPipeline()
results = pipeline.run(image_paths, category_orderings)
```

### Pattern 2: Use Individual Modules
```python
from features import extract_features_batch
from ranking_svm import RankingSVM
from evaluation import pairwise_accuracy

features = extract_features_batch(image_paths)
ranker = RankingSVM()
ranker.fit(X_diff_train)
acc = pairwise_accuracy(ranker.w, features, test_pairs)
```

### Pattern 3: Custom Workflow
```python
# Extract features
X = extract_features_batch(images)

# Prepare data
pairs = construct_ordered_pairs(orderings)
X_diff, _ = prepare_batch_data(X, pairs)

# Train with custom parameters
ranker = RankingSVM(C=0.01, epsilon=0.5)
ranker.fit(X_diff)

# Evaluate and visualize
acc = pairwise_accuracy(ranker.w, X, test_pairs)
plot_per_attribute_performance({'attr': acc})
```

---

## 📚 Complete Example

```python
import numpy as np
from pathlib import Path

# ========== IMPORT ==========
from features import extract_features_batch, FeatureNormalizer
from data_prep import construct_ordered_pairs, prepare_batch_data, split_pairs_train_test
from ranking_svm import RankingSVM
from evaluation import pairwise_accuracy
from visualization import plot_per_attribute_performance

# ========== LOAD DATA ==========
image_paths = list(Path('./data/OSR/outdoor').glob('*/*.jpg'))

# ========== EXTRACT FEATURES ==========
print("Extracting features...")
features = extract_features_batch(image_paths, resize=(256, 256))
# Output: (2688, 557) for OSR dataset

# ========== NORMALIZE ==========
normalizer = FeatureNormalizer()
features = normalizer.fit_transform(features)

# ========== PREPARE DATA ==========
# Define category orderings (categories ranked by attribute strength)
orderings = {
    'natural': [5, 3, 1, 0, 7, 2, 4, 6],     # Example ordering
    'open': [2, 1, 5, 6, 0, 3, 4, 7]
}

# Generate pairs
pairs_dict = construct_ordered_pairs(orderings, max_pairs_per_attribute=3000)

# Prepare SVM input
pairs = pairs_dict['natural']
train_pairs, test_pairs = split_pairs_train_test(pairs, train_ratio=0.8)

X_diff_train, _ = prepare_batch_data(features, train_pairs)
X_diff_test, _ = prepare_batch_data(features, test_pairs)

# ========== TRAIN ==========
print("Training Ranking SVM...")
ranker = RankingSVM(C=0.01, epsilon=0.5)
ranker.fit(X_diff_train)

# ========== EVALUATE ==========
print("Evaluating...")
train_acc = pairwise_accuracy(ranker.w, features, train_pairs)
test_acc = pairwise_accuracy(ranker.w, features, test_pairs)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ========== VISUALIZE ==========
print("Visualizing...")
plot_per_attribute_performance(
    {'natural': test_acc},
    save_path='./result.png'
)

print("✓ Complete!")
```

---

## 🎯 Key Improvements Over Notebook

| Aspect | Notebook | Modularized Code |
|--------|----------|------------------|
| **Reusability** | Copy cells | `from modules import function` |
| **Testing** | Run all cells | Unit test individual functions |
| **Debugging** | Trace through 50 cells | Debug one module at a time |
| **Collaboration** | Hard to merge | Easy to work on different modules |
| **Maintenance** | Update cell, re-run notebook | Update function, import updated module |
| **Production** | Not suitable | Ready to deploy |
| **Documentation** | Markdown cells | Docstrings + MODULES_README.md |
| **Type hints** | None | Full type hints on all functions |

---

## 🚀 Deployment

### As Python Package
```bash
# Create setup.py
python setup.py install

# Use in your project
from relative_attributes import RankingSVM, extract_features_batch
```

### As Docker Container
```dockerfile
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY *.py /app/
WORKDIR /app
```

### As API Service
```python
from flask import Flask, request
from ranking_svm import RankingSVM

app = Flask(__name__)
ranker = RankingSVM()

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json['features']
    scores = ranker.predict(features)
    return {'scores': scores.tolist()}
```

---

## 📖 Documentation

- **Module Details**: See `MODULES_README.md`
- **API Reference**: Docstrings in each .py file
- **Examples**: See "Complete Example" section above
- **Type Hints**: Full type hints in all functions

---

## ✅ What You Get

- ✅ 7 focused, reusable Python modules
- ✅ ~2,550 lines of well-documented code
- ✅ Full type hints and docstrings
- ✅ Complete feature extraction pipeline
- ✅ Ranking SVM implementation
- ✅ Comprehensive evaluation metrics
- ✅ Publication-ready visualizations
- ✅ End-to-end pipeline orchestration
- ✅ Production-ready code
- ✅ Ready for deployment

---

## 🔗 File Relationships

```
pipeline.py (orchestrator)
    ↓
├── utils.py (configuration)
├── features.py (extract features)
├── data_prep.py (prepare data)
├── ranking_svm.py (train models)
├── evaluation.py (evaluate metrics)
└── visualization.py (create plots)

Each module can be used independently!
```

---

**Status**: ✅ Complete Modularization  
**Total Code**: ~2,550 lines  
**Modules**: 7  
**Documentation**: Complete  
**Production Ready**: Yes  

Ready to use, test, deploy! 🚀
