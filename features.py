"""
Feature Extraction Module

Extracts visual features from images using GIST descriptors, Gabor filters,
and color histograms. Provides a 557-dimensional feature vector per image.

Components:
- GIST descriptors (512-dim): Global image structure
- Color histograms (45-dim): Lab color distribution
"""

import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional
from pathlib import Path
import os
from tqdm import tqdm

# ============================================================================
# GABOR FILTER CONSTRUCTION
# ============================================================================

def build_gabor_filters(n_orients: int = 8, n_scales: int = 4, 
                        kernel_size: int = 32) -> List[np.ndarray]:
    """
    Build Gabor filter bank.
    
    Creates Gabor filters at multiple scales and orientations for texture
    and edge detection at different resolutions.
    
    Parameters
    ----------
    n_orients : int
        Number of orientations (default: 8 = 0°, 45°, ..., 315°)
    n_scales : int
        Number of scales (default: 4)
    kernel_size : int
        Kernel size (default: 32x32)
        
    Returns
    -------
    filters : list of ndarray
        List of Gabor filters (n_scales * n_orients filters)
        
    Formula
    -------
    Gabor filter: G(x, y) = exp(-pi * ((x'^2/sigma_x^2) + (y'^2/sigma_y^2))) * 
                              cos(2 * pi * f * x')
    where x' = x*cos(theta) + y*sin(theta)
          y' = -x*sin(theta) + y*cos(theta)
    """
    filters = []
    kernel_half = kernel_size // 2
    
    # Lambda values for different scales (wavelengths)
    lambdas = [kernel_size / (2 ** (i + 1)) for i in range(n_scales)]
    
    for scale, lambda_val in enumerate(lambdas):
        for orient in range(n_orients):
            # Orientation angle
            theta = orient * np.pi / n_orients
            
            # Create filter
            gabor = np.zeros((kernel_size, kernel_size))
            
            for x in range(-kernel_half, kernel_half):
                for y in range(-kernel_half, kernel_half):
                    # Rotate coordinates
                    x_theta = x * np.cos(theta) + y * np.sin(theta)
                    y_theta = -x * np.sin(theta) + y * np.cos(theta)
                    
                    # Gabor equation
                    sigma = lambda_val / np.pi
                    real = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2)) * \
                           np.cos(2 * np.pi * x_theta / lambda_val)
                    
                    gabor[y + kernel_half, x + kernel_half] = real
            
            # Normalize
            gabor = gabor / np.sum(np.abs(gabor))
            filters.append(gabor)
    
    return filters


# ============================================================================
# GIST FEATURE EXTRACTION
# ============================================================================

def extract_gist_features(image: np.ndarray, 
                         filters: Optional[List[np.ndarray]] = None,
                         n_blocks: int = 4) -> np.ndarray:
    """
    Extract GIST features from image.
    
    GIST captures global image statistics using Gabor filter responses,
    preserving spatial information through block pooling.
    
    Parameters
    ----------
    image : ndarray (h, w) or (h, w, 3)
        Input image (grayscale or RGB)
    filters : list of ndarray, optional
        Pre-built Gabor filters
    n_blocks : int
        Number of spatial blocks (n_blocks x n_blocks grid)
        
    Returns
    -------
    gist_features : ndarray (n_filters * n_blocks^2,)
        GIST feature vector
        
    Process
    -------
    1. Convert to grayscale if needed
    2. Apply each Gabor filter
    3. Divide into n_blocks x n_blocks regions
    4. Pool (mean) filter responses in each region
    5. Concatenate all pooled responses
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize image
    image = image.astype(np.float32) / 255.0
    
    # Build filters if not provided
    if filters is None:
        filters = build_gabor_filters(n_orients=8, n_scales=4, kernel_size=32)
    
    # Apply filters and extract spatial blocks
    gist = []
    h, w = image.shape
    block_h = h // n_blocks
    block_w = w // n_blocks
    
    for gabor_filter in filters:
        # Convolve image with filter
        response = cv2.filter2D(image, -1, gabor_filter)
        
        # Spatial pooling
        for i in range(n_blocks):
            for j in range(n_blocks):
                y_start = i * block_h
                x_start = j * block_w
                block = response[y_start:y_start + block_h, 
                                x_start:x_start + block_w]
                # Mean pooling
                gist.append(np.mean(np.abs(block)))
    
    return np.array(gist)


# ============================================================================
# COLOR HISTOGRAM EXTRACTION
# ============================================================================

def extract_color_histogram(image: np.ndarray, n_bins: int = 15) -> np.ndarray:
    """
    Extract color histogram from image in Lab color space.
    
    Parameters
    ----------
    image : ndarray (h, w, 3)
        Input RGB image
    n_bins : int
        Number of bins per channel (default: 15, total: 15*3=45)
        
    Returns
    -------
    hist : ndarray (n_bins * 3,)
        Concatenated histograms for L, a, b channels
        
    Rationale
    ---------
    Lab color space is perceptually uniform:
    - L channel: Lightness (0-100)
    - a channel: Green-Red (-127 to 127)
    - b channel: Blue-Yellow (-127 to 127)
    """
    # Convert to Lab color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Extract histograms for each channel
    hist = []
    for i in range(3):
        channel = image_lab[:, :, i]
        h = cv2.calcHist([channel], [0], None, [n_bins], 
                        [0, 256] if i == 0 else [-128, 128])
        # Normalize histogram
        h = h / np.sum(h)
        hist.extend(h.flatten())
    
    return np.array(hist)


# ============================================================================
# BATCH FEATURE EXTRACTION
# ============================================================================

def extract_features_batch(image_paths: List[str], 
                          resize: Tuple[int, int] = (256, 256),
                          batch_size: int = 32) -> np.ndarray:
    """
    Extract features from batch of images.
    
    Parameters
    ----------
    image_paths : list of str
        Paths to images
    resize : tuple
        Target image size (height, width)
    batch_size : int
        Batch size for progress reporting
        
    Returns
    -------
    features : ndarray (n_images, 557)
        Feature matrix
    """
    # Pre-build Gabor filters
    gabor_filters = build_gabor_filters(n_orients=8, n_scales=4, kernel_size=32)
    
    n_images = len(image_paths)
    features = []
    
    print(f"Extracting features from {n_images} images...")
    
    for idx, img_path in enumerate(tqdm(image_paths, total=n_images)):
        if not os.path.exists(img_path):
            print(f"Warning: File not found: {img_path}")
            continue
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load: {img_path}")
            continue
        
        # Resize
        image = cv2.resize(image, (resize[1], resize[0]))
        
        # Extract GIST features
        gist = extract_gist_features(image, gabor_filters, n_blocks=4)
        
        # Extract color histogram
        color = extract_color_histogram(image, n_bins=15)
        
        # Concatenate features
        feature_vec = np.concatenate([gist, color])
        features.append(feature_vec)
    
    features = np.array(features)
    
    print(f"\n✓ Extracted {len(features)} feature vectors")
    print(f"✓ Feature dimension: {features.shape[1]}")
    
    return features


# ============================================================================
# FEATURE NORMALIZATION
# ============================================================================

class FeatureNormalizer:
    """
    Fit and apply feature normalization (standardization).
    
    Ensures zero mean and unit variance for each feature dimension.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> None:
        """Fit normalizer on training data."""
        self.scaler.fit(X)
        self.fitted = True
        print(f"✓ Normalizer fitted on {X.shape[0]} samples")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    print("=" * 80)
    print("Feature Extraction Module")
    print("=" * 80)
    print("\nProvides:")
    print("  - Gabor filter construction")
    print("  - GIST feature extraction")
    print("  - Color histogram extraction")
    print("  - Batch feature processing")
    print("\nFeature Dimensions:")
    print("  - GIST: 512 (8 orientations × 4 scales × 16 spatial blocks)")
    print("  - Color: 45 (3 channels × 15 bins)")
    print("  - Total: 557")
