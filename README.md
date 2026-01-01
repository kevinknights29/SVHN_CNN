# SVHN CNN - Street View House Number Detection

Deep learning project for detecting and classifying multi-digit sequences from street view images using Convolutional Neural Networks.

![Detection Examples](figs/13.png)

## Overview

This project implements multiple CNN architectures to predict sequences of up to 4 digits from the Street View House Number (SVHN) dataset. It includes both training pipelines and a multi-scale sliding window detection system for real-world digit localization.

**Key Features:**
- Four different CNN architectures (Custom, Improved, Alternative with STN, VGG-16)
- Multi-task learning with 6 output heads
- Multi-scale pyramid detection with sliding windows
- Comprehensive training reports and visualizations

**Dataset:** ~150,000 samples from the [SVHN dataset](http://ufldl.stanford.edu/housenumbers/)

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management (Python 3.14+):

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Install ML frameworks for training
uv add tensorflow keras matplotlib
```

**Core dependencies:** `opencv-python`, `scipy`, `h5py`, `tensorflow`, `keras`, `matplotlib`

**Development:** `ruff` for linting

## Quick Start

### 1. Setup Dataset

Download the pre-processed `.h5` dataset files and configure paths:

```bash
# Download datasets from:
# https://my.pcloud.com/publink/show?code=kZsxCO7ZdnzmYqXWO6ydqkC5kz114f5zUUaV

# Place files in datasets/ directory
# Update config.ini with dataset paths
```

### 2. Train Models

All training scripts are in the `models/` directory (PyTorch):

```bash
# Train the recommended Alternative CNN with STN (BEST: 86.61%)
python models/train_alternative_cnn_pytorch.py

# Train Improved CNN with modern architecture (81.72%)
python models/train_improved_cnn_pytorch.py

# Train pre-trained VGG-16 (78.02%)
python models/train_vgg16_pretrained_pytorch.py

# Train VGG-16 from scratch (77.25%)
python models/train_vgg16_scratch_pytorch.py

# Train custom designed CNN baseline (8.77%)
python models/train_custom_cnn_pytorch.py
```

Each training script will:
- Load and preprocess the SVHN dataset
- Train the model with optimized hyperparameters
- Generate comprehensive training reports and plots
- Save the best model checkpoint to `saved_models/`

See [models/README.md](models/README.md) for detailed documentation on each architecture and training options.

### 3. Run Detection

Use the detection pipeline to find and classify digits in images:

```bash
# Run detection on a specific image
python detection.py
```

The detection system uses a 10-level image pyramid with sliding windows to handle varying digit sizes and scales.

## Current Progress

### Latest Training Results

#### Alternative CNN with STN (Recommended) ⭐ **NEW BEST!**
- **Test Sequence Accuracy:** 86.61%
- **Validation Sequence Accuracy:** 92.62%
- **Training Sequence Accuracy:** 98.82%
- **Status:** Best performing model with Spatial Transformer Network for spatial invariance

**Per-Component Test Accuracy:**
- Number of Digits: 97.39%
- Digit Position 1: 94.90%
- Digit Position 2: 92.07%
- Digit Position 3: 96.22%
- Digit Position 4: 99.52%
- Has Digits: 99.55%

#### Improved CNN
- **Test Sequence Accuracy:** 81.72%
- **Validation Sequence Accuracy:** 90.20%
- **Training Sequence Accuracy:** 96.99%
- **Status:** Strong performer with ResNet + Inception + SE blocks

**Per-Component Test Accuracy:**
- Number of Digits: 96.88%
- Digit Position 1: 92.45%
- Digit Position 2: 89.24%
- Digit Position 3: 95.04%
- Digit Position 4: 99.11%
- Has Digits: 99.47%

#### Pre-trained VGG-16
- **Test Sequence Accuracy:** 78.02%
- **Validation Sequence Accuracy:** 88.26%
- **Training Sequence Accuracy:** 92.90%
- **Status:** Good performance with transfer learning

#### Custom Designed CNN (Baseline)
- **Test Sequence Accuracy:** 8.77%
- **Validation Sequence Accuracy:** 53.48%
- **Training Sequence Accuracy:** 52.87%
- **Status:** Baseline model for comparison

### Training Reports

Detailed automated training reports are available in `plots/`:
- [VGG-16 Pre-trained Report](plots/training_report_vgg16_PreTrain.md)
- [Custom Design Report](plots/training_report_customDesign.md)

These reports include:
- Executive summaries with key metrics
- Automated analysis (overfitting detection, convergence checks)
- Per-digit position performance ranking
- Actionable recommendations

## Project Structure

```
SVHN_CNN/
├── models/                                    # PyTorch training scripts (see models/README.md)
│   ├── pytorch_models.py                      # Model architectures (STN, Alternative, Improved, VGG16)
│   ├── pytorch_utils.py                       # PyTorch training utilities
│   ├── train_alternative_cnn_pytorch.py       # Alternative CNN with STN (BEST: 86.61%)
│   ├── train_improved_cnn_pytorch.py          # Improved CNN (81.72%)
│   ├── train_vgg16_pretrained_pytorch.py      # VGG-16 pretrained (78.02%)
│   ├── train_vgg16_scratch_pytorch.py         # VGG-16 scratch (77.25%)
│   └── train_custom_cnn_pytorch.py            # Custom CNN baseline (8.77%)
├── src/svhn_cnn/             # Core library code
│   ├── data/                 # Data loading and preprocessing
│   └── utils/                # Helper functions
├── datasets/                 # .h5 dataset files (gitignored)
├── saved_models/             # Trained model checkpoints (.pt files)
├── plots/                    # Training visualizations and reports
├── metrics/                  # Performance metrics (pickle files)
├── input/                    # Images for detection
├── output/                   # Detection results
├── detection.py              # Multi-scale detection pipeline
└── config.ini               # Dataset configuration
```

## Model Architectures

### 1. Alternative CNN with STN ⭐ **BEST** - 86.61% Test Accuracy
- Spatial Transformer Network for spatial invariance
- 4 convolutional blocks (32 → 256 filters)
- Batch normalization and dropout (0.25)
- 2 fully connected layers (1024 → 1024)

### 2. Improved CNN - 81.72% Test Accuracy
- ResNet-style skip connections with SE blocks
- Inception-style multi-scale feature extraction
- Progressive feature refinement through 5 stages
- 2 fully connected layers (1024 → 1024)

### 3. VGG-16 (Pre-trained) - 78.02% Test Accuracy
- VGG-16 with ImageNet weights
- Transfer learning approach
- Custom FC layers (1024 → 1024)

### 4. VGG-16 (Scratch) - 77.25% Test Accuracy
- Standard VGG-16 architecture
- Random weight initialization
- Custom FC layers (512 → 512)

### 5. Custom Designed CNN (Baseline) - 8.77% Test Accuracy
- 8 convolutional blocks (16 → 512 filters)
- Batch normalization and strategic dropout
- 3 fully connected layers (2048 → 1024 → 1024)

All models use **multi-task learning** with 6 output heads:
1. Number of digits (0-4)
2-5. Individual digit predictions (11 classes each: 0-9 + blank)
6. Binary classifier (has digits: yes/no)

## Detection Pipeline

The detection system (`detection.py`) implements:

1. **Image Localization:** Edge detection (Sobel) + morphology for candidate regions
2. **Multi-Scale Pyramid:** 10 levels with 0.8x scale reduction
3. **Sliding Window:** Adaptive step sizes based on pyramid level
4. **CNN Classification:** Model predicts digits in each window
5. **Heatmap Fusion:** Probability accumulation across scales
6. **Bounding Box Extraction:** Contour detection on final heatmap
7. **Refinement:** Re-classification of merged regions

## Development

### Linting

```bash
# Check code
uv run ruff check .

# Format code
uv run ruff format .
```

### Configuration

Edit `config.ini` to set dataset paths:

```ini
[PATHS]
train_dir = /path/to/train
test_dir = /path/to/test
extra_dir = /path/to/extra
```

## Dataset Information

**Format:** HDF5 (.h5) for efficient Python access

**Structure:**
- `digits`/`digitsBW`: 48x48 resized images (RGB/grayscale)
- `negdigits`/`negdigitsBW`: Negative samples (non-digit regions)
- `labs5`: Labels `[num_digits, digit1, digit2, digit3, digit4, is_blank]`

**Preprocessing:**
- Images resized to 48x48 pixels
- Per-image mean subtraction
- Optional feature normalization (dataset mean/std)

**Downloads:**
- [Pre-processed .h5 files](https://my.pcloud.com/publink/show?code=kZsxCO7ZdnzmYqXWO6ydqkC5kz114f5zUUaV)
- [Original .mat files](http://ufldl.stanford.edu/housenumbers/)

## References

This project is inspired by and builds upon:

- Yuval Netzer et al. - "Reading Digits in Natural Images with Unsupervised Feature Learning" ([paper](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf))
- Original implementation by [beeps82](https://github.com/beeps82/SVHN_CNN)

