# SVHN CNN - Street View House Number Detection

Deep learning project for detecting and classifying multi-digit sequences from street view images using Convolutional Neural Networks.

![Detection Examples](figs/13.png)

## Overview

This project implements multiple CNN architectures to predict sequences of up to 4 digits from the Street View House Number (SVHN) dataset. It includes both training pipelines and a multi-scale sliding window detection system for real-world digit localization.

**Key Features:**
- Three different CNN architectures (Custom, VGG-16, Pre-trained VGG-16)
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

All training scripts are in the `models/` directory:

```bash
# Train the recommended pre-trained VGG-16 model
python models/train_vgg16_pretrained.py

# Train custom designed CNN
python models/train_custom_cnn.py

# Train VGG-16 from scratch
python models/train_vgg16_scratch.py
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

#### Pre-trained VGG-16 (Recommended)
- **Test Accuracy:** 78.20%
- **Validation Accuracy:** 88.57%
- **Training Accuracy:** 98.71%
- **Status:** Shows some overfitting; could benefit from additional regularization

**Per-Component Test Accuracy:**
- Number of Digits: 96.63%
- Digit Position 1: 90.83%
- Digit Position 2: 86.71%
- Digit Position 3: 94.13%
- Digit Position 4: 99.03%

![VGG16 PreTrain Accuracy](plots/all_digits_accuracy_vgg16_PreTrain.png)
![VGG16 PreTrain Loss](plots/all_losses_vgg16_PreTrain.png)

#### Custom Designed CNN
- **Test Accuracy:** 35.33%
- **Validation Accuracy:** 66.37%
- **Training Accuracy:** 68.40%
- **Status:** Converged without significant overfitting

**Per-Component Test Accuracy:**
- Number of Digits: 95.04%
- Digit Position 1: 67.28%
- Digit Position 2: 52.44%
- Digit Position 3: 81.49%
- Digit Position 4: 98.78%

![Custom Design Accuracy](plots/all_digits_accuracy_customDesign.png)

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
├── models/                    # Training scripts (see models/README.md)
│   ├── train_custom_cnn.py
│   ├── train_vgg16_scratch.py
│   ├── train_vgg16_pretrained.py
│   └── training_utils.py     # Shared utilities
├── src/svhn_cnn/             # Core library code
│   ├── data/                 # Data loading and preprocessing
│   └── utils/                # Helper functions
├── datasets/                 # .h5 dataset files (gitignored)
├── saved_models/             # Trained model checkpoints
├── plots/                    # Training visualizations and reports
├── metrics/                  # Performance metrics (pickle files)
├── input/                    # Images for detection
├── output/                   # Detection results
├── detection.py              # Multi-scale detection pipeline
└── config.ini               # Dataset configuration
```

## Model Architectures

### 1. Custom Designed CNN
- 8 convolutional blocks (16 → 512 filters)
- Batch normalization and strategic dropout
- 3 fully connected layers (2048 → 1024 → 1024)

### 2. VGG-16 (Scratch)
- Standard VGG-16 architecture
- Random weight initialization
- Custom FC layers (512 → 512)

### 3. VGG-16 (Pre-trained) ⭐ Recommended
- VGG-16 with ImageNet weights
- Transfer learning approach
- Custom FC layers (1024 → 1024)
- Best performing model

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

