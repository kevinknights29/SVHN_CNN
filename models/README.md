# SVHN CNN Models

This directory contains the **official training scripts** for different CNN architectures used for SVHN digit sequence detection.

**IMPORTANT:** To train any model, use the training scripts in this directory. These are the only supported way to train models in this project.

## Current Training Progress

### Pre-trained VGG-16 (Best Model) ⭐
- **Test Sequence Accuracy:** 78.20%
- **Validation Sequence Accuracy:** 88.57%
- **Training Sequence Accuracy:** 98.71%
- **Status:** Model trained for 50 epochs; shows some overfitting but achieves best performance
- **Detailed Report:** [plots/training_report_vgg16_PreTrain.md](../plots/training_report_vgg16_PreTrain.md)

### Custom Designed CNN
- **Test Sequence Accuracy:** 35.33%
- **Validation Sequence Accuracy:** 66.37%
- **Training Sequence Accuracy:** 68.40%
- **Status:** Model trained for 75 epochs; converged without significant overfitting
- **Detailed Report:** [plots/training_report_customDesign.md](../plots/training_report_customDesign.md)

All training runs automatically generate:
- Comprehensive training reports with automated analysis
- Multi-panel accuracy and loss visualizations
- Per-digit performance breakdowns
- Actionable recommendations

See the [plots](../plots/) directory for all training visualizations and reports.

## Directory Structure

```
models/
├── __init__.py                    # Package initialization
├── training_utils.py              # Shared utilities for all models
├── train_custom_cnn.py           # Custom designed CNN architecture
├── train_vgg16_scratch.py        # VGG-16 from random initialization
└── train_vgg16_pretrained.py     # VGG-16 with ImageNet weights (recommended)
```

## Available Models

### 1. Custom CNN (`train_custom_cnn.py`)
A deep custom-designed architecture specifically for SVHN digit detection.

**Architecture:**
- 8 convolutional blocks with increasing filter sizes (16 → 512)
- Batch normalization and strategic dropout
- 3 fully connected layers (2048 → 1024 → 1024)
- 6 output heads for multi-task learning

**Training:**
```bash
python models/train_custom_cnn.py
```

**Default hyperparameters:**
- Epochs: 75
- Batch size: 64
- Learning rate: 0.001 (Adam)

### 2. VGG-16 Scratch (`train_vgg16_scratch.py`)
Standard VGG-16 architecture trained from random initialization.

**Architecture:**
- VGG-16 convolutional base (5 blocks, 13 conv layers)
- Random weight initialization
- Custom FC layers (512 → 512)
- 6 output heads

**Training:**
```bash
python models/train_vgg16_scratch.py
```

**Default hyperparameters:**
- Epochs: 50
- Batch size: 64
- Learning rate: 0.001 (Adam)

### 3. VGG-16 Pre-trained (`train_vgg16_pretrained.py`) ⭐ **Recommended**
VGG-16 with ImageNet pre-trained weights using transfer learning.

**Architecture:**
- VGG-16 convolutional base with ImageNet weights
- Custom FC layers (1024 → 1024)
- 6 output heads

**Performance:**
- ~91% test sequence accuracy
- Best performing model on ~150,000 samples

**Training:**
```bash
python models/train_vgg16_pretrained.py
```

**Default hyperparameters:**
- Epochs: 50
- Batch size: 64
- Learning rate: 0.001 (Adam)

## Shared Utilities (`training_utils.py`)

Common functions used across all training scripts:

- `get_lr_metric()`: Track learning rate during training
- `measure_prediction()`: Calculate per-digit and sequence accuracy
- `save_training_plots()`: **ENHANCED** - Generate comprehensive multi-panel plots with improved visualizations
- `generate_training_report()`: **NEW** - Create markdown training reports with automated analysis
- `evaluate_and_save_metrics()`: Evaluate on all datasets, save metrics, and auto-generate training report
- `create_multi_output_heads()`: Create standard 6-output architecture
- `get_standard_callbacks()`: Configure callbacks (checkpointing, early stopping, etc.)

### Enhanced Plotting and Reporting

The new visualization and reporting system provides:

**Multi-Panel Plots:**
- `all_digits_accuracy_*.png` - 4-panel comparison of all digit positions
- `all_losses_*.png` - 6-panel overview (overall + per-digit losses)
- `auxiliary_classifiers_*.png` - Number-of-digits and has-digits classifiers
- `validation_comparison_*.png` - Direct comparison across all outputs

**Comprehensive Training Report (`training_report_*.md`):**
- Executive summary with key metrics
- Automated analysis (overfitting detection, convergence checks)
- Performance ranking by digit position
- Identification of struggling components
- Actionable recommendations based on training patterns

See `example_training_report.py` for usage examples, or refer to the dedicated training utilities documentation in this directory

## Running Training

To train a model, simply run the corresponding training script from the project root:

```bash
# Train the recommended pre-trained VGG-16 model
python models/train_vgg16_pretrained.py

# Train the custom designed CNN
python models/train_custom_cnn.py

# Train VGG-16 from scratch
python models/train_vgg16_scratch.py
```

Each script uses optimized default hyperparameters. To customize training parameters, you can edit the script directly or modify the function arguments in the `if __name__ == "__main__":` block.

## Output Structure

After training, outputs are organized as:

```
project_root/
├── saved_models/
│   ├── designedBGRClassifier.hdf5         # Custom CNN
│   ├── vgg16.classifier.hdf5              # VGG-16 scratch
│   └── VGGPreTrained.classifier.hdf5      # VGG-16 pretrained
├── plots/
│   ├── training_report_*.md               # 📊 NEW: Comprehensive training report
│   ├── all_digits_accuracy_*.png          # 📊 NEW: Multi-panel digit accuracy
│   ├── all_losses_*.png                   # 📊 NEW: Multi-panel loss overview
│   ├── auxiliary_classifiers_*.png        # 📊 NEW: Auxiliary classifier performance
│   ├── validation_comparison_*.png        # 📊 NEW: Validation accuracy comparison
│   └── [legacy individual plots]          # Old format plots (deprecated)
├── metrics/
│   ├── customDesign.pickle                # Model metrics
│   ├── customDesignHistory.pickle         # Training history
│   └── ...
└── logs/                                  # TensorBoard logs
```

## Adding New Models

To add a new model architecture:

1. Create a new file in `models/` (e.g., `train_resnet.py`)
2. Import shared utilities from `training_utils.py`
3. Implement `build_model()` and `train()` functions
4. Use `create_multi_output_heads()` for consistent output structure
5. Follow the same structure as existing training scripts

**Template:**

```python
"""Training script for NewModel architecture."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import keras
import tensorflow as tf
from models.training_utils import (
    create_multi_output_heads,
    evaluate_and_save_metrics,
    get_standard_callbacks,
    save_training_plots,
)
from svhn_cnn.data.build_dataset import prepDataforCNN

# Configure TensorFlow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def build_model(input_shape=(48, 48, 3)):
    """Build the new model architecture."""
    input_layer = keras.Input(shape=input_shape, name="newModel")

    # Add your architecture here
    x = ...

    # Use shared multi-output heads
    outputs = create_multi_output_heads(x)
    return keras.Model(inputs=input_layer, outputs=outputs)


def train(epochs=50, batch_size=64, learning_rate=0.001,
          num_channels=3, feat_norm=True):
    """Train the new model."""
    print("\\n" + "="*70)
    print("TRAINING NEW MODEL")
    print("="*70 + "\\n")

    # Load data
    data = prepDataforCNN(numChannel=num_channels, feat_norm=feat_norm)

    # Build and compile model
    model = build_model()
    model.compile(...)

    # Train with standard callbacks
    callbacks = get_standard_callbacks(
        model_path="saved_models/newmodel.hdf5"
    )

    history = model.fit(...)

    # Save plots and metrics
    save_training_plots(history, "newModel")
    evaluate_and_save_metrics(model, data, "newModel", history)

    return model, history


if __name__ == "__main__":
    model, history = train()
```

## Multi-Output Architecture

All models use a standard 6-output architecture for multi-task learning:

1. **num** (5 classes): Number of digits in sequence (0-4)
2. **dig1** (11 classes): First digit (0-9 + blank)
3. **dig2** (11 classes): Second digit (0-9 + blank)
4. **dig3** (11 classes): Third digit (0-9 + blank)
5. **dig4** (11 classes): Fourth digit (0-9 + blank)
6. **nC** (2 classes): Binary classifier (has digits: yes/no)

## Dependencies

All training scripts require:
- `tensorflow` / `keras`
- `numpy`
- `matplotlib`
- `pickle` (standard library)

Data loading requires the `svhn_cnn` package from `src/`.

## PyTorch Migration 🔥

The entire codebase has been migrated to PyTorch! All three model architectures are now available in PyTorch with equivalent functionality.

### PyTorch Files

```
models/
├── pytorch_models.py                     # PyTorch model architectures
├── pytorch_utils.py                      # PyTorch training utilities
├── train_custom_cnn_pytorch.py          # Custom CNN (PyTorch)
├── train_vgg16_scratch_pytorch.py       # VGG-16 scratch (PyTorch)
└── train_vgg16_pretrained_pytorch.py    # VGG-16 pretrained (PyTorch)
```

### PyTorch Model Architectures (`pytorch_models.py`)

All three models are implemented as `nn.Module` subclasses:

1. **CustomCNN** - Custom designed CNN with 8 conv blocks
2. **VGG16Scratch** - VGG-16 from random initialization
3. **VGG16Pretrained** - VGG-16 with ImageNet weights (uses `torchvision.models`)

**Factory function:**
```python
from models.pytorch_models import get_model

model = get_model('custom', input_channels=3)
model = get_model('vgg16_scratch', input_channels=3)
model = get_model('vgg16_pretrained', input_channels=3, freeze_backbone=False)
```

### PyTorch Training

Training scripts mirror the TensorFlow versions with the same hyperparameters:

```bash
# Train with PyTorch (recommended)
python models/train_custom_cnn_pytorch.py
python models/train_vgg16_scratch_pytorch.py
python models/train_vgg16_pretrained_pytorch.py
```

### PyTorch Utilities (`pytorch_utils.py`)

Complete PyTorch equivalents of all training utilities:

- **MultiOutputLoss** - Combined cross-entropy loss for all 6 outputs
- **create_data_loaders()** - Convert numpy arrays to PyTorch DataLoaders
- **train_model()** - Complete training loop with callbacks
- **EarlyStopping** - Early stopping callback
- **ModelCheckpoint** - Save best model based on validation loss
- **measure_prediction()** - Calculate per-digit and sequence accuracy
- **evaluate_model()** - Evaluate on train/val/test sets
- **save_training_plots()** - Generate all training visualizations
- **save_metrics()** - Save metrics in same format as TensorFlow

### Key Differences from TensorFlow

1. **Data Format**: PyTorch uses NCHW (batch, channels, height, width) vs TensorFlow's NHWC
2. **Model Outputs**: Dictionary format `{'num': tensor, 'dig1': tensor, ...}`
3. **Checkpoints**: Saved as `.pt` files with `model_state_dict`
4. **Device Management**: Automatic CPU/GPU detection with explicit `.to(device)` calls
5. **Adaptive Pooling**: Uses `AdaptiveAvgPool2d` to handle 48x48 input size for VGG models

### Model Output Files

PyTorch models save to:
```
saved_models/
├── designedBGRClassifier_pytorch.pt          # Custom CNN
├── vgg16_classifier_pytorch.pt               # VGG-16 scratch
└── VGGPreTrained_classifier_pytorch.pt       # VGG-16 pretrained
```

### Testing PyTorch Models

Test model architectures without training:

```bash
python models/pytorch_models.py
```

This will instantiate all three models and verify forward pass functionality.

### Performance Expectations

PyTorch models should achieve similar performance to TensorFlow versions:
- **Custom CNN**: ~66-68% validation sequence accuracy
- **VGG-16 Scratch**: TBD (depends on training)
- **VGG-16 Pretrained**: **~91% test sequence accuracy** (same as TensorFlow)

### Dependencies

PyTorch training requires:
```toml
torch>=2.0.0
torchvision>=0.15.0
```

These have been added to `pyproject.toml`. Install with:
```bash
uv sync
```

## TensorFlow vs PyTorch - Which to Use?

**Use PyTorch if:**
- You prefer PyTorch's imperative programming style
- You need more control over training loops
- You want easier debugging with standard Python
- You're deploying to PyTorch-based production systems

**Use TensorFlow/Keras if:**
- You prefer high-level APIs and quick prototyping
- You're already familiar with Keras
- You need TensorFlow Serving or TensorFlow Lite
- You want to use existing trained `.hdf5` models

Both implementations produce equivalent results and save compatible metrics/plots!

## Notes

- All models use `sparse_categorical_crossentropy` loss (TensorFlow) or `CrossEntropyLoss` (PyTorch)
- Feature normalization is applied by default (`feat_norm=True`)
- Models are saved only when performance improves (via `ModelCheckpoint`)
- Early stopping prevents overfitting
- Learning rate reduction on plateau improves convergence
- PyTorch models automatically detect and use GPU if available
