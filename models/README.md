# SVHN CNN Models

This directory contains modular training scripts for different CNN architectures used for SVHN digit sequence detection.

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

# Or via main script
python train.py --model custom
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

# Or via main script
python train.py --model vgg16_scratch
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

# Or via main script
python train.py --model vgg16_pretrained
```

**Default hyperparameters:**
- Epochs: 50
- Batch size: 64
- Learning rate: 0.001 (Adam)

## Shared Utilities (`training_utils.py`)

Common functions used across all training scripts:

- `get_lr_metric()`: Track learning rate during training
- `measure_prediction()`: Calculate per-digit and sequence accuracy
- `save_training_plots()`: Generate and save training history plots
- `evaluate_and_save_metrics()`: Evaluate on all datasets and save metrics
- `create_multi_output_heads()`: Create standard 6-output architecture
- `get_standard_callbacks()`: Configure callbacks (checkpointing, early stopping, etc.)

## Main Training Script

The `train.py` script in the project root provides a unified interface:

```bash
# Train specific model
python train.py --model vgg16_pretrained

# Train with custom hyperparameters
python train.py --model custom --epochs 100 --batch-size 128 --lr 0.0005

# Train all models sequentially
python train.py --all

# View help
python train.py --help
```

## Output Structure

After training, outputs are organized as:

```
project_root/
├── saved_models/
│   ├── designedBGRClassifier.hdf5      # Custom CNN
│   ├── vgg16.classifier.hdf5            # VGG-16 scratch
│   └── VGGPreTrained.classifier.hdf5    # VGG-16 pretrained
├── plots/
│   ├── modelDig1Accuracy_*.png          # Per-digit accuracy plots
│   ├── modelLoss_*.png                  # Training loss plots
│   └── ...
├── metrics/
│   ├── customDesign.pickle              # Model metrics
│   ├── customDesignHistory.pickle       # Training history
│   └── ...
└── logs/                                # TensorBoard logs
```

## Adding New Models

To add a new model architecture:

1. Create a new file in `models/` (e.g., `train_resnet.py`)
2. Import shared utilities from `training_utils.py`
3. Implement `build_model()` and `train()` functions
4. Use `create_multi_output_heads()` for consistent output structure
5. Add the model to `train.py` orchestrator

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

## Notes

- All models use `sparse_categorical_crossentropy` loss
- Feature normalization is applied by default (`feat_norm=True`)
- Models are saved only when performance improves (via `ModelCheckpoint`)
- Early stopping prevents overfitting
- Learning rate reduction on plateau improves convergence
