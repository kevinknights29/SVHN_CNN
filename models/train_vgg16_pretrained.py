"""
Training script for VGG-16 with ImageNet pre-trained weights.

This model uses transfer learning with the VGG-16 convolutional base
pre-trained on ImageNet, with custom fully connected layers for multi-digit
prediction. This typically achieves the best performance (~91% sequence accuracy).

Architecture highlights:
- VGG-16 convolutional base with ImageNet weights
- Transfer learning approach
- Custom FC layers (1024 -> 1024)
- 6 output heads for multi-task learning

Usage:
    python models/train_vgg16_pretrained.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import keras
import tensorflow as tf
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten

from models.training_utils import (
    create_multi_output_heads,
    evaluate_and_save_metrics,
    get_standard_callbacks,
    save_training_plots,
)
from svhn_cnn.data.build_dataset import prepDataforCNN

# Configure TensorFlow to work with both CPU and GPU
# Enable memory growth for GPUs to avoid allocating all GPU memory at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s) - memory growth enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found - training will use CPU")


def build_model(input_shape=(48, 48, 3)):
    """
    Build VGG-16 model with ImageNet pre-trained weights.

    Args:
        input_shape: Shape of input images (height, width, channels)

    Returns:
        Compiled Keras model
    """
    # VGG-16 base with ImageNet pre-trained weights
    pretrained_base = VGG16(include_top=False, weights="imagenet")

    input_layer = keras.Input(shape=input_shape, name="inputVGGPreTrain")
    vgg_features = pretrained_base(input_layer)

    # Custom fully connected layers
    x = Flatten(name="flatten")(vgg_features)
    x = Dense(1024, activation="relu", name="FC1_4096")(x)
    x = Dense(1024, activation="relu", name="FC1_512")(x)

    # Multi-output heads
    outputs = create_multi_output_heads(x)

    return keras.Model(inputs=input_layer, outputs=outputs)


def train(
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    num_channels: int = 3,
    feat_norm: bool = True,
):
    """
    Train the VGG-16 model with pre-trained weights.

    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate for Adam optimizer
        num_channels: Number of input channels (1 for grayscale, 3 for RGB)
        feat_norm: Whether to apply feature normalization

    Returns:
        Tuple of (trained_model, training_history)
    """
    print("\n" + "=" * 70)
    print("TRAINING VGG-16 WITH IMAGENET PRE-TRAINED WEIGHTS")
    print("=" * 70 + "\n")

    # Load and prepare data
    print("Loading and preparing data...")
    data = prepDataforCNN(numChannel=num_channels, feat_norm=feat_norm)
    train_x = data["trainX"]
    val_x = data["valdX"]
    train_y = data["trainY"]
    val_y = data["valdY"]

    _, height, width, channels = train_x.shape
    print(f"Data loaded: {train_x.shape[0]} training samples, {val_x.shape[0]} validation samples")
    print(f"Input shape: ({height}, {width}, {channels})\n")

    # Build model
    print("Building VGG-16 model with pre-trained ImageNet weights...")
    model = build_model(input_shape=(height, width, channels))

    # Compile model
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=True
    )

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    model.summary()

    # Setup callbacks
    callbacks = get_standard_callbacks(
        model_path="saved_models/VGGPreTrained.classifier.keras",
        monitor="loss",
        patience=5,
        reduce_lr_patience=4,
        use_tensorboard=False,
    )

    # Train model
    print(f"\nStarting training for {epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        shuffle=True,
        validation_data=(val_x, val_y),
        callbacks=callbacks,
    )

    # Save plots
    print("\nSaving training plots...")
    save_training_plots(history, "vgg16_PreTrain")

    # Evaluate and save metrics
    print("\nEvaluating model...")
    evaluate_and_save_metrics(model, data, "vgg16_PreTrain", history)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("Model saved to: saved_models/VGGPreTrained.classifier.keras")
    print("Plots saved to: plots/")
    print("Metrics saved to: metrics/")
    print("\nNote: This model typically achieves ~91% sequence accuracy on test set")

    return model, history


if __name__ == "__main__":
    model, history = train()
