"""
Training script for custom designed CNN architecture.

This model uses a deep architecture with multiple convolutional blocks,
batch normalization, and dropout for regularization. It's specifically
designed for SVHN digit sequence detection with multi-output prediction.

Architecture highlights:
- 8 convolutional blocks with increasing filter sizes (16 -> 512)
- Batch normalization after each block
- Strategic dropout placement
- 3 fully connected layers (2048 -> 1024 -> 1024)
- 6 output heads for multi-task learning

Usage:
    python models/train_custom_cnn.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import keras
import tensorflow as tf
from keras import optimizers
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)

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
    Build the custom CNN architecture.

    Args:
        input_shape: Shape of input images (height, width, channels)

    Returns:
        Compiled Keras model
    """
    input_layer = keras.Input(shape=input_shape, name="customModel")

    # Block 1: 16 filters
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="conv_16_1")(
        input_layer
    )
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="conv_16_2")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2: 32 filters
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2_32_01")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2_32_02")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    # Block 3: 48 filters
    x = Conv2D(48, (3, 3), activation="relu", padding="same", name="conv2_48_01")(x)
    x = Conv2D(48, (3, 3), activation="relu", padding="same", name="conv2_48_02")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 4: 64 filters
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_64_1")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_64_2")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_64_3")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((2, 2), strides=1)(x)

    # Block 5: 128 filters
    x = Conv2D(128, (5, 5), activation="relu", padding="same", name="conv2_128_1")(x)
    x = Conv2D(128, (5, 5), activation="relu", padding="same", name="conv2_128_2")(x)
    x = Conv2D(128, (5, 5), activation="relu", padding="same", name="conv2_128_3")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)

    # Block 6: 256 filters
    x = Conv2D(256, (5, 5), activation="relu", padding="same", name="conv2_128_5")(x)
    x = Conv2D(256, (5, 5), activation="relu", padding="same", name="conv2_128_6")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    x = Dropout(0.5)(x)

    # Block 7: 256 filters
    x = Conv2D(256, (5, 5), activation="relu", padding="same", name="conv256_1")(x)
    x = Conv2D(256, (5, 5), activation="relu", padding="same", name="conv256_2")(x)
    x = Conv2D(256, (5, 5), activation="relu", padding="same", name="conv256_3")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((2, 2), strides=1)(x)

    # Block 8: 512 filters
    x = Conv2D(512, (5, 5), activation="relu", padding="same", name="conv2_512_1")(x)
    x = Conv2D(512, (5, 5), activation="relu", padding="same", name="conv2_512_2")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    x = Dropout(0.25)(x)

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(2048, activation="relu", name="FC1_2048")(x)
    x = Dense(1024, activation="relu", name="FC1_1024")(x)
    x = Dense(1024, activation="relu", name="FC2_1024")(x)

    # Multi-output heads
    outputs = create_multi_output_heads(x)

    return keras.Model(inputs=input_layer, outputs=outputs)


def train(
    epochs: int = 75,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    num_channels: int = 3,
    feat_norm: bool = True,
):
    """
    Train the custom CNN model.

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
    print("TRAINING CUSTOM DESIGNED CNN MODEL")
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
    print("Building model...")
    model = build_model(input_shape=(height, width, channels))

    # Compile model
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=True
    )

    # Provide metrics for each of the 6 output heads
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics={
            "num": ["accuracy"],
            "dig1": ["accuracy"],
            "dig2": ["accuracy"],
            "dig3": ["accuracy"],
            "dig4": ["accuracy"],
            "nC": ["accuracy"],
        },
    )

    model.summary()

    # Setup callbacks
    callbacks = get_standard_callbacks(
        model_path="saved_models/designedBGRClassifier.keras",
        monitor="loss",
        patience=5,
        reduce_lr_patience=2,
        use_tensorboard=True,
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
    save_training_plots(history, "customDesign")

    # Evaluate and save metrics
    print("\nEvaluating model...")
    evaluate_and_save_metrics(model, data, "customDesign", history)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("Model saved to: saved_models/designedBGRClassifier.keras")
    print("Plots saved to: plots/")
    print("Metrics saved to: metrics/")

    return model, history


if __name__ == "__main__":
    model, history = train()
