"""
Refactored training pipeline for SVHN digit sequence detection.

This module provides a clean training pipeline that leverages code from the src directory
and removes unused functionality from the original final_runv3.py.

Includes three CNN architectures:
- Custom designed CNN
- VGG-16 from scratch
- VGG-16 with pre-trained ImageNet weights
"""

import pickle
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

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


def get_lr_metric(optimizer):
    """Create a metric function to track learning rate during training."""

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


def measure_prediction(predictions, labels):
    """
    Measure per-digit accuracy and sequence accuracy.

    Args:
        predictions: List of prediction arrays for each output head
        labels: List of label arrays for each output head

    Returns:
        tuple: (per_digit_accuracies, predicted_outputs, sequence_accuracy)
    """
    labs = np.asarray(labels).squeeze()
    num_features, num_samples = labs.shape

    per_digit_acc = []
    predicted_outputs = []

    for i in range(num_features):
        predicted_values = np.argmax(predictions[i], axis=1).astype("uint8")
        accuracy = (
            np.count_nonzero(predicted_values == labs[i].flatten()) / num_samples * 100
        )
        per_digit_acc.append(accuracy)
        predicted_outputs.append(predicted_values)

    # Calculate sequence accuracy (all 4 digits must match)
    predicted_array = np.asarray(predicted_outputs).T
    sequence_acc = (
        np.count_nonzero(
            np.all(predicted_array[:, 1:5] == labs[1:5, :].T, axis=1)
        )
        / np.float64(num_samples)
        * 100
    )

    return per_digit_acc, predicted_outputs, sequence_acc


def create_and_save_metrics(history, model_name, data, model):
    """
    Create performance plots and save metrics for trained model.

    Args:
        history: Keras training history object
        model_name: Name of the model (used for file naming)
        data: Dictionary containing train/val/test data and labels
        model: Trained Keras model
    """
    train_x = data["trainX"]
    test_x = data["testX"]
    val_x = data["valdX"]
    train_y = data["trainY"]
    test_y = data["testY"]
    val_y = data["valdY"]

    # Create output directories if they don't exist
    Path("plots").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    # Plot digit accuracies
    digit_names = ["dig1", "dig2", "dig3", "dig4"]
    for i, digit_name in enumerate(digit_names, 1):
        plt.figure()
        plt.ylim([0, 1])
        plt.plot(history.history[f"{digit_name}_acc"])
        plt.plot(history.history[f"val_{digit_name}_acc"])
        plt.title(f"Digit{i} accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper left")
        plt.savefig(
            f"plots/modelDig{i}Accuracy_{model_name}.png", bbox_inches="tight", dpi=200
        )
        plt.close()

    # Plot number of digits accuracy
    plt.figure()
    plt.ylim([0, 1])
    plt.plot(history.history["num_acc"])
    plt.plot(history.history["val_num_acc"])
    plt.title("Number of digits accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(
        f"plots/modelNumDigitsAccuracy_{model_name}.png", bbox_inches="tight", dpi=200
    )
    plt.close()

    # Plot digit classifier accuracy
    plt.figure()
    plt.ylim([0, 1])
    plt.plot(history.history["nC_acc"])
    plt.plot(history.history["val_nC_acc"])
    plt.title("Digit classifier accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(
        f"plots/modelDigitClassifierAccuracy_{model_name}.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()

    # Plot overall loss
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(f"plots/modelLoss_{model_name}.png", bbox_inches="tight", dpi=200)
    plt.close()

    # Plot per-digit losses
    for i, digit_name in enumerate(digit_names, 1):
        plt.figure()
        plt.plot(history.history[f"{digit_name}_loss"])
        plt.plot(history.history[f"val_{digit_name}_loss"])
        plt.title(f"Digit{i} loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.savefig(
            f"plots/digit{i}Loss_{model_name}.png", bbox_inches="tight", dpi=200
        )
        plt.close()

    # Evaluate on all datasets
    print("\n" + "=" * 50)
    print("TRAINING SET EVALUATION")
    print("=" * 50)
    y_pred_train = model.predict(train_x)
    score_train = model.evaluate(train_x, train_y, verbose=0)
    train_acc, _, seq_train_acc = measure_prediction(y_pred_train, train_y)
    print(f"Train loss: {score_train[0]:.4f}")
    print("Per-digit accuracy (numdigits, digit1, digit2, digit3, digit4):")
    print(f"  {train_acc}")
    print(f"Train sequence accuracy: {seq_train_acc:.2f}%")

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    y_pred_test = model.predict(test_x)
    score_test = model.evaluate(test_x, test_y, verbose=0)
    test_acc, _, seq_test_acc = measure_prediction(y_pred_test, test_y)
    print(f"Test loss: {score_test[0]:.4f}")
    print("Per-digit accuracy (numdigits, digit1, digit2, digit3, digit4):")
    print(f"  {test_acc}")
    print(f"Test sequence accuracy: {seq_test_acc:.2f}%")

    print("\n" + "=" * 50)
    print("VALIDATION SET EVALUATION")
    print("=" * 50)
    y_pred_val = model.predict(val_x)
    score_val = model.evaluate(val_x, val_y, verbose=0)
    val_acc, _, seq_val_acc = measure_prediction(y_pred_val, val_y)
    print(f"Validation loss: {score_val[0]:.4f}")
    print("Per-digit accuracy (numdigits, digit1, digit2, digit3, digit4):")
    print(f"  {val_acc}")
    print(f"Validation sequence accuracy: {seq_val_acc:.2f}%\n")

    # Save metrics to pickle
    metrics = {
        "trainAcc": train_acc,
        "testAcc": test_acc,
        "valAcc": val_acc,
        "trainSeqAcc": seq_train_acc,
        "testSeqAcc": seq_test_acc,
        "valSeqAcc": seq_val_acc,
        "trainScore": score_train,
        "testScore": score_test,
        "valScore": score_val,
    }

    with open(f"metrics/{model_name}.pickle", "wb") as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"metrics/{model_name}History.pickle", "wb") as f:
        pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Metrics and plots saved to metrics/{model_name}.pickle and plots/")


def train_custom_cnn():
    """
    Train a custom-designed CNN architecture for SVHN digit sequence detection.

    This model uses a deep architecture with multiple convolutional blocks,
    batch normalization, and dropout for regularization.
    """
    print("\n" + "=" * 70)
    print("TRAINING CUSTOM DESIGNED CNN MODEL")
    print("=" * 70 + "\n")

    # Load and prepare data
    data = prepDataforCNN(numChannel=3, feat_norm=True)
    train_x = data["trainX"]
    val_x = data["valdX"]
    train_y = data["trainY"]
    val_y = data["valdY"]

    _, height, width, channels = train_x.shape
    num_length_classes = 5  # 0-4 digits
    num_digit_classes = 11  # 0-9 + blank (10)
    epochs = 75
    batch_size = 64

    # Build model
    input_layer = keras.Input(shape=(height, width, channels), name="customModel")

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
    num_digits_out = Dense(
        num_length_classes, activation="softmax", name="num"
    )(x)
    digit1_out = Dense(num_digit_classes, activation="softmax", name="dig1")(x)
    digit2_out = Dense(num_digit_classes, activation="softmax", name="dig2")(x)
    digit3_out = Dense(num_digit_classes, activation="softmax", name="dig3")(x)
    digit4_out = Dense(num_digit_classes, activation="softmax", name="dig4")(x)
    has_digits_out = Dense(2, activation="softmax", name="nC")(x)

    outputs = [
        num_digits_out,
        digit1_out,
        digit2_out,
        digit3_out,
        digit4_out,
        has_digits_out,
    ]

    model = keras.Model(inputs=input_layer, outputs=outputs)

    # Compile model
    optimizer = optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True
    )

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    model.summary()

    # Setup callbacks
    Path("saved_models").mkdir(exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="saved_models/designedBGRClassifier.keras",
            monitor="loss",
            save_best_only=True,
            verbose=2,
        ),
        keras.callbacks.EarlyStopping(
            monitor="loss", min_delta=0.000001, patience=5, verbose=1, mode="auto"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            verbose=1,
            patience=2,
            cooldown=1,
            min_lr=0.00001,
        ),
        keras.callbacks.TensorBoard(
            log_dir="logs", write_graph=True, batch_size=batch_size, write_images=True
        ),
    ]

    # Train model
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

    # Save metrics and create plots
    create_and_save_metrics(history, "customDesign", data, model)

    return model, history


def train_vgg16_scratch():
    """
    Train a VGG-16 architecture from scratch (random initialization).

    Uses the standard VGG-16 convolutional base without ImageNet weights,
    with custom fully connected layers for multi-digit prediction.
    """
    print("\n" + "=" * 70)
    print("TRAINING VGG-16 FROM SCRATCH")
    print("=" * 70 + "\n")

    # Load and prepare data
    data = prepDataforCNN(numChannel=3, feat_norm=True)
    train_x = data["trainX"]
    val_x = data["valdX"]
    train_y = data["trainY"]
    val_y = data["valdY"]

    _, height, width, channels = train_x.shape
    num_length_classes = 5
    num_digit_classes = 11
    epochs = 50
    batch_size = 64

    # Build model with VGG-16 base (no pre-training)
    vgg16_base = VGG16(include_top=False, weights=None)
    vgg16_base.summary()

    input_layer = keras.Input(shape=(height, width, channels), name="vgg16Scratch")
    vgg_features = vgg16_base(input_layer)

    # Custom fully connected layers
    x = Flatten()(vgg_features)
    x = Dense(512, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)

    # Multi-output heads
    num_digits_out = Dense(num_length_classes, activation="softmax", name="num")(x)
    digit1_out = Dense(num_digit_classes, activation="softmax", name="dig1")(x)
    digit2_out = Dense(num_digit_classes, activation="softmax", name="dig2")(x)
    digit3_out = Dense(num_digit_classes, activation="softmax", name="dig3")(x)
    digit4_out = Dense(num_digit_classes, activation="softmax", name="dig4")(x)
    has_digits_out = Dense(2, activation="softmax", name="nC")(x)

    outputs = [
        num_digits_out,
        digit1_out,
        digit2_out,
        digit3_out,
        digit4_out,
        has_digits_out,
    ]

    model = keras.Model(inputs=input_layer, outputs=outputs)

    # Compile model
    optimizer = optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True
    )

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    model.summary()

    # Setup callbacks
    Path("saved_models").mkdir(exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="saved_models/vgg16.classifier.keras",
            monitor="loss",
            save_best_only=True,
            verbose=2,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.00000001,
            patience=5,
            verbose=1,
            mode="auto",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.1,
            verbose=1,
            patience=3,
            cooldown=0,
            min_lr=0.000001,
        ),
    ]

    # Train model
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

    # Save metrics and create plots
    create_and_save_metrics(history, "vgg16_Scratch", data, model)

    return model, history


def train_vgg16_pretrained():
    """
    Train a VGG-16 architecture with ImageNet pre-trained weights.

    Uses transfer learning with frozen VGG-16 convolutional base and
    custom fully connected layers for multi-digit prediction.
    """
    print("\n" + "=" * 70)
    print("TRAINING VGG-16 WITH IMAGENET PRE-TRAINED WEIGHTS")
    print("=" * 70 + "\n")

    # Load and prepare data
    data = prepDataforCNN(numChannel=3, feat_norm=True)
    train_x = data["trainX"]
    val_x = data["valdX"]
    train_y = data["trainY"]
    val_y = data["valdY"]

    _, height, width, channels = train_x.shape
    num_length_classes = 5
    num_digit_classes = 11
    epochs = 50
    batch_size = 64

    # Build model with pre-trained VGG-16 base
    pretrained_base = VGG16(include_top=False, weights="imagenet")
    pretrained_base.summary()

    input_layer = keras.Input(
        shape=(height, width, channels), name="inputVGGPreTrain"
    )
    vgg_features = pretrained_base(input_layer)

    # Custom fully connected layers
    x = Flatten(name="flatten")(vgg_features)
    x = Dense(1024, activation="relu", name="FC1_4096")(x)
    x = Dense(1024, activation="relu", name="FC1_512")(x)

    # Multi-output heads
    num_digits_out = Dense(num_length_classes, activation="softmax", name="num")(x)
    digit1_out = Dense(num_digit_classes, activation="softmax", name="dig1")(x)
    digit2_out = Dense(num_digit_classes, activation="softmax", name="dig2")(x)
    digit3_out = Dense(num_digit_classes, activation="softmax", name="dig3")(x)
    digit4_out = Dense(num_digit_classes, activation="softmax", name="dig4")(x)
    has_digits_out = Dense(2, activation="softmax", name="nC")(x)

    outputs = [
        num_digits_out,
        digit1_out,
        digit2_out,
        digit3_out,
        digit4_out,
        has_digits_out,
    ]

    model = keras.Model(inputs=input_layer, outputs=outputs)

    # Compile model
    optimizer = optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True
    )

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    model.summary()

    # Setup callbacks
    Path("saved_models").mkdir(exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="saved_models/VGGPreTrained.classifier.keras",
            monitor="loss",
            save_best_only=True,
            verbose=2,
        ),
        keras.callbacks.EarlyStopping(
            monitor="loss", min_delta=0.000001, patience=5, verbose=1, mode="auto"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.1,
            verbose=1,
            patience=4,
            cooldown=1,
            min_lr=0.0001,
        ),
    ]

    # Train model
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

    # Save metrics and create plots
    create_and_save_metrics(history, "vgg16_PreTrain", data, model)

    return model, history


def train_digit_detector():
    """
    Train a binary classifier to detect presence of digits in image patches.

    This model is used during the sliding window detection phase to quickly
    identify regions that might contain digit sequences.

    Note: This function requires helper.preprocessDigDetector() which is not
    yet refactored into the src directory.
    """
    print("\n" + "=" * 70)
    print("TRAINING DIGIT DETECTOR CNN")
    print("=" * 70 + "\n")

    # Note: This requires refactoring helper.preprocessDigDetector() to src
    try:
        import helper

        x, y = helper.preprocessDigDetector()
    except (ImportError, AttributeError):
        print(
            "ERROR: helper.preprocessDigDetector() not available. "
            "This function needs to be refactored into src directory."
        )
        return None, None

    _, height, width = x.shape
    channels = 1

    train_data = np.reshape(x, (x.shape[0], height, width, channels))
    num_samples = train_data.shape[0]

    epochs = 100
    batch_size = 64

    # Train/validation split
    np.random.seed(25)
    split_idx = int(np.round(0.95 * num_samples))
    indices = np.random.permutation(num_samples - 1)
    train_idx = indices[0:split_idx]
    val_idx = indices[split_idx:num_samples]

    y = y.astype(dtype="int8")
    x_train = train_data[train_idx]
    x_val = train_data[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    # Data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=30,
        width_shift_range=0.5,
        height_shift_range=0.5,
        horizontal_flip=True,
        vertical_flip=True,
    )
    datagen.fit(x_train)

    # Build model
    input_layer = keras.Input(shape=(height, width, channels), name="in")

    x = Conv2D(16, (3, 3), activation="relu", padding="same")(input_layer)
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="conv1.5_128")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2_16")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2.5_16")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_32")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2.5_32")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((2, 2), strides=1)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (5, 5), activation="relu", padding="same", name="conv41_256")(x)
    x = Conv2D(128, (5, 5), activation="relu", padding="same", name="conv4_256")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    x = Dropout(0.5)(x)

    x = Conv2D(256, (5, 5), activation="relu", padding="same", name="some256")(x)
    x = Conv2D(256, (5, 5), activation="relu", padding="same", name="some1256")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((2, 2), strides=1)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(1024, activation="relu", name="FC1_1024")(x)
    x = Dense(512, activation="relu", name="FC2_1024")(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation="sigmoid", name="num")(x)

    model = keras.Model(inputs=input_layer, outputs=output)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.summary()

    # Setup callbacks
    Path("saved_models").mkdir(exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="saved_models/weights.DigitsClassifier.keras",
            monitor="loss",
            save_best_only=True,
            verbose=2,
        ),
    ]

    # Train with data augmentation
    history = model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) / batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )

    print(f"\nTraining history keys: {history.history.keys()}")

    return model, history


if __name__ == "__main__":
    # Train the custom designed CNN (default)
    # Uncomment the model you want to train:

    model, history = train_custom_cnn()

    # To train VGG-16 from scratch:
    # model, history = train_vgg16_scratch()

    # To train pre-trained VGG-16:
    # model, history = train_vgg16_pretrained()

    # To train digit detector:
    # model, history = train_digit_detector()
