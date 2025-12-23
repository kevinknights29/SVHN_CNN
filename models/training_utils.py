"""
Shared utilities for training SVHN digit detection models.

This module contains common functions used across different model training scripts,
including metrics calculation, plotting, and callback configuration.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt


def get_lr_metric(optimizer):
    """
    Create a metric function to track learning rate during training.

    Args:
        optimizer: Keras optimizer instance

    Returns:
        Function that returns the current learning rate
    """

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


def measure_prediction(
    predictions: List[np.ndarray], labels: List[np.ndarray]
) -> Tuple[List[float], List[np.ndarray], float]:
    """
    Measure per-digit accuracy and sequence accuracy.

    Args:
        predictions: List of prediction arrays for each output head
        labels: List of label arrays for each output head

    Returns:
        Tuple containing:
            - per_digit_accuracies: List of accuracy percentages for each digit position
            - predicted_outputs: List of predicted values for each position
            - sequence_accuracy: Percentage of fully correct sequences
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
        np.count_nonzero(np.all(predicted_array[:, 1:5] == labs[1:5, :].T, axis=1))
        / np.float64(num_samples)
        * 100
    )

    return per_digit_acc, predicted_outputs, sequence_acc


def save_training_plots(
    history, model_name: str, output_dir: str = "plots"
) -> None:
    """
    Create and save training history plots.

    Args:
        history: Keras training history object
        model_name: Name of the model (used for file naming)
        output_dir: Directory to save plots (default: "plots")
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Plot digit accuracies
    digit_names = ["dig1", "dig2", "dig3", "dig4"]
    for i, digit_name in enumerate(digit_names, 1):
        plt.figure()
        plt.ylim([0, 1])
        plt.plot(history.history[f"{digit_name}_acc"])
        plt.plot(history.history[f"val_{digit_name}_acc"])
        plt.title(f"Digit {i} accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["train", "val"], loc="upper left")
        plt.savefig(
            output_path / f"modelDig{i}Accuracy_{model_name}.png",
            bbox_inches="tight",
            dpi=200,
        )
        plt.close()

    # Plot number of digits accuracy
    plt.figure()
    plt.ylim([0, 1])
    plt.plot(history.history["num_acc"])
    plt.plot(history.history["val_num_acc"])
    plt.title("Number of digits accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(
        output_path / f"modelNumDigitsAccuracy_{model_name}.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()

    # Plot digit classifier accuracy
    plt.figure()
    plt.ylim([0, 1])
    plt.plot(history.history["nC_acc"])
    plt.plot(history.history["val_nC_acc"])
    plt.title("Digit classifier accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(
        output_path / f"modelDigitClassifierAccuracy_{model_name}.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()

    # Plot overall loss
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(
        output_path / f"modelLoss_{model_name}.png", bbox_inches="tight", dpi=200
    )
    plt.close()

    # Plot per-digit losses
    for i, digit_name in enumerate(digit_names, 1):
        plt.figure()
        plt.plot(history.history[f"{digit_name}_loss"])
        plt.plot(history.history[f"val_{digit_name}_loss"])
        plt.title(f"Digit {i} loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.savefig(
            output_path / f"digit{i}Loss_{model_name}.png", bbox_inches="tight", dpi=200
        )
        plt.close()


def evaluate_and_save_metrics(
    model, data: Dict, model_name: str, history, metrics_dir: str = "metrics"
) -> Dict:
    """
    Evaluate model on all datasets and save metrics.

    Args:
        model: Trained Keras model
        data: Dictionary containing train/val/test data and labels
        model_name: Name of the model (used for file naming)
        history: Keras training history object
        metrics_dir: Directory to save metrics (default: "metrics")

    Returns:
        Dictionary containing all computed metrics
    """
    metrics_path = Path(metrics_dir)
    metrics_path.mkdir(exist_ok=True)

    train_x = data["trainX"]
    test_x = data["testX"]
    val_x = data["valdX"]
    train_y = data["trainY"]
    test_y = data["testY"]
    val_y = data["valdY"]

    # Evaluate on training set
    print("\n" + "=" * 70)
    print("TRAINING SET EVALUATION")
    print("=" * 70)
    y_pred_train = model.predict(train_x)
    score_train = model.evaluate(train_x, train_y, verbose=0)
    train_acc, _, seq_train_acc = measure_prediction(y_pred_train, train_y)
    print(f"Train loss: {score_train[0]:.4f}")
    print("Per-digit accuracy (numdigits, digit1, digit2, digit3, digit4):")
    print(f"  {[f'{acc:.2f}%' for acc in train_acc]}")
    print(f"Train sequence accuracy: {seq_train_acc:.2f}%")

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    y_pred_test = model.predict(test_x)
    score_test = model.evaluate(test_x, test_y, verbose=0)
    test_acc, _, seq_test_acc = measure_prediction(y_pred_test, test_y)
    print(f"Test loss: {score_test[0]:.4f}")
    print("Per-digit accuracy (numdigits, digit1, digit2, digit3, digit4):")
    print(f"  {[f'{acc:.2f}%' for acc in test_acc]}")
    print(f"Test sequence accuracy: {seq_test_acc:.2f}%")

    # Evaluate on validation set
    print("\n" + "=" * 70)
    print("VALIDATION SET EVALUATION")
    print("=" * 70)
    y_pred_val = model.predict(val_x)
    score_val = model.evaluate(val_x, val_y, verbose=0)
    val_acc, _, seq_val_acc = measure_prediction(y_pred_val, val_y)
    print(f"Validation loss: {score_val[0]:.4f}")
    print("Per-digit accuracy (numdigits, digit1, digit2, digit3, digit4):")
    print(f"  {[f'{acc:.2f}%' for acc in val_acc]}")
    print(f"Validation sequence accuracy: {seq_val_acc:.2f}%\n")

    # Create metrics dictionary
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

    # Save metrics
    with open(metrics_path / f"{model_name}.pickle", "wb") as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save training history
    with open(metrics_path / f"{model_name}History.pickle", "wb") as f:
        pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Metrics saved to {metrics_path / model_name}.pickle")
    print(f"Training history saved to {metrics_path / model_name}History.pickle")

    return metrics


def create_multi_output_heads(features, num_length_classes: int = 5, num_digit_classes: int = 11):
    """
    Create the standard multi-output heads for SVHN digit detection.

    Args:
        features: Input tensor from the feature extraction layers
        num_length_classes: Number of classes for sequence length (default: 5 for 0-4 digits)
        num_digit_classes: Number of classes for each digit (default: 11 for 0-9 + blank)

    Returns:
        List of output tensors [num_digits, digit1, digit2, digit3, digit4, has_digits]
    """
    from keras.layers import Dense

    num_digits_out = Dense(num_length_classes, activation="softmax", name="num")(
        features
    )
    digit1_out = Dense(num_digit_classes, activation="softmax", name="dig1")(features)
    digit2_out = Dense(num_digit_classes, activation="softmax", name="dig2")(features)
    digit3_out = Dense(num_digit_classes, activation="softmax", name="dig3")(features)
    digit4_out = Dense(num_digit_classes, activation="softmax", name="dig4")(features)
    has_digits_out = Dense(2, activation="softmax", name="nC")(features)

    return [
        num_digits_out,
        digit1_out,
        digit2_out,
        digit3_out,
        digit4_out,
        has_digits_out,
    ]


def get_standard_callbacks(
    model_path: str,
    monitor: str = "loss",
    patience: int = 5,
    reduce_lr_patience: int = 2,
    use_tensorboard: bool = True,
):
    """
    Get standard callbacks for model training.

    Args:
        model_path: Path to save the best model
        monitor: Metric to monitor for early stopping (default: "loss")
        patience: Number of epochs with no improvement for early stopping
        reduce_lr_patience: Number of epochs with no improvement before reducing LR
        use_tensorboard: Whether to use TensorBoard callback

    Returns:
        List of Keras callbacks
    """
    import keras

    Path("saved_models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor=monitor, save_best_only=True, verbose=2
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor, min_delta=0.000001, patience=patience, verbose=1, mode="auto"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if "val" in monitor else monitor,
            factor=0.1,
            verbose=1,
            patience=reduce_lr_patience,
            cooldown=1,
            min_lr=0.00001,
        ),
    ]

    if use_tensorboard:
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir="logs", write_graph=True, write_images=True
            )
        )

    return callbacks
