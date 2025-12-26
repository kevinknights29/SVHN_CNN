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
    Create and save enhanced training history plots with multi-panel layouts.

    Args:
        history: Keras training history object
        model_name: Name of the model (used for file naming)
        output_dir: Directory to save plots (default: "plots")
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    digit_names = ["dig1", "dig2", "dig3", "dig4"]

    # Create comprehensive 4-panel accuracy plot for all digits
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Per-Digit Training Accuracy - {model_name}', fontsize=16, fontweight='bold')

    for i, (digit_name, ax) in enumerate(zip(digit_names, axes.flat), 1):
        train_acc = history.history[f"{digit_name}_accuracy"]
        val_acc = history.history[f"val_{digit_name}_accuracy"]
        epochs = range(1, len(train_acc) + 1)

        ax.plot(epochs, train_acc, 'b-', linewidth=2, label='Train', alpha=0.8)
        ax.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation', alpha=0.8)
        ax.set_ylim([0, 1])
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(f'Digit Position {i}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Add final accuracy as text annotation
        final_train = train_acc[-1]
        final_val = val_acc[-1]
        ax.text(0.02, 0.98, f'Final: Train={final_train:.3f}, Val={final_val:.3f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path / f"all_digits_accuracy_{model_name}.png",
                bbox_inches="tight", dpi=200)
    plt.close()

    # Create comprehensive loss overview (overall + per-digit)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Loss Overview - {model_name}', fontsize=16, fontweight='bold')

    # Overall loss
    ax = axes[0, 0]
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Train', alpha=0.8)
    ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('Overall Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Per-digit losses
    for i, digit_name in enumerate(digit_names, 1):
        ax = axes[0, i] if i <= 2 else axes[1, i-3]
        train_loss = history.history[f"{digit_name}_loss"]
        val_loss = history.history[f"val_{digit_name}_loss"]
        epochs = range(1, len(train_loss) + 1)

        ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Train', alpha=0.8)
        ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation', alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(f'Digit {i} Loss', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # Additional classifier losses
    ax = axes[1, 2]
    train_num = history.history.get("num_loss", [])
    val_num = history.history.get("val_num_loss", [])
    if train_num:
        epochs = range(1, len(train_num) + 1)
        ax.plot(epochs, train_num, 'b-', linewidth=2, label='Train', alpha=0.8)
        ax.plot(epochs, val_num, 'r-', linewidth=2, label='Validation', alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title('Num Digits Loss', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"all_losses_{model_name}.png",
                bbox_inches="tight", dpi=200)
    plt.close()

    # Create auxiliary classifiers accuracy plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Auxiliary Classifiers - {model_name}', fontsize=16, fontweight='bold')

    # Number of digits accuracy
    ax = axes[0]
    train_acc = history.history["num_accuracy"]
    val_acc = history.history["val_num_accuracy"]
    epochs = range(1, len(train_acc) + 1)
    ax.plot(epochs, train_acc, 'b-', linewidth=2, label='Train', alpha=0.8)
    ax.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation', alpha=0.8)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_title('Number of Digits Classifier', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Digit/no-digit classifier accuracy
    ax = axes[1]
    train_acc = history.history["nC_accuracy"]
    val_acc = history.history["val_nC_accuracy"]
    epochs = range(1, len(train_acc) + 1)
    ax.plot(epochs, train_acc, 'b-', linewidth=2, label='Train', alpha=0.8)
    ax.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation', alpha=0.8)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_title('Has-Digits Binary Classifier', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"auxiliary_classifiers_{model_name}.png",
                bbox_inches="tight", dpi=200)
    plt.close()

    # Create learning curve comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = range(1, len(history.history["dig1_accuracy"]) + 1)

    for i, digit_name in enumerate(digit_names, 1):
        val_acc = history.history[f"val_{digit_name}_accuracy"]
        ax.plot(epochs, val_acc, linewidth=2, label=f'Digit {i}', alpha=0.8)

    ax.plot(epochs, history.history["val_num_accuracy"],
            linewidth=2, linestyle='--', label='Num Digits', alpha=0.8)
    ax.plot(epochs, history.history["val_nC_accuracy"],
            linewidth=2, linestyle='--', label='Has Digits', alpha=0.8)

    ax.set_ylim([0, 1])
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title(f'Validation Accuracy Comparison Across All Outputs - {model_name}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"validation_comparison_{model_name}.png",
                bbox_inches="tight", dpi=200)
    plt.close()


def generate_training_report(
    history,
    metrics: Dict,
    model_name: str,
    output_dir: str = "plots",
) -> None:
    """
    Generate a comprehensive markdown training report with analysis and embedded plots.

    Args:
        history: Keras training history object
        metrics: Dictionary containing evaluation metrics
        model_name: Name of the model
        output_dir: Directory where plots and report will be saved
    """
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Analyze training progression
    digit_names = ["dig1", "dig2", "dig3", "dig4"]
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    min_val_loss = min(history.history["val_loss"])
    min_val_loss_epoch = history.history["val_loss"].index(min_val_loss) + 1

    # Detect overfitting
    overfitting_gap = final_train_loss - final_val_loss
    is_overfitting = overfitting_gap < -0.1  # Val loss significantly higher than train

    # Check convergence
    last_5_val_losses = history.history["val_loss"][-5:]
    loss_variance = np.var(last_5_val_losses) if len(last_5_val_losses) >= 5 else float('inf')
    is_converged = loss_variance < 0.001

    # Per-digit performance analysis
    digit_performance = []
    for i, digit_name in enumerate(digit_names, 1):
        final_val_acc = history.history[f"val_{digit_name}_accuracy"][-1]
        final_train_acc = history.history[f"{digit_name}_accuracy"][-1]
        digit_performance.append({
            'position': i,
            'val_acc': final_val_acc,
            'train_acc': final_train_acc,
            'gap': final_train_acc - final_val_acc
        })

    # Sort by validation accuracy to identify struggling digits
    digit_performance_sorted = sorted(digit_performance, key=lambda x: x['val_acc'])

    # Generate markdown report
    report_lines = []
    report_lines.append(f"# Training Report: {model_name}")
    report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---\n")

    # Executive Summary
    report_lines.append("## Executive Summary\n")
    report_lines.append(f"- **Total Epochs:** {len(history.history['loss'])}")
    report_lines.append(f"- **Best Validation Loss:** {min_val_loss:.4f} (Epoch {min_val_loss_epoch})")
    report_lines.append(f"- **Final Training Loss:** {final_train_loss:.4f}")
    report_lines.append(f"- **Final Validation Loss:** {final_val_loss:.4f}")
    report_lines.append(f"- **Test Sequence Accuracy:** {metrics['testSeqAcc']:.2f}%")
    report_lines.append(f"- **Validation Sequence Accuracy:** {metrics['valSeqAcc']:.2f}%")
    report_lines.append(f"- **Training Sequence Accuracy:** {metrics['trainSeqAcc']:.2f}%\n")

    # Training Status
    report_lines.append("## Training Status\n")
    if is_converged:
        report_lines.append("✅ **Converged:** Model has converged (low loss variance in final epochs)\n")
    else:
        report_lines.append("⚠️ **Not Fully Converged:** Model may benefit from additional training\n")

    if is_overfitting:
        report_lines.append("⚠️ **Overfitting Detected:** Validation loss significantly higher than training loss")
        report_lines.append("   - Consider adding regularization, dropout, or early stopping\n")
    else:
        report_lines.append("✅ **No Significant Overfitting**\n")

    # Performance Metrics Table
    report_lines.append("## Detailed Performance Metrics\n")
    report_lines.append("### Sequence-Level Accuracy\n")
    report_lines.append("| Dataset | Sequence Accuracy |")
    report_lines.append("|---------|-------------------|")
    report_lines.append(f"| Training | {metrics['trainSeqAcc']:.2f}% |")
    report_lines.append(f"| Validation | {metrics['valSeqAcc']:.2f}% |")
    report_lines.append(f"| Test | {metrics['testSeqAcc']:.2f}% |\n")

    report_lines.append("### Per-Component Accuracy (Test Set)\n")
    report_lines.append("| Component | Accuracy |")
    report_lines.append("|-----------|----------|")
    for i, acc in enumerate(metrics['testAcc']):
        if i == 0:
            label = "Number of Digits"
        elif i <= 4:
            label = f"Digit Position {i}"
        else:
            label = "Has Digits (Binary)"
        report_lines.append(f"| {label} | {acc:.2f}% |")
    report_lines.append("")

    # Per-Digit Analysis
    report_lines.append("## Per-Digit Position Analysis\n")
    report_lines.append("### Performance Ranking (by validation accuracy)\n")
    for i, perf in enumerate(digit_performance_sorted):
        rank_emoji = "🥇" if i == 3 else "🥈" if i == 2 else "🥉" if i == 1 else "📊"
        report_lines.append(f"{rank_emoji} **Digit Position {perf['position']}**")
        report_lines.append(f"   - Validation Accuracy: {perf['val_acc']*100:.2f}%")
        report_lines.append(f"   - Training Accuracy: {perf['train_acc']*100:.2f}%")
        report_lines.append(f"   - Train-Val Gap: {perf['gap']*100:.2f}%\n")

    # Identify struggles
    worst_digit = digit_performance_sorted[0]
    report_lines.append("### Areas of Difficulty\n")
    if worst_digit['val_acc'] < 0.85:
        report_lines.append(f"⚠️ **Digit Position {worst_digit['position']}** shows the lowest accuracy ({worst_digit['val_acc']*100:.2f}%)")
        report_lines.append("   - This position may need additional attention or data augmentation\n")
    else:
        report_lines.append("✅ All digit positions performing well (>85% accuracy)\n")

    # Training Progression
    report_lines.append("## Training Progression\n")
    first_epoch_val_loss = history.history["val_loss"][0]
    improvement = ((first_epoch_val_loss - min_val_loss) / first_epoch_val_loss) * 100
    report_lines.append(f"- **Initial Validation Loss:** {first_epoch_val_loss:.4f}")
    report_lines.append(f"- **Best Validation Loss:** {min_val_loss:.4f}")
    report_lines.append(f"- **Total Improvement:** {improvement:.1f}%\n")

    # Check for early plateau
    if min_val_loss_epoch < len(history.history['loss']) * 0.5:
        report_lines.append("⚠️ **Early Plateau:** Best validation loss achieved in first half of training")
        report_lines.append("   - Model may have converged early or learning rate may be too high\n")

    # Visualizations
    report_lines.append("## Visualizations\n")
    report_lines.append("### Per-Digit Accuracy\n")
    report_lines.append(f"![Per-Digit Accuracy](all_digits_accuracy_{model_name}.png)\n")
    report_lines.append("### Loss Overview\n")
    report_lines.append(f"![Loss Overview](all_losses_{model_name}.png)\n")
    report_lines.append("### Auxiliary Classifiers\n")
    report_lines.append(f"![Auxiliary Classifiers](auxiliary_classifiers_{model_name}.png)\n")
    report_lines.append("### Validation Accuracy Comparison\n")
    report_lines.append(f"![Validation Comparison](validation_comparison_{model_name}.png)\n")

    # Recommendations
    report_lines.append("## Recommendations\n")
    if is_overfitting:
        report_lines.append("1. **Address Overfitting:**")
        report_lines.append("   - Increase dropout rates")
        report_lines.append("   - Add L2 regularization")
        report_lines.append("   - Use data augmentation")
        report_lines.append("   - Reduce model complexity\n")

    if not is_converged:
        report_lines.append("2. **Improve Convergence:**")
        report_lines.append("   - Train for more epochs")
        report_lines.append("   - Adjust learning rate schedule")
        report_lines.append("   - Use different optimizer settings\n")

    if worst_digit['val_acc'] < 0.85:
        report_lines.append(f"3. **Improve Digit Position {worst_digit['position']}:**")
        report_lines.append("   - Analyze misclassified examples")
        report_lines.append("   - Consider position-specific data augmentation")
        report_lines.append("   - Review label quality for this position\n")

    if metrics['testSeqAcc'] < metrics['valSeqAcc'] - 5:
        report_lines.append("4. **Test-Validation Gap:**")
        report_lines.append("   - Test accuracy significantly lower than validation")
        report_lines.append("   - Possible data distribution mismatch")
        report_lines.append("   - Review test set characteristics\n")

    # Training Configuration Summary
    report_lines.append("## Training History Summary\n")
    report_lines.append("```")
    report_lines.append(f"Total epochs: {len(history.history['loss'])}")
    report_lines.append(f"Best epoch: {min_val_loss_epoch}")
    report_lines.append(f"Loss improvement: {improvement:.1f}%")
    report_lines.append(f"Converged: {is_converged}")
    report_lines.append(f"Overfitting detected: {is_overfitting}")
    report_lines.append("```\n")

    # Save report
    report_path = output_path / f"training_report_{model_name}.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n{'='*70}")
    print(f"Training report generated: {report_path}")
    print(f"{'='*70}\n")


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

    # Generate comprehensive training report
    generate_training_report(history, metrics, model_name, output_dir="plots")

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
