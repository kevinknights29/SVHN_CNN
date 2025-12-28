"""
PyTorch training utilities for SVHN digit detection models.

This module provides PyTorch equivalents of the Keras training utilities,
including training loops, callbacks, metrics, and plotting functions.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# Make TensorBoard optional to avoid TensorFlow dependency
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except (ImportError, AttributeError):
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class MultiOutputLoss(nn.Module):
    """
    Combined loss for multi-output model.

    Computes cross-entropy loss for each of the 6 outputs and returns
    the weighted sum.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Optional dictionary of loss weights for each output head.
                     If None, all outputs are weighted equally.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        # Default equal weights
        self.weights = weights or {
            'num': 1.0,
            'dig1': 1.0,
            'dig2': 1.0,
            'dig3': 1.0,
            'dig4': 1.0,
            'nC': 1.0,
        }

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-output loss.

        Args:
            outputs: Dictionary of model outputs
            targets: Dictionary of target labels

        Returns:
            Tuple of (total_loss, individual_losses_dict)
        """
        losses = {}
        total_loss = 0.0

        for key in outputs.keys():
            loss = self.criterion(outputs[key], targets[key])
            losses[key] = loss
            total_loss += self.weights[key] * loss

        return total_loss, losses


def measure_prediction(
    outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
) -> Tuple[List[float], List[np.ndarray], float]:
    """
    Measure per-digit accuracy and sequence accuracy.

    Args:
        outputs: Dictionary of model outputs (logits)
        targets: Dictionary of target labels

    Returns:
        Tuple containing:
            - per_digit_accuracies: List of accuracy percentages [num, dig1-4, nC]
            - predicted_outputs: List of predicted values for each position
            - sequence_accuracy: Percentage of fully correct sequences
    """
    output_keys = ['num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC']
    per_digit_acc = []
    predicted_outputs = []

    # Get predictions
    predictions = {key: torch.argmax(outputs[key], dim=1).cpu().numpy() for key in output_keys}
    labels = {key: targets[key].cpu().numpy() for key in output_keys}

    num_samples = len(labels['num'])

    # Calculate per-digit accuracies
    for key in output_keys:
        correct = np.sum(predictions[key] == labels[key])
        accuracy = (correct / num_samples) * 100
        per_digit_acc.append(accuracy)
        predicted_outputs.append(predictions[key])

    # Calculate sequence accuracy (all 4 digits must match)
    digit_keys = ['dig1', 'dig2', 'dig3', 'dig4']
    predicted_array = np.stack([predictions[key] for key in digit_keys], axis=1)
    labels_array = np.stack([labels[key] for key in digit_keys], axis=1)

    all_correct = np.all(predicted_array == labels_array, axis=1)
    sequence_acc = (np.sum(all_correct) / num_samples) * 100

    return per_digit_acc, predicted_outputs, sequence_acc


class EarlyStopping:
    """
    Early stopping callback to stop training when monitored metric stops improving.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-6, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs with no improvement after which training stops
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for minimizing metric, 'max' for maximizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            current_value: Current value of monitored metric

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class ModelCheckpoint:
    """
    Save model checkpoints based on monitored metric.
    """

    def __init__(self, filepath: str, mode: str = 'min', verbose: bool = True):
        """
        Args:
            filepath: Path to save model checkpoint
            mode: 'min' for minimizing metric, 'max' for maximizing
            verbose: Whether to print save messages
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.verbose = verbose
        self.best_value = None

    def __call__(self, model: nn.Module, current_value: float, epoch: int):
        """
        Save model if current value is better than best value.

        Args:
            model: Model to save
            current_value: Current value of monitored metric
            epoch: Current epoch number
        """
        if self.best_value is None:
            save = True
        elif self.mode == 'min':
            save = current_value < self.best_value
        else:
            save = current_value > self.best_value

        if save:
            self.best_value = current_value
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metric_value': current_value,
            }, self.filepath)

            if self.verbose:
                print(f"\nEpoch {epoch+1}: Saving model to {self.filepath} (metric: {current_value:.6f})")


def create_data_loaders(
    data: Dict,
    batch_size: int = 64,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from numpy arrays.

    Args:
        data: Dictionary containing trainX, valdX, testX and corresponding Y data
        batch_size: Batch size for training
        shuffle_train: Whether to shuffle training data
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    def dict_collate_fn(batch):
        """Custom collate function to create dict of tensors."""
        images = torch.stack([item[0] for item in batch])

        # Stack targets into a dictionary
        targets = {
            'num': torch.stack([item[1]['num'] for item in batch]),
            'dig1': torch.stack([item[1]['dig1'] for item in batch]),
            'dig2': torch.stack([item[1]['dig2'] for item in batch]),
            'dig3': torch.stack([item[1]['dig3'] for item in batch]),
            'dig4': torch.stack([item[1]['dig4'] for item in batch]),
            'nC': torch.stack([item[1]['nC'] for item in batch]),
        }

        return images, targets

    class MultiOutputDataset(torch.utils.data.Dataset):
        """Dataset that returns images and dict of targets."""

        def __init__(self, images, labels):
            # Convert NHWC to NCHW and normalize to [0, 1] if needed
            self.images = torch.FloatTensor(images).permute(0, 3, 1, 2)

            # Labels come from prepDataforCNN as a list of 6 arrays: [num, dig1, dig2, dig3, dig4, nC]
            # Each array has shape (N, 1), so we need to squeeze and convert to tensors
            if isinstance(labels, list):
                # Handle list format from prepDataforCNN
                self.labels = {
                    'num': torch.LongTensor(labels[0].squeeze()),
                    'dig1': torch.LongTensor(labels[1].squeeze()),
                    'dig2': torch.LongTensor(labels[2].squeeze()),
                    'dig3': torch.LongTensor(labels[3].squeeze()),
                    'dig4': torch.LongTensor(labels[4].squeeze()),
                    'nC': torch.LongTensor(labels[5].squeeze()),
                }
            else:
                # Handle 2D array format [num, dig1, dig2, dig3, dig4, nC]
                self.labels = {
                    'num': torch.LongTensor(labels[:, 0]),
                    'dig1': torch.LongTensor(labels[:, 1]),
                    'dig2': torch.LongTensor(labels[:, 2]),
                    'dig3': torch.LongTensor(labels[:, 3]),
                    'dig4': torch.LongTensor(labels[:, 4]),
                    'nC': torch.LongTensor(labels[:, 5]),
                }

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], {key: val[idx] for key, val in self.labels.items()}

    # Create datasets
    train_dataset = MultiOutputDataset(data['trainX'], data['trainY'])
    val_dataset = MultiOutputDataset(data['valdX'], data['valdY'])
    test_dataset = MultiOutputDataset(data['testX'], data['testY'])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=dict_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dict_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dict_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: MultiOutputLoss,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Tuple of (avg_loss, per_head_losses, per_head_accuracies)
    """
    model.train()
    total_loss = 0.0
    head_losses = {key: 0.0 for key in ['num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC']}
    head_correct = {key: 0 for key in ['num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC']}
    total_samples = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = {key: val.to(device) for key, val in targets.items()}

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Compute loss
        loss, losses = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        for key in head_losses.keys():
            head_losses[key] += losses[key].item()

        # Calculate accuracies
        batch_size = images.size(0)
        total_samples += batch_size
        for key in head_correct.keys():
            preds = torch.argmax(outputs[key], dim=1)
            head_correct[key] += (preds == targets[key]).sum().item()

    # Calculate averages
    avg_loss = total_loss / len(train_loader)
    avg_head_losses = {key: val / len(train_loader) for key, val in head_losses.items()}
    avg_head_accuracies = {key: (val / total_samples) * 100 for key, val in head_correct.items()}

    return avg_loss, avg_head_losses, avg_head_accuracies


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: MultiOutputLoss,
    device: torch.device,
) -> Tuple[float, Dict[str, float], Dict[str, float], float]:
    """
    Validate for one epoch.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (avg_loss, per_head_losses, per_head_accuracies, sequence_accuracy)
    """
    model.eval()
    total_loss = 0.0
    head_losses = {key: 0.0 for key in ['num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC']}
    all_outputs = {key: [] for key in ['num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC']}
    all_targets = {key: [] for key in ['num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC']}

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = {key: val.to(device) for key, val in targets.items()}

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss, losses = criterion(outputs, targets)

            # Track metrics
            total_loss += loss.item()
            for key in head_losses.keys():
                head_losses[key] += losses[key].item()

            # Collect outputs and targets
            for key in all_outputs.keys():
                all_outputs[key].append(outputs[key])
                all_targets[key].append(targets[key])

    # Concatenate all batches
    all_outputs = {key: torch.cat(val) for key, val in all_outputs.items()}
    all_targets = {key: torch.cat(val) for key, val in all_targets.items()}

    # Calculate metrics
    per_digit_acc, _, sequence_acc = measure_prediction(all_outputs, all_targets)

    avg_loss = total_loss / len(val_loader)
    avg_head_losses = {key: val / len(val_loader) for key, val in head_losses.items()}
    avg_head_accuracies = {
        'num': per_digit_acc[0],
        'dig1': per_digit_acc[1],
        'dig2': per_digit_acc[2],
        'dig3': per_digit_acc[3],
        'dig4': per_digit_acc[4],
        'nC': per_digit_acc[5],
    }

    return avg_loss, avg_head_losses, avg_head_accuracies, sequence_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: MultiOutputLoss,
    optimizer: Optimizer,
    scheduler: Optional[ReduceLROnPlateau],
    device: torch.device,
    epochs: int,
    model_path: str,
    early_stopping_patience: int = 5,
    use_tensorboard: bool = True,
) -> Dict:
    """
    Complete training loop with callbacks.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epochs: Number of epochs
        model_path: Path to save best model
        early_stopping_patience: Patience for early stopping
        use_tensorboard: Whether to use TensorBoard logging

    Returns:
        Dictionary containing training history
    """
    # Initialize callbacks
    checkpoint = ModelCheckpoint(model_path, mode='min', verbose=True)
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')

    # Initialize TensorBoard
    writer = None
    if use_tensorboard:
        if not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard not available. Logging disabled.")
            print("To enable TensorBoard, ensure torch.utils.tensorboard is properly installed.\n")
        else:
            log_dir = Path("logs") / Path(model_path).stem
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir)

    # Training history
    history = {
        'loss': [],
        'val_loss': [],
        'lr': [],
    }

    # Add per-head metrics to history
    for key in ['num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC']:
        history[f'{key}_loss'] = []
        history[f'{key}_accuracy'] = []
        history[f'val_{key}_loss'] = []
        history[f'val_{key}_accuracy'] = []

    history['val_sequence_accuracy'] = []

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")

    for epoch in range(epochs):
        # Train
        train_loss, train_head_losses, train_head_accs = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_head_losses, val_head_accs, val_seq_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        # Update learning rate
        previous_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            # Log learning rate changes
            if current_lr != previous_lr:
                print(f"  Learning rate reduced: {previous_lr:.6f} -> {current_lr:.6f}")
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        for key in ['num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC']:
            history[f'{key}_loss'].append(train_head_losses[key])
            history[f'{key}_accuracy'].append(train_head_accs[key] / 100)  # Normalize to [0, 1]
            history[f'val_{key}_loss'].append(val_head_losses[key])
            history[f'val_{key}_accuracy'].append(val_head_accs[key] / 100)  # Normalize to [0, 1]

        history['val_sequence_accuracy'].append(val_seq_acc)

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            writer.add_scalar('Sequence_Accuracy/val', val_seq_acc, epoch)

            for key in ['num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC']:
                writer.add_scalar(f'Accuracy_Train/{key}', train_head_accs[key], epoch)
                writer.add_scalar(f'Accuracy_Val/{key}', val_head_accs[key], epoch)

        # Print progress
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Seq Acc: {val_seq_acc:.2f}% | LR: {current_lr:.6f}")

        # Callbacks
        checkpoint(model, val_loss, epoch)

        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered after epoch {epoch+1}")
            break

    if writer is not None:
        writer.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70 + "\n")

    return history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: MultiOutputLoss,
    device: torch.device,
    dataset_name: str = "Test",
) -> Tuple[float, List[float], float]:
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device to evaluate on
        dataset_name: Name of dataset for printing

    Returns:
        Tuple of (loss, per_digit_accuracies, sequence_accuracy)
    """
    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} SET EVALUATION")
    print("=" * 70)

    val_loss, _, val_head_accs, val_seq_acc = validate_epoch(
        model, data_loader, criterion, device
    )

    per_digit_acc = [
        val_head_accs['num'],
        val_head_accs['dig1'],
        val_head_accs['dig2'],
        val_head_accs['dig3'],
        val_head_accs['dig4'],
        val_head_accs['nC'],
    ]

    print(f"{dataset_name} loss: {val_loss:.4f}")
    print("Per-digit accuracy (numdigits, digit1-4, has_digits):")
    print(f"  {[f'{acc:.2f}%' for acc in per_digit_acc]}")
    print(f"{dataset_name} sequence accuracy: {val_seq_acc:.2f}%")

    return val_loss, per_digit_acc, val_seq_acc


def save_training_plots(
    history: Dict, model_name: str, output_dir: str = "plots"
) -> None:
    """
    Create and save training history plots (PyTorch version).

    Args:
        history: Training history dictionary
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
        train_acc = history[f"{digit_name}_accuracy"]
        val_acc = history[f"val_{digit_name}_accuracy"]
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
    train_loss = history["loss"]
    val_loss = history["val_loss"]
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
        train_loss = history[f"{digit_name}_loss"]
        val_loss = history[f"val_{digit_name}_loss"]
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
    train_num = history.get("num_loss", [])
    val_num = history.get("val_num_loss", [])
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
    train_acc = history["num_accuracy"]
    val_acc = history["val_num_accuracy"]
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
    train_acc = history["nC_accuracy"]
    val_acc = history["val_nC_accuracy"]
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
    epochs = range(1, len(history["dig1_accuracy"]) + 1)

    for i, digit_name in enumerate(digit_names, 1):
        val_acc = history[f"val_{digit_name}_accuracy"]
        ax.plot(epochs, val_acc, linewidth=2, label=f'Digit {i}', alpha=0.8)

    ax.plot(epochs, history["val_num_accuracy"],
            linewidth=2, linestyle='--', label='Num Digits', alpha=0.8)
    ax.plot(epochs, history["val_nC_accuracy"],
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

    print(f"\nPlots saved to {output_path}/")


def save_metrics(
    train_metrics: Tuple,
    val_metrics: Tuple,
    test_metrics: Tuple,
    history: Dict,
    model_name: str,
    metrics_dir: str = "metrics",
) -> None:
    """
    Save evaluation metrics to pickle files.

    Args:
        train_metrics: Tuple of (loss, accuracies, sequence_acc) for training set
        val_metrics: Tuple of (loss, accuracies, sequence_acc) for validation set
        test_metrics: Tuple of (loss, accuracies, sequence_acc) for test set
        history: Training history dictionary
        model_name: Name of the model
        metrics_dir: Directory to save metrics
    """
    metrics_path = Path(metrics_dir)
    metrics_path.mkdir(exist_ok=True)

    train_loss, train_acc, train_seq_acc = train_metrics
    val_loss, val_acc, val_seq_acc = val_metrics
    test_loss, test_acc, test_seq_acc = test_metrics

    metrics = {
        "trainAcc": train_acc,
        "testAcc": test_acc,
        "valAcc": val_acc,
        "trainSeqAcc": train_seq_acc,
        "testSeqAcc": test_seq_acc,
        "valSeqAcc": val_seq_acc,
        "trainScore": [train_loss],
        "testScore": [test_loss],
        "valScore": [val_loss],
    }

    # Save metrics
    with open(metrics_path / f"{model_name}.pickle", "wb") as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save training history
    with open(metrics_path / f"{model_name}History.pickle", "wb") as f:
        pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nMetrics saved to {metrics_path / model_name}.pickle")
    print(f"Training history saved to {metrics_path / model_name}History.pickle")
