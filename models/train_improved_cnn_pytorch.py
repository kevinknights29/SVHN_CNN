"""
PyTorch training script for improved CNN architecture.

This model incorporates modern deep learning techniques:
- ResNet-style skip connections for better gradient flow
- Inception-style multi-scale feature extraction (different kernel sizes)
- Squeeze-and-Excitation blocks for channel attention
- Progressive feature extraction with residual blocks
- Strategic dropout and regularization

Architecture improvements over baseline Custom CNN:
- Skip connections prevent vanishing gradients in deeper networks
- Multi-scale convolutions capture features at different scales simultaneously
- SE blocks add channel-wise attention mechanisms
- Better weight initialization (He/Kaiming)
- More sophisticated regularization strategy

Expected performance: Significant improvement over baseline custom CNN
(8.77% test accuracy) with goal of approaching pre-trained VGG16 (78% test accuracy).

Usage:
    python models/train_improved_cnn_pytorch.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.pytorch_models import ImprovedCNN
from models.pytorch_utils import (
    MultiOutputLoss,
    create_data_loaders,
    evaluate_model,
    save_metrics,
    save_training_plots,
    train_model,
)
from svhn_cnn.data.build_dataset import prepDataforCNN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def train(
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    num_channels: int = 3,
    feat_norm: bool = True,
    dropout_rate: float = 0.5,
    weight_decay: float = 1e-4,
):
    """
    Train the improved CNN model with PyTorch.

    Args:
        epochs: Number of training epochs (default: 100, higher for complex architecture)
        batch_size: Training batch size
        learning_rate: Initial learning rate for Adam optimizer
        num_channels: Number of input channels (1 for grayscale, 3 for RGB)
        feat_norm: Whether to apply feature normalization
        dropout_rate: Dropout rate for regularization (default: 0.5)
        weight_decay: L2 regularization strength (default: 1e-4)

    Returns:
        Tuple of (trained_model, training_history)
    """
    print("\n" + "=" * 70)
    print("TRAINING IMPROVED CNN MODEL (PyTorch)")
    print("=" * 70)
    print("\nArchitecture features:")
    print("  - ResNet-style skip connections")
    print("  - Inception-style multi-scale convolutions")
    print("  - Squeeze-and-Excitation channel attention")
    print("  - Progressive feature extraction")
    print("  - Advanced regularization strategies\n")
    print("=" * 70 + "\n")

    # Load and prepare data
    print("Loading and preparing data...")
    data = prepDataforCNN(numChannel=num_channels, feat_norm=feat_norm)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data, batch_size=batch_size, shuffle_train=True
    )

    print(f"Data loaded: {len(data['trainX'])} training samples, {len(data['valdX'])} validation samples")
    print(f"Input shape: {data['trainX'].shape[1:]}\n")

    # Build model
    print("Building improved CNN model...")
    model = ImprovedCNN(input_channels=num_channels, dropout_rate=dropout_rate).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Setup loss and optimizer
    criterion = MultiOutputLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=weight_decay,  # L2 regularization
        amsgrad=True,
    )

    # Setup learning rate scheduler
    # More aggressive for complex model
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # Reduce by half instead of 0.1
        patience=3,  # Wait 3 epochs before reducing
        cooldown=2,
        min_lr=1e-6,
    )

    # Train model
    print(f"Starting training for {epochs} epochs with batch size {batch_size}...")
    print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}, Dropout: {dropout_rate}\n")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        model_path="saved_models/improvedCNN_pytorch.pt",
        early_stopping_patience=7,  # More patience for complex model
        use_tensorboard=True,
    )

    # Load best model
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load("saved_models/improvedCNN_pytorch.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on all datasets
    train_loss, train_per_digit_acc, train_seq_acc = evaluate_model(model, train_loader, criterion, device, "Train")
    val_loss, val_per_digit_acc, val_seq_acc = evaluate_model(model, val_loader, criterion, device, "Validation")
    test_loss, test_per_digit_acc, test_seq_acc = evaluate_model(model, test_loader, criterion, device, "Test")

    # Save plots
    print("\nSaving training plots...")
    save_training_plots(history, "improvedCNN_pytorch")

    # Save metrics
    print("\nSaving metrics...")
    train_metrics = (train_loss, train_per_digit_acc, train_seq_acc)
    val_metrics = (val_loss, val_per_digit_acc, val_seq_acc)
    test_metrics = (test_loss, test_per_digit_acc, test_seq_acc)
    save_metrics(train_metrics, val_metrics, test_metrics, history, "improvedCNN_pytorch")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("Model saved to: saved_models/improvedCNN_pytorch.pt")
    print("Plots saved to: plots/")
    print("Metrics saved to: metrics/")
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"Improved CNN Test Accuracy: {test_seq_acc:.2f}%")
    print("\nModel comparisons:")
    print("  Custom CNN (baseline):    8.77% test accuracy  ❌")
    print("  VGG16 Scratch:           77.25% test accuracy  ✓")
    print("  VGG16 Pretrained:        78.02% test accuracy  ✓")
    print(f"  Improved CNN:          {test_seq_acc:.2f}% test accuracy  {'✅ BEST!' if test_seq_acc > 78.02 else '✓'}")
    print("\nOriginal Keras benchmark:")
    print("  Custom CNN (Keras):      85.40% test accuracy")
    print("  VGG16 Pretrained (Keras): 91.24% test accuracy")
    print("=" * 70)

    return model, history


if __name__ == "__main__":
    # Train with default parameters
    # You can modify these parameters for experimentation:
    # - epochs: Increase for potentially better convergence
    # - learning_rate: Try 0.0005 or 0.002
    # - dropout_rate: Try 0.3-0.6 range
    # - weight_decay: Try 1e-5 to 1e-3 range

    model, history = train(
        epochs=100,
        batch_size=64,
        learning_rate=0.001,
        num_channels=3,
        feat_norm=True,
        dropout_rate=0.5,
        weight_decay=1e-4,
    )
