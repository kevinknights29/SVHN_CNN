"""
PyTorch training script for Alternative CNN architecture with Spatial Transformer Network.

This model combines a Spatial Transformer Network (STN) with a custom CNN architecture
for SVHN digit sequence detection. The STN learns to apply spatial transformations to
the input, helping the network focus on relevant features regardless of position or orientation.

Architecture highlights:
- Spatial Transformer Network for spatial invariance
- 4 convolutional blocks with increasing filter sizes (32 -> 256)
- Batch normalization after each conv layer
- Strategic dropout placement (0.25 default)
- 2 fully connected layers (1024 -> 1024)
- 6 output heads for multi-task learning

Usage:
    python models/train_alternative_cnn_pytorch.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.pytorch_models import AlternativeCNN
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
    epochs: int = 75,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    num_channels: int = 3,
    feat_norm: bool = True,
    dropout_rate: float = 0.25,
):
    """
    Train the Alternative CNN model with PyTorch.

    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate for Adam optimizer
        num_channels: Number of input channels (1 for grayscale, 3 for RGB)
        feat_norm: Whether to apply feature normalization
        dropout_rate: Dropout rate for regularization

    Returns:
        Tuple of (trained_model, training_history)
    """
    print("\n" + "=" * 70)
    print("TRAINING ALTERNATIVE CNN MODEL (PyTorch)")
    print("WITH SPATIAL TRANSFORMER NETWORK")
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
    print("Building model...")
    model = AlternativeCNN(input_channels=num_channels, dropout_rate=dropout_rate).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Dropout rate: {dropout_rate}\n")

    # Setup loss and optimizer
    criterion = MultiOutputLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-7,
        amsgrad=True,
    )

    # Setup learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2,
        cooldown=1,
        min_lr=0.00001,
    )

    # Train model
    print(f"Starting training for {epochs} epochs with batch size {batch_size}...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        model_path="saved_models/alternativeCNN_pytorch.pt",
        early_stopping_patience=5,
        use_tensorboard=True,
    )

    # Load best model
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load("saved_models/alternativeCNN_pytorch.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on all datasets
    train_metrics = evaluate_model(model, train_loader, criterion, device, "Train")
    val_metrics = evaluate_model(model, val_loader, criterion, device, "Validation")
    test_metrics = evaluate_model(model, test_loader, criterion, device, "Test")

    # Save plots
    print("\nSaving training plots...")
    save_training_plots(history, "alternativeCNN_pytorch")

    # Save metrics
    print("\nSaving metrics...")
    save_metrics(train_metrics, val_metrics, test_metrics, history, "alternativeCNN_pytorch")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("Model saved to: saved_models/alternativeCNN_pytorch.pt")
    print("Plots saved to: plots/")
    print("Metrics saved to: metrics/")

    return model, history


if __name__ == "__main__":
    model, history = train()
