"""
PyTorch training script for VGG-16 architecture from scratch.

This model uses the standard VGG-16 convolutional base without ImageNet weights,
trained from random initialization with custom fully connected layers for
multi-digit prediction.

Architecture highlights:
- VGG-16 convolutional base (5 blocks, 13 conv layers)
- Random weight initialization
- Custom FC layers (512 -> 512)
- 6 output heads for multi-task learning

Usage:
    python models/train_vgg16_scratch_pytorch.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.pytorch_models import VGG16Scratch
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
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    num_channels: int = 3,
    feat_norm: bool = True,
):
    """
    Train the VGG-16 model from scratch with PyTorch.

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
    print("TRAINING VGG-16 FROM SCRATCH (PyTorch)")
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
    print("Building VGG-16 model from scratch...")
    model = VGG16Scratch(input_channels=num_channels).to(device)

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
        amsgrad=True,
    )

    # Setup learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
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
        model_path="saved_models/vgg16_classifier_pytorch.pt",
        early_stopping_patience=5,
        use_tensorboard=False,
    )

    # Load best model
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load("saved_models/vgg16_classifier_pytorch.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on all datasets
    train_metrics = evaluate_model(model, train_loader, criterion, device, "Train")
    val_metrics = evaluate_model(model, val_loader, criterion, device, "Validation")
    test_metrics = evaluate_model(model, test_loader, criterion, device, "Test")

    # Save plots
    print("\nSaving training plots...")
    save_training_plots(history, "vgg16_Scratch_pytorch")

    # Save metrics
    print("\nSaving metrics...")
    save_metrics(train_metrics, val_metrics, test_metrics, history, "vgg16_Scratch_pytorch")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("Model saved to: saved_models/vgg16_classifier_pytorch.pt")
    print("Plots saved to: plots/")
    print("Metrics saved to: metrics/")

    return model, history


if __name__ == "__main__":
    model, history = train()
