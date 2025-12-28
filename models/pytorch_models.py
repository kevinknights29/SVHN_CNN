"""
PyTorch implementations of SVHN digit detection models.

This module contains PyTorch versions of the three model architectures:
1. Custom designed CNN
2. VGG-16 from scratch
3. VGG-16 with ImageNet pre-trained weights

All models use multi-output prediction with 6 heads:
- Number of digits (0-4)
- Digit 1 (0-9 + blank=10)
- Digit 2 (0-9 + blank=10)
- Digit 3 (0-9 + blank=10)
- Digit 4 (0-9 + blank=10)
- Has digits binary classifier (yes/no)
"""

import torch
import torch.nn as nn
from torchvision import models


class MultiOutputHead(nn.Module):
    """
    Multi-output classification heads for SVHN digit detection.

    Creates 6 output heads:
    - num: Number of digits (5 classes: 0-4)
    - dig1-dig4: Individual digit predictions (11 classes: 0-9 + blank)
    - nC: Binary has-digits classifier (2 classes)
    """

    def __init__(self, in_features: int, num_length_classes: int = 5, num_digit_classes: int = 11):
        """
        Args:
            in_features: Number of input features from the backbone
            num_length_classes: Number of classes for sequence length (default: 5)
            num_digit_classes: Number of classes for each digit (default: 11)
        """
        super().__init__()

        self.num_digits_head = nn.Linear(in_features, num_length_classes)
        self.digit1_head = nn.Linear(in_features, num_digit_classes)
        self.digit2_head = nn.Linear(in_features, num_digit_classes)
        self.digit3_head = nn.Linear(in_features, num_digit_classes)
        self.digit4_head = nn.Linear(in_features, num_digit_classes)
        self.has_digits_head = nn.Linear(in_features, 2)

    def forward(self, x):
        """
        Forward pass through all output heads.

        Returns:
            Dictionary with keys: 'num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC'
        """
        return {
            'num': self.num_digits_head(x),
            'dig1': self.digit1_head(x),
            'dig2': self.digit2_head(x),
            'dig3': self.digit3_head(x),
            'dig4': self.digit4_head(x),
            'nC': self.has_digits_head(x),
        }


class CustomCNN(nn.Module):
    """
    Custom designed CNN for SVHN digit detection.

    Architecture:
    - 8 convolutional blocks with increasing filter sizes (16 -> 512)
    - Batch normalization after each block
    - Strategic dropout placement
    - 3 fully connected layers (2048 -> 1024 -> 1024)
    - 6 output heads for multi-task learning
    """

    def __init__(self, input_channels: int = 3):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super().__init__()

        # Block 1: 16 filters
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 32 filters
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
        )

        # Block 3: 48 filters
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 64 filters
        self.block4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        # Block 5: 128 filters
        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        # Block 6: 256 filters
        self.block6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.5),
        )

        # Block 7: 256 filters
        self.block7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        # Block 8: 512 filters
        self.block8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.25),
        )

        # Calculate the flattened feature size
        # Input: (48, 48, 3)
        # After block1 (MaxPool2d 2x2): 24x24
        # After block2 (MaxPool2d 2x2): 12x12
        # After block3 (MaxPool2d 2x2): 6x6
        # After block4 (MaxPool2d 2x2, stride=1): 5x5
        # After block5 (MaxPool2d 2x2, stride=1): 4x4
        # After block6 (MaxPool2d 2x2, stride=1): 3x3
        # After block7 (MaxPool2d 2x2, stride=1): 2x2
        # After block8 (MaxPool2d 2x2, stride=1): 1x1
        self.flatten_size = 512 * 1 * 1

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )

        # Multi-output heads
        self.output_heads = MultiOutputHead(in_features=1024)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Dictionary with 6 output tensors
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.fc_layers(x)
        return self.output_heads(x)


class VGG16Scratch(nn.Module):
    """
    VGG-16 architecture trained from scratch (no pre-training).

    Uses the standard VGG-16 convolutional base with random initialization,
    followed by custom fully connected layers for multi-digit prediction.
    """

    def __init__(self, input_channels: int = 3):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super().__init__()

        # VGG-16 feature extractor (without pre-trained weights)
        vgg16 = models.vgg16(weights=None)

        # If grayscale input, modify first conv layer
        if input_channels != 3:
            vgg16.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)

        self.features = vgg16.features

        # Adaptive pooling to handle 48x48 input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Multi-output heads
        self.output_heads = MultiOutputHead(in_features=512)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Dictionary with 6 output tensors
        """
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.fc_layers(x)
        return self.output_heads(x)


class VGG16Pretrained(nn.Module):
    """
    VGG-16 with ImageNet pre-trained weights for transfer learning.

    Uses the VGG-16 convolutional base pre-trained on ImageNet,
    with custom fully connected layers for multi-digit prediction.
    This typically achieves the best performance (~91% sequence accuracy).
    """

    def __init__(self, input_channels: int = 3, freeze_backbone: bool = False):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            freeze_backbone: Whether to freeze the pre-trained VGG-16 weights
        """
        super().__init__()

        # VGG-16 with ImageNet pre-trained weights
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # If grayscale input, modify first conv layer and lose pre-trained weights for that layer
        if input_channels != 3:
            original_first_layer = vgg16.features[0]
            vgg16.features[0] = nn.Conv2d(
                input_channels, 64, kernel_size=3, padding=1
            )
            # Initialize from pre-trained weights if possible
            if input_channels == 1:
                # Average RGB weights for grayscale
                with torch.no_grad():
                    vgg16.features[0].weight[:, 0, :, :] = original_first_layer.weight.mean(dim=1)

        self.features = vgg16.features

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Adaptive pooling to handle 48x48 input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )

        # Multi-output heads
        self.output_heads = MultiOutputHead(in_features=1024)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Dictionary with 6 output tensors
        """
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.fc_layers(x)
        return self.output_heads(x)


def get_model(model_name: str, input_channels: int = 3, **kwargs):
    """
    Factory function to get a model by name.

    Args:
        model_name: One of 'custom', 'vgg16_scratch', 'vgg16_pretrained'
        input_channels: Number of input channels
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance

    Raises:
        ValueError: If model_name is not recognized
    """
    models_map = {
        'custom': CustomCNN,
        'vgg16_scratch': VGG16Scratch,
        'vgg16_pretrained': VGG16Pretrained,
    }

    if model_name not in models_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models_map.keys())}")

    return models_map[model_name](input_channels=input_channels, **kwargs)


if __name__ == "__main__":
    # Test model instantiation and forward pass
    print("Testing PyTorch model architectures...\n")

    batch_size = 4
    channels = 3
    height, width = 48, 48
    x = torch.randn(batch_size, channels, height, width)

    for model_name in ['custom', 'vgg16_scratch', 'vgg16_pretrained']:
        print(f"\n{'='*70}")
        print(f"Testing {model_name.upper()}")
        print('='*70)

        model = get_model(model_name, input_channels=channels)
        model.eval()

        with torch.no_grad():
            outputs = model(x)

        print(f"Input shape: {x.shape}")
        print("\nOutput shapes:")
        for key, value in outputs.items():
            print(f"  {key}: {value.shape}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    print("\n" + "="*70)
    print("All models tested successfully!")
    print("="*70)
