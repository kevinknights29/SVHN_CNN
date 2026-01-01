"""
PyTorch implementations of SVHN digit detection models.

This module contains PyTorch versions of the model architectures:
1. Custom designed CNN (baseline)
2. Improved CNN (best performing custom architecture)
3. Alternative CNN (with Spatial Transformer Network)
4. VGG-16 from scratch
5. VGG-16 with ImageNet pre-trained weights

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
import torch.nn.functional as F
from torchvision import models


class STN(nn.Module):
    """
    Spatial Transformer Network for learning invariance to translation, scale, and rotation.

    The STN learns to apply spatial transformations to input images, helping the network
    focus on relevant features regardless of their position or orientation.

    Reference: Jaderberg et al., "Spatial Transformer Networks" (2015)
    """

    def __init__(self, input_channel: int = 3):
        """
        Args:
            input_channel: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super().__init__()

        # Localization Network - learns where to focus
        # Input: (Batch, input_channel, 48, 48)
        self.localization = nn.Sequential(
            nn.Conv2d(input_channel, 8, kernel_size=7),  # -> 42x42
            nn.MaxPool2d(2, stride=2),                   # -> 21x21
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),             # -> 17x17
            nn.MaxPool2d(2, stride=2),                   # -> 8x8
            nn.ReLU(True)
        )

        # Regressor - calculates 6 parameters (2x3 affine transformation matrix)
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 8 * 8, 32),  # 10 channels * 8 * 8
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)  # 6 parameters output
        )

        # Initialize weights to identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        """
        Apply learned spatial transformation to input.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Spatially transformed tensor of same shape as input
        """
        # Calculate transformation parameters (theta)
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 8 * 8)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Generate sampling grid
        grid = F.affine_grid(theta, x.size(), align_corners=True)

        # Sample from input using grid to get transformed image
        x = F.grid_sample(x, grid, align_corners=True)

        return x


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


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    Adaptively recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between channels.

    Reference: Hu et al., "Squeeze-and-Excitation Networks" (2018)
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for the bottleneck
        """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class InceptionModule(nn.Module):
    """
    Inception-style module with multi-scale feature extraction.

    Uses parallel convolutions with different kernel sizes to capture
    features at multiple scales simultaneously.

    Inspired by: Szegedy et al., "Going Deeper with Convolutions" (2015)
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (divided among branches)
        """
        super().__init__()

        # Ensure output channels is divisible by 4
        branch_channels = out_channels // 4

        # 1x1 convolution branch
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 5x5 convolution branch (using two 3x3 for efficiency)
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch2 = self.branch3x3(x)
        branch3 = self.branch5x5(x)
        branch4 = self.branch_pool(x)

        # Concatenate along channel dimension
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Implements the residual learning framework where the block learns
    a residual mapping F(x) and adds it to the input: H(x) = F(x) + x

    Reference: He et al., "Deep Residual Learning for Image Recognition" (2016)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = True):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolution
            use_se: Whether to use Squeeze-and-Excitation
        """
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE block
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE block
        out = self.se(out)

        # Add skip connection
        out += identity
        out = self.relu(out)

        return out


class ImprovedCNN(nn.Module):
    """
    Improved CNN architecture incorporating modern deep learning techniques.

    This architecture combines:
    - ResNet-style skip connections for better gradient flow
    - Inception-style multi-scale feature extraction
    - Squeeze-and-Excitation blocks for channel attention
    - Progressive feature extraction with residual blocks
    - Strategic dropout and regularization

    The model is designed to significantly improve upon the baseline CustomCNN
    performance (8.77% test accuracy) and approach the pre-trained VGG16
    performance (78.02% test accuracy) using a custom architecture.
    """

    def __init__(self, input_channels: int = 3, dropout_rate: float = 0.5):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        # Initial convolution - extract low-level features
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 48x48 -> 24x24
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, stride=1, use_se=True),
            ResidualBlock(64, 64, stride=1, use_se=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Stage 2: 24x24 -> 12x12 with Inception module
        self.stage2 = nn.Sequential(
            InceptionModule(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.5)  # Spatial dropout
        )

        # Stage 3: 12x12 -> 6x6
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=1, use_se=True),
            ResidualBlock(256, 256, stride=1, use_se=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Stage 4: 6x6 -> 3x3 with Inception module
        self.stage4 = nn.Sequential(
            InceptionModule(256, 384),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.6)
        )

        # Stage 5: 3x3 -> 1x1
        self.stage5 = nn.Sequential(
            ResidualBlock(384, 512, stride=1, use_se=True),
            ResidualBlock(512, 512, stride=1, use_se=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Fully connected layers with residual-style connections
        self.fc1 = nn.Linear(512, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc1_dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(1024, 1024)
        self.fc2_bn = nn.BatchNorm1d(1024)
        self.fc2_dropout = nn.Dropout(dropout_rate * 0.5)

        # Multi-output heads
        self.output_heads = MultiOutputHead(in_features=1024)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Dictionary with 6 output tensors
        """
        # Feature extraction with progressive refinement
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers with skip-like structure
        identity = x
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = torch.relu(x)
        x = self.fc1_dropout(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = torch.relu(x)
        x = self.fc2_dropout(x)

        # Multi-output prediction
        return self.output_heads(x)


class AlternativeCNN(nn.Module):
    """
    Alternative CNN architecture with Spatial Transformer Network.

    This architecture explores a different approach using:
    - Spatial Transformer Network (STN) for spatial invariance
    - 4 convolutional blocks with progressive feature extraction
    - Batch normalization and dropout for regularization
    - Dual fully connected layers (1024 -> 1024)
    - 6 output heads for multi-task learning

    The STN module learns to apply spatial transformations to focus on
    relevant image regions before passing through the main CNN pipeline.
    """

    def __init__(self, input_channels: int = 3, dropout_rate: float = 0.25):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        # Spatial Transformer Network for spatial invariance
        self.stn = STN(input_channel=input_channels)

        # Convolutional blocks with progressive feature extraction
        def conv_block(in_channels, out_channels):
            """Create a convolutional block with two conv layers, batch norm, pooling, and dropout."""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout(dropout_rate)
            )

        # Feature extraction blocks
        # Input: 48x48 -> Block1: 24x24 -> Block2: 12x12 -> Block3: 6x6 -> Block4: 3x3
        self.block1 = conv_block(input_channels, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)
        self.block4 = conv_block(128, 256)

        # Calculate flattened feature size: 256 channels * 3x3 spatial dims
        self.flatten_dim = 256 * 3 * 3

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_dim, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)

        # Multi-output heads (using separate Linear layers for compatibility)
        self.head_len = nn.Linear(1024, 5)   # Number of digits (0-4)
        self.head_d1 = nn.Linear(1024, 11)   # Digit 1 (0-9 + blank)
        self.head_d2 = nn.Linear(1024, 11)   # Digit 2 (0-9 + blank)
        self.head_d3 = nn.Linear(1024, 11)   # Digit 3 (0-9 + blank)
        self.head_d4 = nn.Linear(1024, 11)   # Digit 4 (0-9 + blank)
        self.head_nc = nn.Linear(1024, 2)    # Has digits classifier

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Dictionary with 6 output tensors
        """
        # Apply spatial transformation
        x = self.stn(x)

        # Feature extraction through convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = torch.relu(self.bn_fc2(self.fc2(x)))

        # Multi-output prediction (return as dictionary for consistency)
        return {
            'num': self.head_len(x),
            'dig1': self.head_d1(x),
            'dig2': self.head_d2(x),
            'dig3': self.head_d3(x),
            'dig4': self.head_d4(x),
            'nC': self.head_nc(x),
        }


def get_model(model_name: str, input_channels: int = 3, **kwargs):
    """
    Factory function to get a model by name.

    Args:
        model_name: One of 'custom', 'improved', 'alternative', 'vgg16_scratch', 'vgg16_pretrained'
        input_channels: Number of input channels
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance

    Raises:
        ValueError: If model_name is not recognized
    """
    models_map = {
        'custom': CustomCNN,
        'improved': ImprovedCNN,
        'alternative': AlternativeCNN,
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

    for model_name in ['custom', 'improved', 'alternative', 'vgg16_scratch', 'vgg16_pretrained']:
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
