"""
Main training orchestrator for SVHN digit detection models.

This script provides a unified interface to train any of the available model
architectures. It can be run directly or imported as a module.

Available models:
- custom: Custom designed deep CNN architecture
- vgg16_scratch: VGG-16 trained from random initialization
- vgg16_pretrained: VGG-16 with ImageNet pre-trained weights (recommended)

Usage:
    # Train specific model
    python train.py --model custom
    python train.py --model vgg16_pretrained

    # Train with custom parameters
    python train.py --model custom --epochs 100 --batch-size 128

    # Train all models
    python train.py --all
"""

import argparse
import sys
from pathlib import Path

# Ensure models package is importable
sys.path.insert(0, str(Path(__file__).parent))


def train_model(
    model_name: str,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
):
    """
    Train a specific model.

    Args:
        model_name: Name of model to train ('custom', 'vgg16_scratch', 'vgg16_pretrained')
        epochs: Number of training epochs (uses default if None)
        batch_size: Training batch size (uses default if None)
        learning_rate: Initial learning rate (uses default if None)

    Returns:
        Tuple of (trained_model, training_history)

    Raises:
        ValueError: If model_name is not recognized
    """
    # Default hyperparameters per model
    defaults = {
        "custom": {"epochs": 75, "batch_size": 64, "learning_rate": 0.001},
        "vgg16_scratch": {"epochs": 50, "batch_size": 64, "learning_rate": 0.001},
        "vgg16_pretrained": {"epochs": 50, "batch_size": 64, "learning_rate": 0.001},
    }

    if model_name not in defaults:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {', '.join(defaults.keys())}"
        )

    # Get default parameters
    params = defaults[model_name].copy()

    # Override with provided parameters
    if epochs is not None:
        params["epochs"] = epochs
    if batch_size is not None:
        params["batch_size"] = batch_size
    if learning_rate is not None:
        params["learning_rate"] = learning_rate

    # Import and train the appropriate model
    if model_name == "custom":
        from models.train_custom_cnn import train

        return train(**params)

    elif model_name == "vgg16_scratch":
        from models.train_vgg16_scratch import train

        return train(**params)

    elif model_name == "vgg16_pretrained":
        from models.train_vgg16_pretrained import train

        return train(**params)


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train SVHN digit detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train pre-trained VGG-16 (recommended, achieves ~91% accuracy)
  python train.py --model vgg16_pretrained

  # Train custom CNN with more epochs
  python train.py --model custom --epochs 100

  # Train all models sequentially
  python train.py --all

Available models:
  custom            - Custom designed deep CNN (75 epochs default)
  vgg16_scratch     - VGG-16 from random initialization (50 epochs default)
  vgg16_pretrained  - VGG-16 with ImageNet weights (50 epochs default, best performance)
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["custom", "vgg16_scratch", "vgg16_pretrained"],
        help="Model architecture to train",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all available models sequentially",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides default)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size (overrides default)",
    )

    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        help="Initial learning rate (overrides default)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.model and not args.all:
        parser.error("Must specify either --model or --all")

    if args.model and args.all:
        parser.error("Cannot specify both --model and --all")

    # Train models
    if args.all:
        print("\n" + "=" * 70)
        print("TRAINING ALL MODELS")
        print("=" * 70 + "\n")

        models_to_train = ["custom", "vgg16_scratch", "vgg16_pretrained"]
        results = {}

        for model_name in models_to_train:
            try:
                print(f"\n{'='*70}")
                print(f"Starting training: {model_name}")
                print(f"{'='*70}\n")

                model, history = train_model(
                    model_name,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                )
                results[model_name] = {"model": model, "history": history}

                print(f"\n✓ {model_name} training complete\n")

            except Exception as e:
                print(f"\n✗ Error training {model_name}: {e}\n")
                results[model_name] = {"error": str(e)}

        # Print summary
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        for model_name, result in results.items():
            if "error" in result:
                print(f"✗ {model_name}: Failed - {result['error']}")
            else:
                print(f"✓ {model_name}: Success")
        print()

    else:
        # Train single model
        try:
            model, history = train_model(
                args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            print(f"\n✓ Successfully trained {args.model}\n")
            return model, history

        except Exception as e:
            print(f"\n✗ Error training {args.model}: {e}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
