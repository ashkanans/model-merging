import argparse

from src.core.cli_tool import CLITool


def main():
    """
    Main entry point for the CLI tool.
    """
    parser = argparse.ArgumentParser(description="Train, merge, validate, and visualize neural networks.")
    parser.add_argument("--dataset", type=str, nargs="+", required=True,
                        choices=["mnist", "fashion-mnist", "cifar10"],
                        help="Dataset(s) to use. Provide a list for divergent models (e.g., MNIST and Fashion-MNIST).")
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "cnn", "resnet", "vgg"],
                        help="Model architecture to train.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--merge_type", type=str, choices=["fisher", "isotropic", "ensemble"],
                        help="Type of merging technique to use.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha for Fisher-weighted averaging (default: 0.5).")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze the components of Fisher merging.")
    parser.add_argument("--comparison_chart", action="store_true",
                        help="Draw Comparison Chart")
    parser.add_argument("--layer_dist", action="store_true",
                        help="Draw Layer Distribution Chart Chart")
    parser.add_argument("--generalize", action="store_true",
                        help="Test merging generalization across architectures.")
    parser.add_argument("--command", type=str, nargs="+", required=True,
                        choices=["train", "merge", "validate", "visualize", "ablation"],
                        help="Commands to execute: train, merge, validate, ablation, visualize.")
    parser.add_argument("--noisy_models", type=str, nargs="*", choices=["model1", "model2"],
                        help="Specify which models to train with noise (model1, model2, or both).")

    args = parser.parse_args()

    cli_tool = CLITool(args)
    cli_tool.run()


if __name__ == "__main__":
    main()

