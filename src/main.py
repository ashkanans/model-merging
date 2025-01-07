import argparse

from torch import nn

from models.cifar10_models import CIFAR10ResNet, CIFAR10VGG
from src.core.divergent_models_pipeline import divergent_models_pipeline
from src.core.merging_methods import comparison_pipeline
from src.models.mnist_model import MNISTMLP, MNISTCNN
from src.utils.utiliy import validate_model
from utils.dataset_loader import DatasetLoader
from core.training_pipeline import TrainingPipeline
from core.fisher_weighted_averaging import FisherWeightedAveraging
from utils.model_io import ModelIO


def main():
    parser = argparse.ArgumentParser(description="Train, merge, and compare neural networks.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10"],
                        help="Dataset to use.")
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "cnn", "resnet", "vgg"],
                        help="Model architecture to train.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--merge", action="store_true", help="Merge two pre-trained models.")
    parser.add_argument("--compare", action="store_true", help="Run the comparison pipeline.")
    parser.add_argument("--divergent", action="store_true", help="Run the divergent models pipeline.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for Fisher-weighted averaging (default: 0.5).")
    args = parser.parse_args()

    # Check for mutually exclusive flags
    if sum([args.merge, args.compare, args.divergent]) > 1:
        raise ValueError("You can only specify one of --merge, --compare, or --divergent.")

    # Handle Divergent Models Pipeline
    if args.divergent:
        print("Running Divergent Models Pipeline...")
        divergent_models_pipeline()
        return

    # Dataset and Model Initialization
    if args.dataset == "mnist":
        train_loader, test_loader = DatasetLoader.load_mnist()
        if args.model == "mlp":
            model1 = MNISTMLP()
            model2 = MNISTMLP()
        elif args.model == "cnn":
            model1 = MNISTCNN()
            model2 = MNISTCNN()
    elif args.dataset == "cifar10":
        train_loader, test_loader = DatasetLoader.load_cifar10()
        if args.model == "resnet":
            model1 = CIFAR10ResNet()
            model2 = CIFAR10ResNet()
        elif args.model == "vgg":
            model1 = CIFAR10VGG()
            model2 = CIFAR10VGG()

    # Handle Training Pipeline
    if not args.merge and not args.compare:
        print(f"Training {args.model} on {args.dataset}...")
        pipeline1 = TrainingPipeline(model1, train_loader, test_loader)
        pipeline1.train(epochs=args.epochs)
        pipeline1.test()
        criterion = nn.CrossEntropyLoss()
        print("Model 1 Validation:", validate_model(model1, test_loader, criterion))
        ModelIO.save_model(model1, f"{args.dataset}_{args.model}_1.pth")

        pipeline2 = TrainingPipeline(model2, train_loader, test_loader)
        pipeline2.train(epochs=args.epochs)
        pipeline2.test()
        print("Model 2 Validation:", validate_model(model2, test_loader, criterion))
        ModelIO.save_model(model2, f"{args.dataset}_{args.model}_2.pth")
        print(f"Trained models saved as '{args.dataset}_{args.model}_1.pth' and '{args.dataset}_{args.model}_2.pth'.")

    # Handle Model Merging
    if args.merge:
        print("Merging Models...")
        model1 = ModelIO.load_model(model1, f"{args.dataset}_{args.model}_1.pth")
        model2 = ModelIO.load_model(model2, f"{args.dataset}_{args.model}_2.pth")
        criterion = nn.CrossEntropyLoss()

        print("Computing Fisher Information...")
        fisher1 = FisherWeightedAveraging.compute_fisher_information(model1, train_loader, criterion)
        fisher2 = FisherWeightedAveraging.compute_fisher_information(model2, train_loader, criterion)

        merged_model = FisherWeightedAveraging.merge_models(model1, model2, fisher1, fisher2, alpha=args.alpha)
        ModelIO.save_model(merged_model, f"{args.dataset}_{args.model}_merged.pth")
        print(f"Merged model saved as '{args.dataset}_{args.model}_merged.pth'.")

        print("\nValidating Merged Model...")
        merged_results = FisherWeightedAveraging.validate_model(merged_model, test_loader, criterion)
        print("Merged Model Results:", merged_results)

    # Handle Comparison Pipeline
    if args.compare:
        print("Running Comparison Pipeline...")
        model1 = ModelIO.load_model(model1, f"{args.dataset}_{args.model}_1.pth")
        model2 = ModelIO.load_model(model2, f"{args.dataset}_{args.model}_2.pth")
        criterion = nn.CrossEntropyLoss()
        comparison_pipeline(train_loader, test_loader, model1, model2, criterion)


if __name__ == "__main__":
    main()
