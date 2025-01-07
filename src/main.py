import argparse

from torch import nn

from models.cifar10_models import CIFAR10ResNet, CIFAR10VGG
from src.core.merging_methods import comparison_pipeline
from src.models.mnist_model import MNISTMLP, MNISTCNN
from src.utils.utiliy import validate_model
from utils.dataset_loader import DatasetLoader
from core.training_pipeline import TrainingPipeline
from core.fisher_weighted_averaging import FisherWeightedAveraging
from utils.model_io import ModelIO


def main():
    parser = argparse.ArgumentParser(description="Train models and merge them with Fisher-weighted averaging.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10"], help="Dataset to use.")
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "cnn", "resnet", "vgg"],
                        help="Model to train.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--merge", action="store_true", help="Merge two pre-trained models.")
    parser.add_argument("--compare", action="store_true", help="Run the comparison pipeline.")
    args = parser.parse_args()

    # Load dataset
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

    # Train models (if not merging or comparing)
    if not args.merge and not args.compare:
        print(f"Training {args.model} on {args.dataset}...")
        pipeline1 = TrainingPipeline(model1, train_loader, test_loader)
        pipeline1.train(epochs=args.epochs)
        pipeline1.test()
        criterion = nn.CrossEntropyLoss()
        print(validate_model(model1, test_loader, criterion))
        ModelIO.save_model(model1, f"{args.dataset}_{args.model}_1.pth")

        pipeline2 = TrainingPipeline(model2, train_loader, test_loader)
        pipeline2.train(epochs=args.epochs)
        pipeline2.test()
        print(validate_model(model2, test_loader, criterion))
        ModelIO.save_model(model2, f"{args.dataset}_{args.model}_2.pth")
        print(f"Trained models saved as '{args.dataset}_{args.model}_1.pth' and '{args.dataset}_{args.model}_2.pth'.")

    # Merge models using Fisher-weighted averaging
    if args.merge:
        # Load pre-trained models
        print("Loading pre-trained models...")
        model1 = ModelIO.load_model(model1, f"{args.dataset}_{args.model}_1.pth")
        model2 = ModelIO.load_model(model2, f"{args.dataset}_{args.model}_2.pth")

        # Load data and criterion
        criterion = nn.CrossEntropyLoss()

        # Compute Fisher Information
        print("Computing Fisher Information for model1...")
        fisher1 = FisherWeightedAveraging.compute_fisher_information(model1, train_loader, criterion)

        print("Computing Fisher Information for model2...")
        fisher2 = FisherWeightedAveraging.compute_fisher_information(model2, train_loader, criterion)

        # Merge Models
        print("Merging models using Fisher-weighted averaging...")
        merged_model = FisherWeightedAveraging.merge_models(model1, model2, fisher1, fisher2, alpha=0.5)

        # Save the Merged Model
        ModelIO.save_model(merged_model, f"{args.dataset}_{args.model}_merged.pth")
        print(f"Merged model saved as '{args.dataset}_{args.model}_merged.pth'.")

        # Perform Validation
        print("\nValidating model1...")
        results_model1 = FisherWeightedAveraging.validate_model(model1, test_loader, criterion)
        print(f"Model 1 Results: {results_model1}")

        print("\nValidating model2...")
        results_model2 = FisherWeightedAveraging.validate_model(model2, test_loader, criterion)
        print(f"Model 2 Results: {results_model2}")

        print("\nValidating merged model...")
        results_merged = FisherWeightedAveraging.validate_model(merged_model, test_loader, criterion)
        print(f"Merged Model Results: {results_merged}")

    # Run the comparison pipeline
    if args.compare:
        # Load pre-trained models
        print("Loading pre-trained models...")
        model1 = ModelIO.load_model(model1, f"{args.dataset}_{args.model}_1.pth")
        model2 = ModelIO.load_model(model2, f"{args.dataset}_{args.model}_2.pth")

        # Define loss criterion
        criterion = nn.CrossEntropyLoss()

        # Run the comparison pipeline
        print("Running the comparison pipeline...")
        comparison_pipeline(train_loader, test_loader, model1, model2, criterion)


if __name__ == "__main__":
    main()
