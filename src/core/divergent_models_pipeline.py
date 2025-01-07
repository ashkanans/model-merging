from torch import nn

from src.core.fisher_weighted_averaging import FisherWeightedAveraging
from src.core.merging_methods import NaiveMerging
from src.core.training_pipeline import TrainingPipeline
from src.models.mnist_model import MNISTMLP
from src.utils.dataset_loader import DatasetLoader
from src.utils.model_io import ModelIO
from src.utils.utiliy import validate_model


def divergent_models_pipeline():
    """
    Pipeline to evaluate merging across divergent models trained on MNIST and Fashion-MNIST.
    """
    # Load MNIST and Fashion-MNIST datasets
    print("Loading datasets...")
    mnist_train_loader, mnist_test_loader = DatasetLoader.load_mnist()
    fashion_train_loader, fashion_test_loader = DatasetLoader.load_fashion_mnist()

    # Define models
    model1 = MNISTMLP()
    model2 = MNISTMLP()

    # Training configurations
    criterion = nn.CrossEntropyLoss()
    optimizer1_params = {"lr": 0.01, "weight_decay": 1e-5}
    optimizer2_params = {"lr": 0.001, "weight_decay": 1e-4}

    # Train model1 on MNIST
    print("Training Model 1 on MNIST...")
    pipeline1 = TrainingPipeline(model1, mnist_train_loader, mnist_test_loader, optimizer_params=optimizer1_params)
    pipeline1.train(epochs=10)
    mnist_results = pipeline1.test()
    print("MNIST Model Results:", mnist_results)

    # Train model2 on Fashion-MNIST
    print("Training Model 2 on Fashion-MNIST...")
    pipeline2 = TrainingPipeline(model2, fashion_train_loader, fashion_test_loader, optimizer_params=optimizer2_params)
    pipeline2.train(epochs=10)
    fashion_results = pipeline2.test()
    print("Fashion-MNIST Model Results:", fashion_results)

    # Save the trained models
    ModelIO.save_model(model1, "mnist_model.pth")
    ModelIO.save_model(model2, "fashion_mnist_model.pth")

    # Merge models using Isotropic Merging
    print("Performing Isotropic Merging...")
    isotropic_model = NaiveMerging.merge_models(model1, model2)

    # Compute Fisher Information and Merge using Fisher Weighted Averaging
    print("Computing Fisher Information for Model 1...")
    fisher1 = FisherWeightedAveraging.compute_fisher_information(model1, mnist_train_loader, criterion)

    print("Computing Fisher Information for Model 2...")
    fisher2 = FisherWeightedAveraging.compute_fisher_information(model2, fashion_train_loader, criterion)

    print("Performing Fisher Weighted Merging...")
    fisher_merged_model = FisherWeightedAveraging.merge_models(model1, model2, fisher1, fisher2)

    # Validate the merged models on both datasets
    print("\nValidating Isotropic Merged Model on MNIST...")
    isotropic_mnist_results = validate_model(isotropic_model, mnist_test_loader, criterion)
    print("Isotropic Merged Model on MNIST:", isotropic_mnist_results)

    print("Validating Isotropic Merged Model on Fashion-MNIST...")
    isotropic_fashion_results = validate_model(isotropic_model, fashion_test_loader, criterion)
    print("Isotropic Merged Model on Fashion-MNIST:", isotropic_fashion_results)

    print("\nValidating Fisher Merged Model on MNIST...")
    fisher_mnist_results = validate_model(fisher_merged_model, mnist_test_loader, criterion)
    print("Fisher Merged Model on MNIST:", fisher_mnist_results)

    print("Validating Fisher Merged Model on Fashion-MNIST...")
    fisher_fashion_results = validate_model(fisher_merged_model, fashion_test_loader, criterion)
    print("Fisher Merged Model on Fashion-MNIST:", fisher_fashion_results)

    # Summary
    print("\n--- Summary of Results ---")
    print("Isotropic Merged Model on MNIST:", isotropic_mnist_results)
    print("Isotropic Merged Model on Fashion-MNIST:", isotropic_fashion_results)
    print("Fisher Merged Model on MNIST:", fisher_mnist_results)
    print("Fisher Merged Model on Fashion-MNIST:", fisher_fashion_results)


