from torch import nn

from src.core.isotropic_merging import NaiveMerging
from src.core.training_pipeline import TrainingPipeline
from src.utils.utiliy import validate_model


def generalization_pipeline(dataset, model1_type, model2_type, train_loader, test_loader, epochs=10):
    """
    Pipeline to evaluate merging techniques across different architectures.

    :param dataset: The dataset being used (e.g., CIFAR-10).
    :param model1_type: The architecture of the first model (e.g., ResNet-18).
    :param model2_type: The architecture of the second model (e.g., VGG).
    :param train_loader: DataLoader for the training dataset.
    :param test_loader: DataLoader for the test dataset.
    :param epochs: Number of training epochs for each model.
    """
    from src.models.cifar10_models import CIFAR10ResNet, CIFAR10VGG
    from src.core.fisher_weighted_averaging import FisherWeightedAveraging
    from src.utils.model_io import ModelIO

    # Define models based on architecture type
    if model1_type == "resnet":
        model1 = CIFAR10ResNet()
    elif model1_type == "vgg":
        model1 = CIFAR10VGG()

    if model2_type == "resnet":
        model2 = CIFAR10ResNet()
    elif model2_type == "vgg":
        model2 = CIFAR10VGG()

    # Training configurations
    criterion = nn.CrossEntropyLoss()
    optimizer_params = {"lr": 0.01, "weight_decay": 1e-5}

    # Train first model
    print(f"Training {model1_type.upper()} Model 1 on {dataset}...")
    pipeline1 = TrainingPipeline(model1, train_loader, test_loader, optimizer_params=optimizer_params)
    pipeline1.train(epochs=epochs)
    results1 = pipeline1.test()
    print(validate_model(model1, test_loader, criterion))
    print(f"{model1_type.upper()} Model 1 Results:", results1)
    ModelIO.save_model(model1, f"{dataset}_{model1_type}_1.pth")

    # Train second model
    print(f"Training {model2_type.upper()} Model 2 on {dataset}...")
    pipeline2 = TrainingPipeline(model2, train_loader, test_loader, optimizer_params=optimizer_params)
    pipeline2.train(epochs=epochs)
    results2 = pipeline2.test()
    print(validate_model(model2, test_loader, criterion))
    print(f"{model2_type.upper()} Model 2 Results:", results2)
    ModelIO.save_model(model2, f"{dataset}_{model2_type}_2.pth")

    # Merge models using Isotropic Merging
    print("Performing Isotropic Merging...")
    isotropic_model = NaiveMerging.merge_models(model1, model2)

    # Compute Fisher Information and Merge using Fisher Weighted Averaging
    print("Computing Fisher Information for Model 1...")
    fisher1 = FisherWeightedAveraging.compute_fisher_information(model1, train_loader, criterion)

    print("Computing Fisher Information for Model 2...")
    fisher2 = FisherWeightedAveraging.compute_fisher_information(model2, train_loader, criterion)

    print("Performing Fisher Weighted Merging...")
    fisher_merged_model = FisherWeightedAveraging.merge_models(model1, model2, fisher1, fisher2)

    # Validate models on the test set
    print("\nValidating Isotropic Merged Model...")
    isotropic_results = validate_model(isotropic_model, test_loader, criterion)
    print("Isotropic Merged Model Results:", isotropic_results)

    print("\nValidating Fisher Merged Model...")
    fisher_results = validate_model(fisher_merged_model, test_loader, criterion)
    print("Fisher Merged Model Results:", fisher_results)

    # Summary
    return {
        "model1_results": results1,
        "model2_results": results2,
        "isotropic_results": isotropic_results,
        "fisher_results": fisher_results,
    }
