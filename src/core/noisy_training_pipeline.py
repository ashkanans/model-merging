import torch
import torch.nn as nn
import torch.optim as optim

from src.core.merging_methods import NaiveMerging
from src.core.training_pipeline import TrainingPipeline
from src.core.fisher_weighted_averaging import FisherWeightedAveraging
from src.utils.dataset_loader import DatasetLoader
from src.utils.model_io import ModelIO
from src.models.mnist_model import MNISTMLP
from src.utils.utiliy import validate_model


def noise_augmentation(data, noise_factor=0.2):
    """
    Add random noise to input data.

    :param data: Input data (e.g., images).
    :param noise_factor: Factor to scale the added noise.
    :return: Noisy data.
    """
    noisy_data = data + noise_factor * torch.randn_like(data)
    noisy_data = torch.clamp(noisy_data, 0.0, 1.0)  # Ensure data remains in valid range
    return noisy_data


def add_label_noise(labels, num_classes, noise_ratio=0.1):
    """
    Add noise to labels by randomly flipping some labels.

    :param labels: Original labels.
    :param num_classes: Number of classes.
    :param noise_ratio: Proportion of labels to randomize.
    :return: Labels with noise added.
    """
    noisy_labels = labels.clone()
    num_noisy = int(noise_ratio * len(labels))
    noisy_indices = torch.randperm(len(labels))[:num_noisy]
    noisy_labels[noisy_indices] = torch.randint(0, num_classes, (len(noisy_indices),))
    return noisy_labels


def noisy_training_pipeline():
    """
    Pipeline to evaluate merging methods for noisy and clean models.
    """
    # Load MNIST dataset
    train_loader, test_loader = DatasetLoader.load_mnist()

    # Define models
    clean_model = MNISTMLP()
    noisy_model = MNISTMLP()

    # Training configurations
    criterion = nn.CrossEntropyLoss()
    optimizer_params = {"lr": 0.01, "weight_decay": 1e-5}

    # Train clean model
    print("Training Clean Model...")
    clean_pipeline = TrainingPipeline(clean_model, train_loader, test_loader, optimizer_params=optimizer_params)
    clean_pipeline.train(epochs=10)
    clean_results = clean_pipeline.test()
    print("Clean Model Results:", clean_results)
    ModelIO.save_model(clean_model, "mnist_clean_model.pth")

    # Train noisy model
    print("Training Noisy Model...")
    # Modify data loader to inject noise
    noisy_train_loader = [
        (noise_augmentation(data), add_label_noise(target, num_classes=10)) for data, target in train_loader
    ]
    noisy_pipeline = TrainingPipeline(noisy_model, noisy_train_loader, test_loader, optimizer_params=optimizer_params)
    noisy_pipeline.train(epochs=10)
    noisy_results = noisy_pipeline.test()
    print("Noisy Model Results:", noisy_results)
    ModelIO.save_model(noisy_model, "mnist_noisy_model.pth")

    # Merge models using Isotropic Merging
    print("Performing Isotropic Merging...")
    isotropic_model = NaiveMerging.merge_models(clean_model, noisy_model)

    # Compute Fisher Information and Merge using Fisher Weighted Averaging
    print("Computing Fisher Information for Clean Model...")
    fisher_clean = FisherWeightedAveraging.compute_fisher_information(clean_model, train_loader, criterion)

    print("Computing Fisher Information for Noisy Model...")
    fisher_noisy = FisherWeightedAveraging.compute_fisher_information(noisy_model, train_loader, criterion)

    print("Performing Fisher Weighted Merging...")
    fisher_merged_model = FisherWeightedAveraging.merge_models(clean_model, noisy_model, fisher_clean, fisher_noisy)

    # Validate models on clean validation set
    print("\nValidating Isotropic Merged Model...")
    isotropic_results = validate_model(isotropic_model, test_loader, criterion)
    print("Isotropic Merged Model:", isotropic_results)

    print("Validating Fisher Merged Model...")
    fisher_results = validate_model(fisher_merged_model, test_loader, criterion)
    print("Fisher Merged Model:", fisher_results)

    # Summary
    print("\n--- Summary of Results ---")
    print("Clean Model:", clean_results)
    print("Noisy Model:", noisy_results)
    print("Isotropic Merged Model:", isotropic_results)
    print("Fisher Merged Model:", fisher_results)
