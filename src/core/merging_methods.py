import copy

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.core.fisher_weighted_averaging import FisherWeightedAveraging
from src.utils.utiliy import validate_model, plot_layer_distributions_with_stats


class NaiveMerging:
    @staticmethod
    def merge_models(model1, model2, alpha=0.5):
        """
        Naive isotropic merging of two models by averaging their parameters.

        :param model1: The first model.
        :param model2: The second model.
        :param alpha: Weight for model1 (default: 0.5).
        :return: A new merged model.
        """
        # Create a deep copy of model1 to avoid modifying it directly
        merged_model = copy.deepcopy(model1)

        # Ensure both models are on the same device
        device = next(model1.parameters()).device
        model2.to(device)
        merged_model.to(device)

        # Merge parameters
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 == name2:
                merged_param = alpha * param1.data + (1 - alpha) * param2.data
                merged_model.state_dict()[name1].copy_(merged_param)

        return merged_model


import torch
import torch.nn as nn


class OutputEnsembling:
    @staticmethod
    def merge_models(model1, model2):
        """
        Create an ensemble model that averages the predictions of model1 and model2.

        :param model1: The first model.
        :param model2: The second model.
        :return: A merged ensemble model.
        """

        class MergedEnsembleModel(nn.Module):
            def __init__(self, model1, model2):
                super(MergedEnsembleModel, self).__init__()
                self.model1 = model1
                self.model2 = model2

            def forward(self, x):
                # Forward pass through both models and average the predictions
                output1 = self.model1(x)
                output2 = self.model2(x)
                return (output1 + output2) / 2.0

        # Ensure models are on the same device
        device = next(model1.parameters()).device
        model2.to(device)

        # Create and return the ensembled model
        merged_model = MergedEnsembleModel(model1, model2)
        return merged_model


def comparison_pipeline(train_loader, test_loader, model1, model2, criterion):
    """
    Run the comparison pipeline for Isotropic Merging, Fisher Merging, and Output Ensembling.
    """
    # Step 1: Perform Isotropic Merging
    print("Performing Isotropic Merging...")
    isotropic_model = NaiveMerging.merge_models(model1, model2)

    # Step 2: Perform Fisher Merging
    print("Computing Fisher Information for model1...")
    fisher1 = FisherWeightedAveraging.compute_fisher_information(model1, train_loader, criterion)

    print("Computing Fisher Information for model2...")
    fisher2 = FisherWeightedAveraging.compute_fisher_information(model2, train_loader, criterion)

    print("Performing Fisher Merging...")
    fisher_merged_model = FisherWeightedAveraging.merge_models(model1, model2, fisher1, fisher2)

    # Step 3: Perform Output Ensembling
    print("Performing Output Ensembling...")
    ensemble_model = OutputEnsembling.merge_models(model1, model2)

    # Step 4: Evaluate all models
    print("Evaluating Isotropic Merged Model...")
    isotropic_results = validate_model(isotropic_model, test_loader, criterion)

    print("Evaluating Fisher Merged Model...")
    fisher_results = validate_model(fisher_merged_model, test_loader, criterion)

    print("Evaluating Output Ensembeling Merged Model...")
    ensemble_results = validate_model(ensemble_model, test_loader, criterion)

    print("Isotropic validation:", isotropic_results)
    print("Ensembeling validation:", ensemble_results)
    print("Fisher validation:", fisher_results)

    # Step 5: Generate Comparison Chart
    draw_comparison_chart(isotropic_results, ensemble_results, fisher_results)

    # plot_layer_distributions(ensemble_model, ["fc1.weight", "fc2.weight"], "Ensembeled Merged Model")
    models = [model1, model2, isotropic_model, fisher_merged_model]
    model_labels = ["Model 1", "Model 2", "Isotropic Merged", "Fisher Merged"]
    layer_names = ["fc1.weight", "fc2.weight"]

    # Plot all layer distributions together
    plot_layer_distributions_with_stats(models, layer_names, model_labels)


def draw_comparison_chart(isotropic_results, ensemble_results, fisher_results):
    """
    Generate a comparison chart based on the validation results of isotropic merging,
    Fisher merging, and output ensembling.

    :param isotropic_results: Dictionary containing validation metrics for isotropic merging.
    :param ensemble_results: Dictionary containing validation metrics for output ensembling.
    :param fisher_results: Dictionary containing validation metrics for Fisher merging.
    """
    # Extract accuracy and F1-score from validation results
    accuracies = [
        isotropic_results['accuracy'] * 100,
        fisher_results['accuracy'] * 100,
        ensemble_results['accuracy'] * 100
    ]
    f1_scores = [
        isotropic_results['f1_score'] * 100,
        fisher_results['f1_score'] * 100,
        ensemble_results['f1_score'] * 100
    ]

    # Define labels for the methods
    methods = ["Isotropic Merging", "Fisher Merging", "Output Ensembling"]

    # Plot accuracy comparison
    plt.figure(figsize=(10, 5))
    plt.bar(methods, accuracies, color=["orange", "blue", "green"])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Comparison of Merging Methods: Accuracy")
    plt.show()

    # Plot F1-score comparison
    plt.figure(figsize=(10, 5))
    plt.bar(methods, f1_scores, color=["orange", "blue", "green"])
    plt.ylim(0, 100)
    plt.ylabel("F1-Score (%)")
    plt.title("Comparison of Merging Methods: F1-Score")
    plt.show(block=True)

