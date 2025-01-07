import matplotlib.pyplot as plt

from src.core.fisher_weighted_averaging import FisherWeightedAveraging
from src.utils.utiliy import validate_model


def fisher_scaling_experiment(train_loader, test_loader, model1, model2, fisher1, fisher2, criterion):
    """
    Experiment with different scaling factors for Fisher Information during merging.

    :param train_loader: DataLoader for the training dataset.
    :param test_loader: DataLoader for the test dataset.
    :param model1: The first pre-trained model.
    :param model2: The second pre-trained model.
    :param fisher1: Fisher information for the first model.
    :param fisher2: Fisher information for the second model.
    :param criterion: Loss function for evaluation.
    """
    scaling_factors = [0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0]
    accuracies = []

    print("Running Fisher Scaling Experiment...")
    for beta in scaling_factors:
        print(f"Testing Fisher Scaling Factor β = {beta}...")
        merged_model = FisherWeightedAveraging.merge_models(
            model1, model2, fisher1, fisher2, fisher_scaling=beta
        )

        # Evaluate the merged model
        results = validate_model(merged_model, test_loader, criterion)
        print(f"Accuracy for β = {beta}: {results['accuracy']:.4f}")
        accuracies.append(results['accuracy'])

    # Plot Accuracy vs. β
    plt.figure(figsize=(8, 6))
    plt.plot(scaling_factors, [acc * 100 for acc in accuracies], marker='o', label="Accuracy")
    plt.xlabel("Fisher Scaling Factor (β)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Fisher Scaling Factor (β)")
    plt.grid(True)
    plt.legend()
    plt.show(block=True)
