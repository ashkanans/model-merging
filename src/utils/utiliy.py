import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


def validate_model(model, data_loader, criterion):
    """
    Validate a model on a dataset or multiple datasets and compute metrics.

    :param model: The model to validate.
    :param data_loader: DataLoader or a dictionary of DataLoaders for validation datasets.
    :param criterion: Loss function used for evaluation.
    :return: A dictionary containing evaluation metrics for each dataset:
             {dataset_name: {"accuracy", "f1_score", "precision", "recall", "loss"}}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def _validate_single(data_loader):
        """
        Validate the model on a single dataset.

        :param data_loader: DataLoader for the dataset.
        :return: Dictionary with evaluation metrics.
        """
        all_targets = []
        all_predictions = []
        total_loss = 0.0

        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Validating Model", unit="batch"):
                # Move data and target to the appropriate device
                data, target = data.to(device), target.to(device)

                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

                # Convert logits to predictions
                _, predictions = torch.max(output, dim=1)

                # Collect predictions and targets for metric calculation
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')

        return {
            "loss": total_loss / len(data_loader),
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }

    # Handle single DataLoader or dictionary of DataLoaders
    if isinstance(data_loader, dict):
        results = {}
        for dataset_name, loader in data_loader.items():
            print(f"Validating on {dataset_name} dataset...")
            results[dataset_name] = _validate_single(loader)
        return results
    else:
        # Single dataset
        return _validate_single(data_loader)


def plot_layer_distributions(model, layer_names, title):
    for layer_name in layer_names:
        layer_weights = model.state_dict()[layer_name].flatten().cpu().numpy()
        plt.hist(layer_weights, bins=50, alpha=0.6, label=layer_name)
    plt.title(title)
    plt.xlabel("Weight Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show(block=True)


def plot_layer_distributions_with_stats(models, layer_names, model_labels):
    """
    Plot the weight distributions for specific layers across multiple models in a grid layout,
    with statistics (mean, std, min, max) displayed on each plot.

    :param models: List of models to visualize (e.g., [model1, model2, isotropic_model, fisher_merged_model]).
    :param layer_names: List of layer names (keys from state_dict) to plot.
    :param model_labels: List of labels corresponding to each model.
    """
    # Ensure input lists are consistent
    assert len(models) == len(model_labels), "Each model must have a corresponding label."

    num_layers = len(layer_names)
    num_models = len(models)

    # Create subplots
    fig, axes = plt.subplots(num_layers, num_models, figsize=(4 * num_models, 4 * num_layers), sharey=True)
    axes = axes if num_layers > 1 else [axes]  # Ensure axes is always 2D

    for i, layer_name in enumerate(layer_names):
        for j, (model, label) in enumerate(zip(models, model_labels)):
            ax = axes[i][j] if num_layers > 1 else axes[j]

            # Extract weights for the specified layer
            state_dict = model.state_dict()
            if layer_name in state_dict:
                weights = state_dict[layer_name].flatten().cpu().numpy()
                mean_val = weights.mean()
                std_val = weights.std()
                min_val = weights.min()
                max_val = weights.max()

                # Plot histogram
                ax.hist(weights, bins=50, alpha=0.6, color="blue")
                ax.set_title(f"{label} ({layer_name})", fontsize=10)
                ax.set_xlabel("Weight Values", fontsize=8)
                ax.set_ylabel("Frequency", fontsize=8)

                # Add statistics as text on the plot
                stats_text = f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}"
                ax.text(0.95, 0.95, stats_text, fontsize=8, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle="round", alpha=0.3))
            else:
                # If layer not found, display a message
                ax.text(0.5, 0.5, f"Layer {layer_name} not found", ha='center', va='center')
                ax.set_title(f"{label} ({layer_name})")
                ax.axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show(block=True)


def plot_fisher_information(fisher_matrix, layer_names):
    """
    Plot Fisher Information across layers to identify contributions.

    :param fisher_matrix: Dictionary of Fisher Information for each layer.
    :param layer_names: List of layer names to visualize.
    """
    # Sum Fisher Information for each layer
    values = [fisher_matrix[layer_name].sum().item() for layer_name in layer_names]

    # Plot Fisher Information
    plt.figure(figsize=(10, 6))
    plt.bar(layer_names, values, color='skyblue')
    plt.title("Fisher Information by Layer")
    plt.xlabel("Layer")
    plt.ylabel("Fisher Information Sum")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show(block=True)


def plot_generalization_results(results, model1_type, model2_type):
    """
    Plot generalization results for merging techniques.

    :param results: Dictionary containing validation results.
    :param model1_type: Architecture of the first model.
    :param model2_type: Architecture of the second model.
    """
    # Extract accuracy values for plotting
    accuracy = [
        results["model1_results"],  # Direct accuracy value
        results["model2_results"],  # Direct accuracy value
        results["isotropic_results"]["accuracy"] * 100,  # Nested dictionary
        results["fisher_results"]["accuracy"] * 100,  # Nested dictionary
    ]

    # Define labels for each bar
    labels = [
        f"{model1_type.upper()} Model",
        f"{model2_type.upper()} Model",
        "Isotropic Merged",
        "Fisher Merged",
    ]

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(labels, accuracy, color=["blue", "green", "orange", "purple"])
    plt.title(f"Generalization of Merging Techniques ({model1_type.upper()} vs. {model2_type.upper()})")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show(block=True)


def draw_comparison_chart(isotropic_results, ensemble_results, fisher_results):
    """
    Generate comparison charts for each dataset or a single dataset based on the validation results
    of isotropic merging, Fisher merging, and output ensembling.

    :param isotropic_results: Dictionary containing validation metrics for isotropic merging.
    :param ensemble_results: Dictionary containing validation metrics for output ensembling.
    :param fisher_results: Dictionary containing validation metrics for Fisher merging.
    """
    # Check if the results are for a single dataset or multiple datasets
    is_single_dataset = isinstance(isotropic_results, dict) and 'accuracy' in isotropic_results

    if is_single_dataset:
        # Handle single dataset case
        print("Generating comparison charts for a single dataset...")

        # Extract accuracy and F1-score
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
        plt.show()
    else:
        # Handle multiple datasets case
        print("Generating comparison charts for multiple datasets...")
        dataset_names = isotropic_results.keys()

        for dataset in dataset_names:
            print(f"Generating comparison charts for {dataset}...")

            # Extract accuracy and F1-score for the current dataset
            accuracies = [
                isotropic_results[dataset]['accuracy'] * 100,
                fisher_results[dataset]['accuracy'] * 100,
                ensemble_results[dataset]['accuracy'] * 100
            ]
            f1_scores = [
                isotropic_results[dataset]['f1_score'] * 100,
                fisher_results[dataset]['f1_score'] * 100,
                ensemble_results[dataset]['f1_score'] * 100
            ]

            # Define labels for the methods
            methods = ["Isotropic Merging", "Fisher Merging", "Output Ensembling"]

            # Plot accuracy comparison for the current dataset
            plt.figure(figsize=(10, 5))
            plt.bar(methods, accuracies, color=["orange", "blue", "green"])
            plt.ylim(0, 100)
            plt.ylabel("Accuracy (%)")
            plt.title(f"Comparison of Merging Methods: Accuracy ({dataset})")
            plt.show()

            # Plot F1-score comparison for the current dataset
            plt.figure(figsize=(10, 5))
            plt.bar(methods, f1_scores, color=["orange", "blue", "green"])
            plt.ylim(0, 100)
            plt.ylabel("F1-Score (%)")
            plt.title(f"Comparison of Merging Methods: F1-Score ({dataset})")
            plt.show()
