from src.core.fisher_weighted_averaging import FisherWeightedAveraging
from src.utils.utiliy import plot_fisher_information


def fisher_information_visualization_pipeline(model, data_loader, criterion):
    """
    Pipeline to compute and visualize Fisher Information across layers.

    :param model: The model to analyze.
    :param data_loader: DataLoader providing the dataset.
    :param criterion: Loss function for Fisher Information computation.
    """
    # Compute Fisher Information
    print("Computing Fisher Information...")
    fisher_matrix = FisherWeightedAveraging.compute_fisher_information(model, data_loader, criterion)

    # Extract layer names
    layer_names = list(fisher_matrix.keys())

    # Plot Fisher Information
    print("Visualizing Fisher Information...")
    plot_fisher_information(fisher_matrix, layer_names)
