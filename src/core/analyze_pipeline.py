from src.core.fisher_weighted_averaging import FisherWeightedAveraging
from src.core.merging_methods import NaiveMerging
from src.utils.utiliy import validate_model


def analyze_pipeline(
        train_loader,
        test_loader,
        model1,
        model2,
        criterion,
        use_uniform_weights=False,
        layer_wise_fisher=None,
        fisher_scaling=1.0
):
    """
    Compare Fisher-weighted averaging, isotropic merging, and additional experiments.

    :param train_loader: DataLoader for training data.
    :param test_loader: DataLoader for testing data.
    :param model1: The first model to merge.
    :param model2: The second model to merge.
    :param criterion: Loss function for Fisher Information computation.
    :param use_uniform_weights: Disable Fisher weighting (use uniform weights).
    :param layer_wise_fisher: List of layers to apply Fisher merging; isotropic for others.
    :param fisher_scaling: Scaling factor for Fisher Information.
    """
    # Perform Isotropic Merging
    print("Performing Isotropic Merging...")
    isotropic_model = NaiveMerging.merge_models(model1, model2)

    # Compute Fisher Information
    print("Computing Fisher Information for Model 1...")
    fisher1 = FisherWeightedAveraging.compute_fisher_information(model1, train_loader, criterion)

    print("Computing Fisher Information for Model 2...")
    fisher2 = FisherWeightedAveraging.compute_fisher_information(model2, train_loader, criterion)

    # Perform Fisher Weighted Merging
    print("Performing Fisher Merging...")
    fisher_merged_model = FisherWeightedAveraging.merge_models(
        model1, model2, fisher1, fisher2,
        uniform_weights=use_uniform_weights,
        layer_wise_fisher=layer_wise_fisher,
        fisher_scaling=fisher_scaling
    )

    # Validate Models
    print("Validating Isotropic Merged Model...")
    isotropic_results = validate_model(isotropic_model, test_loader, criterion)
    print("Isotropic Merged Model:", isotropic_results)

    print("Validating Fisher Merged Model...")
    fisher_results = validate_model(fisher_merged_model, test_loader, criterion)
    print("Fisher Merged Model:", fisher_results)
