from src.core.fisher_weighted_averaging import FisherWeightedAveraging


class FisherCalculations:
    @staticmethod
    def compute_fisher_information(model, data_loader, criterion):
        return FisherWeightedAveraging.compute_fisher_information(model, data_loader, criterion)
