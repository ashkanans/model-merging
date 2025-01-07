from src.core.fisher_weighted_averaging import FisherWeightedAveraging
from src.utils.fisher_calculations import FisherCalculations


class FisherMergingService:
    def __init__(self, model1, model2, data_loader, criterion):
        self.model1 = model1
        self.model2 = model2
        self.data_loader = data_loader
        self.criterion = criterion

    def merge(self):
        fisher1 = FisherCalculations.compute_fisher_information(self.model1, self.data_loader, self.criterion)
        fisher2 = FisherCalculations.compute_fisher_information(self.model2, self.data_loader, self.criterion)
        return FisherWeightedAveraging.merge_models(self.model1, self.model2, fisher1, fisher2)
