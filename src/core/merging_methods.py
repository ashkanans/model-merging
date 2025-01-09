import copy

import torch.nn as nn


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
