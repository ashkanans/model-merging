from torch import nn


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
