import copy
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class FisherWeightedAveraging:
    @staticmethod
    def compute_fisher_information(model, data_loader, criterion):
        """
        Compute Fisher Information for a given model and dataset.

        :param model: The model for which Fisher information is computed.
        :param data_loader: DataLoader providing the dataset.
        :param criterion: Loss function used for backpropagation.
        :return: A dictionary containing the Fisher information for each parameter.
        """
        fisher_matrix = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        for data, target in tqdm(data_loader, desc="Computing Fisher Information", unit="batch"):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in fisher_matrix:
                        fisher_matrix[name] = param.grad.data.clone().pow(2)
                    else:
                        fisher_matrix[name] += param.grad.data.clone().pow(2)

        # Normalize Fisher information
        for name in fisher_matrix:
            fisher_matrix[name] /= len(data_loader)

        return fisher_matrix

    @staticmethod
    def merge_models(model1, model2, fisher1, fisher2, alpha=0.5):
        """
        Merge two models using Fisher-weighted averaging.

        :param model1: The first model to merge.
        :param model2: The second model to merge.
        :param fisher1: Fisher information for the first model.
        :param fisher2: Fisher information for the second model.
        :param alpha: Balancing factor for Fisher-weighted averaging (default: 0.5).
        :return: The merged model.
        """
        # Create a deep copy of model1 for the merged model
        merged_model = copy.deepcopy(model1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        merged_model.to(device)
        model2.to(device)

        # Add epsilon for numerical stability
        epsilon = 1e-8

        # Merge parameters using Fisher-weighted averaging
        for (name1, param1), (name2, param2) in tqdm(
            zip(merged_model.named_parameters(), model2.named_parameters()),
            desc="Merging Models",
            unit="param"
        ):
            if name1 in fisher1 and name1 in fisher2:
                # Get Fisher weights and normalize
                weight1 = fisher1[name1].to(device)
                weight2 = fisher2[name1].to(device)

                # Normalize Fisher weights
                total_weight = weight1 + weight2 + epsilon
                merged_param = (alpha * weight1 * param1.data + (1 - alpha) * weight2 * param2.data) / total_weight

                # Update the merged model parameter
                param1.data.copy_(merged_param)

        return merged_model

    @staticmethod
    def validate_model(model, data_loader, criterion):
        """
        Validate a model on a dataset and compute metrics.

        :param model: The model to validate.
        :param data_loader: DataLoader providing the validation dataset.
        :param criterion: Loss function used for evaluation.
        :return: A dictionary containing evaluation metrics: accuracy, F1-score, precision, recall, and loss.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        all_targets = []
        all_predictions = []
        total_loss = 0.0

        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Validating Model", unit="batch"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

                _, predictions = torch.max(output, dim=1)
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

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
