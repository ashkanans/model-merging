import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


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
