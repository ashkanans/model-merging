import torch
import torch.nn as nn
import torch.optim as optim


class TrainingPipeline:
    def __init__(self, model, train_loader, test_loader, device=None, optimizer_params=None):
        """
        Initialize the TrainingPipeline.

        :param model: The model to train and evaluate.
        :param train_loader: DataLoader for the training dataset.
        :param test_loader: DataLoader for the test dataset.
        :param device: The device to use (default: automatically selects "cuda" if available).
        :param optimizer_params: Dictionary of optimizer parameters (e.g., {"lr": 0.01, "weight_decay": 1e-5}).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()

        # If optimizer_params is provided, use them; otherwise, use default Adam optimizer
        if optimizer_params:
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_params)
        else:
            self.optimizer = optim.Adam(self.model.parameters())

    def train(self, epochs=10):
        """
        Train the model.

        :param epochs: Number of training epochs (default: 10).
        """
        print(f"Training on {self.device}...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(self.train_loader)}")

        # After training, move the model to CPU
        self.model.cpu()
        print("Training complete. Model moved to CPU.")

    def test(self):
        """
        Evaluate the model on the test dataset.
        """
        # Ensure the model is on the same device as the evaluation data
        self.model.to(self.device)
        self.model.eval()

        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(self.test_loader.dataset) * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy
