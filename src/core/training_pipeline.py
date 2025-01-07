import torch
import torch.nn as nn
import torch.optim as optim


class TrainingPipeline:
    def __init__(self, model, train_loader, test_loader, device="cpu", optimizer_params=None):
        """
        Initialize the TrainingPipeline.

        :param model: The model to train and evaluate.
        :param train_loader: DataLoader for the training dataset.
        :param test_loader: DataLoader for the test dataset.
        :param device: The device to use (default: "cpu").
        :param optimizer_params: Dictionary of optimizer parameters (e.g., {"lr": 0.01, "weight_decay": 1e-5}).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
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

    def test(self):
        """
        Evaluate the model on the test dataset.
        """
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
