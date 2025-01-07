import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def save(self, path):
        """
        Save the model's state_dict to the specified path.
        """
        torch.save(self.state_dict(), path)

    def load(self, path, device="cpu"):
        """
        Load the model's state_dict from the specified path.
        """
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()  # Set model to evaluation mode

    def initialize_weights(self):
        """
        Optional: Initialize weights for the model. Override this if needed.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
