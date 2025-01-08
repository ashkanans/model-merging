import torch
import os


class ModelIO:
    # Default folder for saving and loading models
    MODEL_DIR = "saved_models"

    @staticmethod
    def save_model(model, filename):
        """
        Save the model's state_dict to the specified path inside the default model directory.

        :param model: The model instance to save.
        :param filename: The filename for saving the model.
        """
        # Ensure the default directory exists
        os.makedirs(ModelIO.MODEL_DIR, exist_ok=True)

        # Build the full path
        path = os.path.join(ModelIO.MODEL_DIR, filename)

        # Save the model state_dict
        torch.save(model.state_dict(), path)
        print(f"Model saved at: {path}")

    @staticmethod
    def load_model(model, filename, device="cpu"):
        """
        Load the model's state_dict from the specified file in the default model directory.

        :param model: The model instance to load weights into.
        :param filename: The filename of the saved model weights.
        :param device: The device to load the model on ('cpu' or 'cuda').
        :return: The model with loaded weights.
        """
        # Build the full path
        path = os.path.join(ModelIO.MODEL_DIR, filename)

        # Check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load weights-only mode for safer deserialization
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"Model loaded from: {path}")
        return model
