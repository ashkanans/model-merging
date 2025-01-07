import torch


class ModelIO:
    @staticmethod
    def save_model(model, path):
        """
        Save the model's state_dict to the specified path.
        """
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_model(model, path, device="cpu"):
        """
        Load the model's state_dict from the specified path.

        :param model: The model instance to load weights into.
        :param path: The file path of the saved model weights.
        :param device: The device to load the model on ('cpu' or 'cuda').
        :return: The model with loaded weights.
        """
        # Load weights-only mode for safer deserialization
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device)
        return model
