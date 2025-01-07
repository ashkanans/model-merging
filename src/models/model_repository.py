import os

class ModelRepository:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def save_model(self, model, model_name):
        """
        Save a model to the repository.
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.pth")
        model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_class, model_name, device="cpu"):
        """
        Load a model from the repository.
        :param model_class: The class of the model to load.
        :param model_name: The name of the saved model file (without extension).
        :param device: The device to load the model onto ('cpu' or 'cuda').
        :return: An instance of the loaded model.
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found")

        model = model_class()
        model.load(model_path, device)
        print(f"Model loaded from {model_path}")
        return model

    def list_models(self):
        """
        List all models stored in the repository.
        """
        models = [f for f in os.listdir(self.models_dir) if f.endswith(".pth")]
        return models
