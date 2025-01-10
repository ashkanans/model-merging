from torch import nn

from src.core.fisher_scaling_experiment import fisher_scaling_experiment
from src.core.fisher_weighted_averaging import FisherWeightedAveraging
from src.core.isotropic_merging import NaiveMerging
from src.core.output_ensemble_merging import OutputEnsembling
from src.core.training_pipeline import TrainingPipeline
from src.models.cifar10_models import CIFAR10ResNet, CIFAR10VGG
from src.models.mnist_model import MNISTMLP, MNISTCNN
from src.utils.dataset_loader import DatasetLoader
from src.utils.model_io import ModelIO
from src.utils.noisy_training_utils import noise_augmentation, add_label_noise
from src.utils.utiliy import validate_model, plot_layer_distributions_with_stats, draw_comparison_chart


class CLITool:
    """
    CLI tool for managing neural network training, merging, and validation.
    """
    def __init__(self, args):
        self.train_loader = None
        self.test_loader = None
        self.args = args

        self.ablation = args.ablation
        self.abl_scaling_factors = args.scaling_factors if self.ablation else []
        self.abl_layers = args.layers if self.ablation else []
        self.abl_weighting = "weighting" in args.ablation if self.ablation else False

    def run(self):
        """
        Executes the requested commands in sequence.
        """
        train_loader, test_loader, model1, model2 = self._initialize_models()
        self.train_loader, self.test_loader = train_loader, test_loader

        merged_models = {}
        validation_results = None

        for command in self.args.command:
            if command == "train":
                if self.args.noisy_models:
                    print("Training with noise enabled for specified models...")
                    self._run_noisy_training(model1, model2, train_loader, test_loader, self.args.noisy_models)
                else:
                    self._run_training(model1, model2, train_loader, test_loader)
            elif command == "merge":
                merged_models = self._run_merging(model1, model2, train_loader)
            elif command == "validate":
                validation_results = self._run_validation(model1, model2, merged_models, test_loader)
            elif command == "visualize":
                # Enable default visualization settings
                self._run_visualization(validation_results, {
                    "model1": model1,
                    "model2": model2,
                    "isotropic": merged_models.get("isotropic"),
                    "fisher": merged_models.get("fisher"),
                    "ensemble": merged_models.get("ensemble")
                })


    def _initialize_models(self):
        """
        Initializes models and data loaders based on the selected dataset and model type.
        If a list of datasets is provided, handles divergent models (e.g., MNIST and Fashion-MNIST).
        """
        train_loader, test_loader = None, None
        model1, model2 = None, None

        if isinstance(self.args.dataset, list) and len(self.args.dataset) == 2:
            if "mnist" in self.args.dataset and "fashion-mnist" in self.args.dataset:
                print("Initializing divergent datasets: MNIST and Fashion-MNIST")
                mnist_train_loader, mnist_test_loader = DatasetLoader.load_mnist()
                fashion_train_loader, fashion_test_loader = DatasetLoader.load_fashion_mnist()
                train_loader = {"mnist": mnist_train_loader, "fashion-mnist": fashion_train_loader}
                test_loader = {"mnist": mnist_test_loader, "fashion-mnist": fashion_test_loader}
                model1, model2 = MNISTMLP(), MNISTMLP()  # Assume MLP for both datasets
            else:
                raise ValueError(f"Unsupported dataset combination: {self.args.dataset}")
        elif "mnist" in self.args.dataset:
            train_loader, test_loader = DatasetLoader.load_mnist()
            model1, model2 = self._get_mnist_models()
        elif "fashion-mnist" in self.args.dataset:
            train_loader, test_loader = DatasetLoader.load_fashion_mnist()
            model1, model2 = self._get_mnist_models()
        elif "cifar10" in self.args.dataset:
            train_loader, test_loader = DatasetLoader.load_cifar10()
            model1, model2 = self._get_cifar10_models()
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        if model1 is None or model2 is None:
            raise ValueError(f"Invalid model type: {self.args.model}")

        return train_loader, test_loader, model1, model2

    def _get_mnist_models(self):
        if self.args.model == "mlp":
            return MNISTMLP(), MNISTMLP()
        elif self.args.model == "cnn":
            return MNISTCNN(), MNISTCNN()
        return None, None

    def _get_cifar10_models(self):
        if self.args.model == "resnet":
            return CIFAR10ResNet(), CIFAR10ResNet()
        elif self.args.model == "vgg":
            return CIFAR10VGG(), CIFAR10VGG()
        return None, None

    def _run_training(self, model1, model2, train_loader, test_loader):
        """
        Trains and saves two models. Supports training on divergent datasets dynamically by using keys from the DataLoader dictionary.

        :param model1: First model to be trained.
        :param model2: Second model to be trained.
        :param train_loader: Training DataLoader(s). If divergent datasets, this is a dictionary.
        :param test_loader: Testing DataLoader(s). If divergent datasets, this is a dictionary.
        """
        if isinstance(train_loader, dict) and isinstance(test_loader, dict):
            # Divergent training setup
            dataset_keys = list(train_loader.keys())
            optimizer1_params = {"lr": 0.01, "weight_decay": 1e-5}
            optimizer2_params = {"lr": 0.001, "weight_decay": 1e-4}

            print(f"Training Model 1 on {dataset_keys[0]}...")
            print(f"Hyperparameters are {optimizer1_params}...")
            self._train_and_save_model(model1, train_loader[dataset_keys[0]], test_loader[dataset_keys[0]],
                                       dataset_keys[0], dataset_keys[0], optimizer1_params)

            print(f"Training Model 2 on {dataset_keys[1]}...")
            print(f"Hyperparameters are {optimizer2_params}...")
            self._train_and_save_model(model2, train_loader[dataset_keys[1]], test_loader[dataset_keys[1]],
                                       dataset_keys[1], dataset_keys[1], optimizer2_params)
        else:
            # Standard training setup
            print("Training Model 1...")
            self._train_and_save_model(model1, train_loader, test_loader, "1", self.args.dataset[0])

            print("Training Model 2...")
            self._train_and_save_model(model2, train_loader, test_loader, "2", self.args.dataset[0])

    def _train_and_save_model(self, model, train_loader, test_loader, suffix, dataset_name, optimizer_params=None):
        """
        Trains and saves a model. Dynamically adjusts for single or multiple datasets.

        :param model: The model to train and save.
        :param train_loader: Training DataLoader.
        :param test_loader: Testing DataLoader.
        :param suffix: Suffix for the saved model file name (e.g., "mnist" or "1").
        :param optimizer_params: Optional parameters for the optimizer.
        """
        # Initialize the training pipeline
        trainer = TrainingPipeline(model, train_loader, test_loader, optimizer_params=optimizer_params)

        # Train and test the model
        trainer.train(self.args.epochs)
        trainer.test()

        # Save the model
        save_path = f"{dataset_name}_{self.args.model}_{suffix}.pth"
        ModelIO.save_model(model, save_path)
        print(f"Model saved at: {save_path}")

    def _run_merging(self, model1, model2, train_loader):
        """
        Merges two models using one or all of the available merging techniques.
        Supports divergent datasets when train_loader is a dictionary and self.args.dataset is a list.

        :param model1: The first trained model.
        :param model2: The second trained model.
        :param train_loader: DataLoader or dictionary of DataLoaders for computing Fisher information.
        :return: Dictionary containing all merged models.
        """
        merged_models = {}
        merge_types = ["fisher", "isotropic", "ensemble"]

        # Use all merging methods by default if no specific type is specified
        if not self.args.merge_type:
            print("No specific merge_type specified. Using all merging methods by default.")
            active_merge_types = merge_types
        else:
            active_merge_types = [self.args.merge_type]

        if isinstance(train_loader, dict) and isinstance(self.args.dataset, list):
            dataset_keys = list(train_loader.keys())
            if len(self.args.dataset) != len(dataset_keys):
                raise ValueError(
                    "Mismatch between dataset keys and provided datasets in train_loader and args.dataset.")

            for merge_type in active_merge_types:
                if merge_type == "fisher":
                    print(f"Merging using Fisher-weighted averaging for {dataset_keys[0]} and {dataset_keys[1]}...")
                    merged_models["fisher"] = self._merge_fisher(model1, model2, train_loader)
                    ModelIO.save_model(
                        merged_models["fisher"],
                        f"{dataset_keys[0]}_{dataset_keys[1]}_{self.args.model}_fisher_merged.pth"
                    )
                    print(
                        f"Fisher merged model saved: {dataset_keys[0]}_{dataset_keys[1]}_{self.args.model}_fisher_merged.pth")
                elif merge_type == "isotropic":
                    print(f"Merging using isotropic averaging for {dataset_keys[0]} and {dataset_keys[1]}...")
                    merged_models["isotropic"] = NaiveMerging.merge_models(model1, model2)
                    ModelIO.save_model(
                        merged_models["isotropic"],
                        f"{dataset_keys[0]}_{dataset_keys[1]}_{self.args.model}_isotropic_merged.pth"
                    )
                    print(
                        f"Isotropic merged model saved: {dataset_keys[0]}_{dataset_keys[1]}_{self.args.model}_isotropic_merged.pth")
                elif merge_type == "ensemble":
                    print(f"Merging using output ensembling for {dataset_keys[0]} and {dataset_keys[1]}...")
                    merged_models["ensemble"] = OutputEnsembling.merge_models(model1, model2)
                    ModelIO.save_model(
                        merged_models["ensemble"],
                        f"{dataset_keys[0]}_{dataset_keys[1]}_{self.args.model}_ensemble_merged.pth"
                    )
                    print(
                        f"Ensemble merged model saved: {dataset_keys[0]}_{dataset_keys[1]}_{self.args.model}_ensemble_merged.pth")
        else:
            # Single dataset handling
            for merge_type in active_merge_types:
                if merge_type == "fisher":
                    print("Merging using Fisher-weighted averaging...")
                    merged_models["fisher"] = self._merge_fisher(model1, model2, train_loader)
                    ModelIO.save_model(
                        merged_models["fisher"],
                        f"{self.args.dataset[0]}_{self.args.model}_fisher_merged.pth"
                    )
                    print(f"Fisher merged model saved: {self.args.dataset}_{self.args.model}_fisher_merged.pth")
                elif merge_type == "isotropic":
                    print("Merging using isotropic averaging...")
                    merged_models["isotropic"] = NaiveMerging.merge_models(model1, model2)
                    ModelIO.save_model(
                        merged_models["isotropic"],
                        f"{self.args.dataset[0]}_{self.args.model}_isotropic_merged.pth"
                    )
                    print(f"Isotropic merged model saved: {self.args.dataset}_{self.args.model}_isotropic_merged.pth")
                elif merge_type == "ensemble":
                    print("Merging using output ensembling...")
                    merged_models["ensemble"] = OutputEnsembling.merge_models(model1, model2)
                    ModelIO.save_model(
                        merged_models["ensemble"],
                        f"{self.args.dataset[0]}_{self.args.model}_ensemble_merged.pth"
                    )
                    print(f"Ensemble merged model saved: {self.args.dataset}_{self.args.model}_ensemble_merged.pth")

        return merged_models

    def _merge_fisher(self, model1, model2, train_loader):
        """
        Merges two models using Fisher-weighted averaging. Supports divergent datasets dynamically.

        :param model1: The first trained model.
        :param model2: The second trained model.
        :param train_loader: Training DataLoader(s). If divergent datasets, this is a dictionary.
        :return: The Fisher-merged model.
        """
        criterion = nn.CrossEntropyLoss()

        if isinstance(train_loader, dict):
            # Divergent datasets
            dataset_keys = list(train_loader.keys())

            print(f"Computing Fisher Information for Model 1 on {dataset_keys[0]}...")
            fisher1 = FisherWeightedAveraging.compute_fisher_information(model1, train_loader[dataset_keys[0]],
                                                                         criterion)

            print(f"Computing Fisher Information for Model 2 on {dataset_keys[1]}...")
            fisher2 = FisherWeightedAveraging.compute_fisher_information(model2, train_loader[dataset_keys[1]],
                                                                         criterion)
            scaling_factor = 1
            if self.ablation and len(self.abl_scaling_factors) == 1:
                scaling_factor = self.abl_scaling_factors[0]
            if self.ablation and len(self.abl_scaling_factors) > 1:
                scaling_factor = fisher_scaling_experiment(self.test_loader, model1, model2, fisher1, fisher2,
                                                           criterion, self.abl_scaling_factors)

            print(
                f"Merging models trained on {dataset_keys[0]} and {dataset_keys[1]} using Fisher-weighted averaging...")
            merged_model = FisherWeightedAveraging.merge_models(model1, model2, fisher1, fisher2,
                                                                alpha=self.args.alpha,
                                                                fisher_scaling=scaling_factor,
                                                                layer_wise_fisher=self.abl_layers,
                                                                uniform_weights=self.abl_weighting)

            save_path = f"{dataset_keys[0]}_{dataset_keys[1]}_{self.args.model}_fisher_merged.pth"
        else:
            # Single dataset
            print("Computing Fisher Information for Model 1...")
            fisher1 = FisherWeightedAveraging.compute_fisher_information(model1, train_loader, criterion)

            print("Computing Fisher Information for Model 2...")
            fisher2 = FisherWeightedAveraging.compute_fisher_information(model2, train_loader, criterion)

            scaling_factor = 1
            if self.ablation and len(self.abl_scaling_factors) == 1:
                scaling_factor = self.abl_scaling_factors[0]
            if self.ablation and len(self.abl_scaling_factors) > 1:
                scaling_factor = fisher_scaling_experiment(self.test_loader, model1, model2, fisher1, fisher2,
                                                           criterion, scaling_factors=self.abl_scaling_factors,
                                                           weighting=self.abl_weighting, layers=self.abl_layers)

            print("Merging models using Fisher-weighted averaging...")
            merged_model = FisherWeightedAveraging.merge_models(model1, model2, fisher1, fisher2,
                                                                alpha=self.args.alpha,
                                                                fisher_scaling=scaling_factor,
                                                                layer_wise_fisher=self.abl_layers,
                                                                uniform_weights=self.abl_weighting)

            save_path = f"{self.args.dataset[0]}_{self.args.model}_fisher_merged.pth"

        ModelIO.save_model(merged_model, save_path)
        print(f"Fisher-merged model saved at: {save_path}")
        return merged_model

    def _run_validation(self, model1, model2, merged_models, test_loader):
        """
        Validates the trained and merged models.

        :param model1: The first trained model.
        :param model2: The second trained model.
        :param merged_models: Dictionary containing merged models (e.g., isotropic, ensemble, fisher).
        :param test_loader: DataLoader for testing the models.
        :return: Dictionary of validation results for all models.
        """
        criterion = nn.CrossEntropyLoss()
        results = {}

        print("Validating Model 1...")
        results['model1'] = validate_model(model1, test_loader, criterion)

        print("Validating Model 2...")
        results['model2'] = validate_model(model2, test_loader, criterion)

        if merged_models:
            for merge_type, model in merged_models.items():
                print(f"Validating {merge_type} Merged Model...")
                results[merge_type] = validate_model(model, test_loader, criterion)

        print("Validation results:")
        for model_name, metrics in results.items():
            print(f"{model_name}: {metrics}")

        return results

    def _run_visualization(self, validation_results, models):
        """
        Runs visualization tasks for the models.
        If no specific visualization flag is provided, both visualizations are performed by default.

        :param validation_results: Dictionary of validation results for all models.
        :param models: Dictionary containing models for visualization (e.g., {"isotropic": isotropic_model, ...}).
        """
        # Default both visualizations if no specific options are given
        if not (self.args.comparison_chart or self.args.layer_dist):
            self.args.comparison_chart = True
            self.args.layer_dist = True

        if self.args.comparison_chart:
            print("Generating comparison chart...")
            draw_comparison_chart(
                isotropic_results=validation_results.get('isotropic', {}),
                ensemble_results=validation_results.get('ensemble', {}),
                fisher_results=validation_results.get('fisher', {})
            )

        if self.args.layer_dist:
            print("Plotting layer distributions...")
            model_list = [models.get('model1'), models.get('model2'), models.get('isotropic'), models.get('fisher')]
            model_labels = ["Model 1", "Model 2", "Isotropic Merged", "Fisher Merged"]

            if self.args.model == "mlp":
                layer_names = ["fc1.weight", "fc2.weight", "fc3.weight"]
            elif self.args.model == "cnn":
                layer_names = ["conv1.weight", "conv2.weight", "fc1.weight", "fc2.weight"]
            else:
                print("Layer distribution visualization is not supported for this model type.")
                return

            plot_layer_distributions_with_stats(
                models=[m for m in model_list if m is not None],  # Filter out None models
                layer_names=layer_names,
                model_labels=model_labels[:len(model_list)]  # Adjust labels based on available models
            )

    def _run_noisy_training(self, model1, model2, train_loader, test_loader, noisy_models):
        """
        Trains models with optional noise injection based on user selection.

        :param model1: The first model.
        :param model2: The second model.
        :param train_loader: DataLoader for training data.
        :param test_loader: DataLoader for test data.
        :param noisy_models: List of models to train with noise (e.g., ["model1"], ["model2"], or both).
        """
        dataset_names = {}
        if len(self.args.dataset) == 2:
            dataset_names[0], dataset_names[1] = self.args.dataset[0], self.args.dataset[1]
        if len(self.args.dataset) == 1:
            dataset_names[0], dataset_names[1] = self.args.dataset[0], self.args.dataset[0]

        if "model1" in noisy_models:
            print("Training Model 1 with noise...")
            noisy_train_loader1 = [
                (noise_augmentation(data), add_label_noise(target, num_classes=10)) for data, target in train_loader
            ]
            self._train_and_save_model(model1, noisy_train_loader1, test_loader, "noisy_1",
                                       dataset_name=dataset_names[0])
        else:
            print("Training Model 1 without noise...")
            self._train_and_save_model(model1, train_loader, test_loader, "1", dataset_name=dataset_names[0])

        if "model2" in noisy_models:
            print("Training Model 2 with noise...")
            noisy_train_loader2 = [
                (noise_augmentation(data), add_label_noise(target, num_classes=10)) for data, target in train_loader
            ]
            self._train_and_save_model(model2, noisy_train_loader2, test_loader, "noisy_2",
                                       dataset_name=dataset_names[1])
        else:
            print("Training Model 2 without noise...")
            self._train_and_save_model(model2, train_loader, test_loader, "2", dataset_name=dataset_names[1])
