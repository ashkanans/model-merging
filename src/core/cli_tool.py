from divergent_models_pipeline import divergent_models_pipeline
from noisy_training_pipeline import noisy_training_pipeline


class CLITool:
    def __init__(self, args):
        self.args = args

    def _register_commands(self):
        # Register "train" command
        train_parser = self.subparsers.add_parser("train", help="Train a model")
        train_parser.add_argument("--dataset", required=True, help="Dataset to use (e.g., mnist)")
        train_parser.add_argument("--model", required=True, help="Model architecture (e.g., mlp)")
        train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

        # Register "merge" command
        merge_parser = self.subparsers.add_parser("merge", help="Merge models using various methods")
        merge_parser.add_argument("--model1", required=True, help="Path to the first model")
        merge_parser.add_argument("--model2", required=True, help="Path to the second model")
        merge_parser.add_argument("--method", choices=["isotropic", "fisher"], default="isotropic",
                                  help="Merging method")

        # Register "visualize" command
        vis_parser = self.subparsers.add_parser("visualize", help="Visualize Fisher Information")
        vis_parser.add_argument("--model", required=True, help="Path to the model")
        vis_parser.add_argument("--dataset", required=True, help="Dataset to use for visualization")

        # Register "experiment" command
        exp_parser = self.subparsers.add_parser("experiment", help="Run scaling experiment")
        exp_parser.add_argument("--model1", required=True, help="Path to the first model")
        exp_parser.add_argument("--model2", required=True, help="Path to the second model")

        # Register "pipeline" command
        pipe_parser = self.subparsers.add_parser("pipeline", help="Run a predefined pipeline")
        pipe_parser.add_argument("--type", choices=["divergent", "generalization", "noisy"], required=True,
                                 help="Pipeline type")

    def run(self):
        args = self.parser.parse_args()

        if args.command == "train":
            self._run_train(args)
        elif args.command == "merge":
            self._run_merge(args)
        elif args.command == "visualize":
            self._run_visualize(args)
        elif args.command == "experiment":
            self._run_experiment(args)
        elif args.command == "pipeline":
            self._run_pipeline(args)
        else:
            self.parser.print_help()

    def _run_train(self, args):
        print(f"Training model {args.model} on dataset {args.dataset} for {args.epochs} epochs")
        # Instantiate and run the training pipeline here

    def _run_merge(self, args):
        print(f"Merging models {args.model1} and {args.model2} using {args.method} method")
        # Add logic to load models and call the appropriate merge method

    def _run_visualize(self, args):
        print(f"Visualizing Fisher Information for model {args.model} with dataset {args.dataset}")
        # Add logic to visualize Fisher information

    def _run_experiment(self, args):
        print(f"Running Fisher Scaling experiment with models {args.model1} and {args.model2}")
        # Add logic for scaling experiment

    def _run_pipeline(self, args):
        if args.type == "divergent":
            print("Running Divergent Models Pipeline")
            divergent_models_pipeline()
        elif args.type == "generalization":
            print("Running Generalization Pipeline")
            # Add logic to call the generalization pipeline
        elif args.type == "noisy":
            print("Running Noisy Training Pipeline")
            noisy_training_pipeline()
