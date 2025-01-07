import argparse
from services.fisher_merging_service import FisherMergingService
from services.rebasin_merging_service import ReBasinMergingService
from utils.model_io import ModelIO
from utils.fisher_calculations import FisherCalculations
import torch
import yaml


def load_model(model_path):
    # Placeholder: Replace this with specific model architecture and loading logic
    model = ...  # Define the model architecture
    model = ModelIO.load_model(model, model_path)
    return model


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def fisher_merge(args):
    # Load models and configuration
    model1 = load_model(args.model1)
    model2 = load_model(args.model2)
    config = load_config(args.config)
    data_loader = ...  # Placeholder: Load your dataset here
    criterion = ...  # Placeholder: Define your loss function here

    # Perform merging using Fisher-Weighted Averaging
    fisher_service = FisherMergingService(model1, model2, data_loader, criterion)
    merged_model = fisher_service.merge()
    ModelIO.save_model(merged_model, args.output)
    print(f"Fisher-Weighted merged model saved to {args.output}")


def rebasin_merge(args):
    # Load models
    model1 = load_model(args.model1)
    model2 = load_model(args.model2)

    # Perform merging using Git Re-Basin
    rebasin_service = ReBasinMergingService(model1, model2)
    merged_model = rebasin_service.merge()
    ModelIO.save_model(merged_model, args.output)
    print(f"Re-Basin merged model saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Model merging CLI using Fisher-Weighted Averaging or Git Re-Basin.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-command for Fisher-Weighted Averaging
    fisher_parser = subparsers.add_parser("fisher", help="Merge models using Fisher-Weighted Averaging.")
    fisher_parser.add_argument("--model1", type=str, required=True, help="Path to the first model.")
    fisher_parser.add_argument("--model2", type=str, required=True, help="Path to the second model.")
    fisher_parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    fisher_parser.add_argument("--output", type=str, required=True, help="Path to save the merged model.")
    fisher_parser.set_defaults(func=fisher_merge)

    # Sub-command for Git Re-Basin
    rebasin_parser = subparsers.add_parser("rebasin", help="Merge models using Git Re-Basin.")
    rebasin_parser.add_argument("--model1", type=str, required=True, help="Path to the first model.")
    rebasin_parser.add_argument("--model2", type=str, required=True, help="Path to the second model.")
    rebasin_parser.add_argument("--output", type=str, required=True, help="Path to save the merged model.")
    rebasin_parser.set_defaults(func=rebasin_merge)

    # Parse arguments and invoke the appropriate command
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
