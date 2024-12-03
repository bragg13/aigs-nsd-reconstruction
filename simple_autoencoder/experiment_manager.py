"""
Experiment runner for managing multiple training configurations.
Handles parameter combinations and organizing results systematically.
"""
import itertools
from pathlib import Path
from datetime import datetime
import yaml
from omegaconf import OmegaConf, DictConfig
import copy
from typing import List, Dict, Any
import json

class ExperimentManager:
    """Manages multiple experiments with different parameter combinations."""

    def __init__(self, base_config_path: str, experiment_base_dir: str = "experiments"):
        """
        Initialize the experiment manager.

        Args:
            base_config_path: Path to the base configuration YAML file
            experiment_base_dir: Base directory for storing experiment results
        """
        # Load base configuration
        self.base_config = OmegaConf.load(base_config_path)
        self.base_dir = Path(experiment_base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def generate_parameter_grid(self, param_ranges: Dict[str, List[Any]]) -> List[DictConfig]:
        """
        Generate all combinations of parameters.

        Args:
            param_ranges: Dictionary of parameters and their possible values
                Example: {
                    'learning_rate': [1e-3, 1e-4],
                    'batch_size': [32, 64],
                    'latent_dim': [128, 256]
                }

        Returns:
            List of configurations, each with a unique parameter combination
        """
        # Get all possible combinations
        keys = param_ranges.keys()
        values = param_ranges.values()
        combinations = list(itertools.product(*values))

        # Create a config for each combination
        configs = []
        for combo in combinations:
            # Create a fresh copy of base config
            config = copy.deepcopy(self.base_config)

            # Update with current parameter combination
            for key, value in zip(keys, combo):
                OmegaConf.update(config, key, value, merge=True)

            configs.append(config)

        return configs

    def create_experiment_dir(self, config: DictConfig) -> Path:
        """
        Create a unique directory for this experiment configuration.

        Args:
            config: Configuration for this experiment

        Returns:
            Path to the experiment directory
        """
        # Create timestamp for unique identification
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a descriptive name using key parameters
        param_str = "_".join([
            f"{k}={v}" for k, v in config.items()
            if k in ['learning_rate', 'batch_size', 'latent_dim', 'sparsity']
        ])

        # Create experiment directory
        exp_dir = self.base_dir / f"{timestamp}_{param_str}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(exp_dir / "config.yaml", "w") as f:
            OmegaConf.save(config=config, f=f)

        return exp_dir

    def run_experiments(self, param_ranges: Dict[str, List[Any]], train_fn) -> None:
        """
        Run all experiments with different parameter combinations.

        Args:
            param_ranges: Dictionary of parameters and their possible values
            train_fn: Training function that takes a config as input
        """
        configs = self.generate_parameter_grid(param_ranges)

        print(f"Running {len(configs)} experiments...")
        for i, config in enumerate(configs, 1):
            print(f"\nExperiment {i}/{len(configs)}")
            print("Parameters:", json.dumps(OmegaConf.to_container(config), indent=2))

            # Create directory for this experiment
            exp_dir = self.create_experiment_dir(config)

            # Update config with experiment directory
            config.results_folder = str(exp_dir)

            # Run the experiment
            try:
                train_fn(config)
                print(f"Experiment completed successfully. Results saved in: {exp_dir}")
            except Exception as e:
                print(f"Experiment failed with error: {str(e)}")
                # Optionally log the error to a file in the experiment directory
                with open(exp_dir / "error.log", "w") as f:
                    f.write(f"Error: {str(e)}")

# Example usage:
if __name__ == "__main__":
    # Define your base configuration file (config.yaml):
    """
    learning_rate: 1e-3
    batch_size: 32
    latent_dim: 256
    num_epochs: 100
    sparsity: 0.9
    l1: 0.01
    """

    # Import your training function
    from train import train_and_evaluate

    # Initialize experiment manager
    manager = ExperimentManager("config.yaml")

    # Define parameter ranges to explore
    param_ranges = {
        'learning_rate': [1e-3, 1e-4],
        'subject': [1],
        'batch_size': [256],
        'latent_dim': [16, 32, 64],
        'sparsity': [0.8, 0.9],
        'l1': [0.01, 0.001],
        # 'ds': ['mnist', 'cifar10']
    }

    # Run all experiments
    manager.run_experiments(param_ranges, train_and_evaluate)
