"""
CNFUtils: Utility functions for Conditional Normalizing Flow (CNF) models in wildfire prediction.

This module provides a set of utility functions for working with CNF models,
including sampling, probability calculations, and various statistical measures.

Example usage:
    from src.models.cnf_utils import CNFUtils, create_fire_size_range
    import torch

    # Assuming 'model' is your trained CNF model and 'device' is your torch device
    cnf_utils = CNFUtils(model, device)

    # Create a range of fire sizes
    fire_size_range = create_fire_size_range()

    # Prepare context (can be numpy array, pandas DataFrame, or torch tensor)
    context = X_val[0]  # Example context

    # Sample from the model
    samples = cnf_utils.sample(context, num_samples=1000)

    # Compute PDF and CDF
    pdf_values = cnf_utils.pdf(fire_size_range, context)
    cdf_values = cnf_utils.cdf(fire_size_range, context)

    # Get quantiles
    median = cnf_utils.quantile(0.5, context, fire_size_range)
    percentile_95 = cnf_utils.quantile(0.95, context, fire_size_range)

    # Get probability of fire size above a threshold
    prob_large_fire = cnf_utils.probability_above_threshold(5000, context, fire_size_range)

    # Get prediction interval
    lower, upper = cnf_utils.prediction_interval(context, fire_size_range)

    # Compute residuals
    residuals = cnf_utils.residuals(y_val, X_val)

    # Plotting example (PDF)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(fire_size_range.cpu().numpy(), pdf_values)
    plt.xlabel('Fire Size (Acres)')
    plt.ylabel('Probability Density')
    plt.title('PDF of Fire Sizes')
    plt.show()

Note: This utility class assumes that the CNF model takes log-transformed fire sizes
      as input and outputs log-probabilities. The utility functions handle the
      necessary transformations between log-space and actual fire sizes.
"""

import torch
import numpy as np
import pandas as pd
from scipy import stats
import os
import wandb
import re
import matplotlib.pyplot as plt
import seaborn as sns

class CNFUtils:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @classmethod
    def from_wandb(cls, run_path, artifact_dir, model_class):
        """
        Load the latest model from wandb and create a CNFUtils instance.

        Args:
        run_path (str): Path to the wandb run (e.g., 'username/project/run-id')
        artifact_dir (str): Directory to save the artifact
        model_class (callable): Function that returns an instance of the model

        Returns:
        CNFUtils: An instance of CNFUtils with the loaded model
        """
        # Extract run_name from run_path
        run_name = run_path.split('/')[-1]
        print(f"Extracting artifacts for run: {run_name}")

        # Ensure the artifact directory exists
        os.makedirs(artifact_dir, exist_ok=True)

        # Create a subdirectory in artifact_dir based on the run name
        artifact_subdir = os.path.join(artifact_dir, run_name)
        os.makedirs(artifact_subdir, exist_ok=True)

        # Initialize wandb API
        api = wandb.Api()

        # Get the run
        run = api.run(run_path)

        # Get the latest model artifact
        artifacts = run.logged_artifacts()
        model_artifacts = [artifact for artifact in artifacts if artifact.type == 'model']
        
        if not model_artifacts:
            raise FileNotFoundError(f"No model artifacts found for run {run_path}")

        latest_model_artifact = model_artifacts[-1]  # Get the latest model artifact
        print(f"Downloading latest model artifact: {latest_model_artifact.name}")

        # Download the artifact
        artifact_dir = latest_model_artifact.download(root=artifact_subdir)

        # Find all model files
        model_files = [f for f in os.listdir(artifact_dir) if f.startswith('model_') and f.endswith('.pth')]
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {artifact_dir}")

        print(f"Available model files: {model_files}")

        # Extract epoch numbers and find the highest
        epoch_numbers = [int(re.search(r'model_(\d+)\.pth', f).group(1)) for f in model_files]
        max_epoch = max(epoch_numbers)
        latest_model_file = f'model_{max_epoch:04d}.pth'

        print(f"Loading the latest model: {latest_model_file}")

        # Load the model
        model_path = os.path.join(artifact_dir, latest_model_file)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class()  # Instantiate the model
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Create and return CNFUtils instance
        return cls(model, device)

    @classmethod
    def from_local(cls, model_dir, model_class):
        """
        Load the latest model from a local directory and create a CNFUtils instance.

        Args:
        model_dir (str): Path to the directory containing model files
        model_class (callable): Function that returns an instance of the model

        Returns:
        CNFUtils: An instance of CNFUtils with the loaded model
        """
        # Find all model files
        model_files = [f for f in os.listdir(model_dir) if f.startswith('model_') and f.endswith('.pth')]
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")

        print(f"Available model files: {model_files}")

        # Extract epoch numbers and find the highest
        epoch_numbers = [int(re.search(r'model_(\d+)\.pth', f).group(1)) for f in model_files]
        max_epoch = max(epoch_numbers)
        latest_model_file = f'model_{max_epoch:04d}.pth'

        print(f"Loading the latest model: {latest_model_file}")

        # Load the model
        model_path = os.path.join(model_dir, latest_model_file)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class()  # Instantiate the model
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        # Remove 'model.' prefix from keys if present
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        print('model set to eval. Ready to go!')

        return cls(model, device)


    def _prepare_context(self, context):
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context)
        elif isinstance(context, pd.DataFrame):
            context = torch.from_numpy(context.values)
        elif not isinstance(context, torch.Tensor):
            raise TypeError("Context must be a numpy array, pandas DataFrame, or torch Tensor")
        
        return context.float().to(self.device)

    def sample(self, context, num_samples=1):
        self.model.eval()
        with torch.no_grad():
            context = self._prepare_context(context)
            context_repeated = context.repeat(num_samples, 1)
            log_samples, _ = self.model.sample(num_samples, context=context_repeated)
        return log_samples

    def log_prob(self, log_fire_sizes, context):
        self.model.eval()
        with torch.no_grad():
            context = self._prepare_context(context)
            log_probs = self.model.log_prob(log_fire_sizes, context=context)
        return log_probs

    def pdf(self, log_fire_sizes, context):
        log_probs = self.log_prob(log_fire_sizes, context)
        return torch.exp(log_probs).cpu().numpy()

    def cdf(self, log_fire_sizes, context):
        pdf_values = self.pdf(log_fire_sizes, context)
        cdf_values = np.cumsum(pdf_values)
        return cdf_values / cdf_values[-1]  # Normalize to [0, 1]

    def inverse_cdf(self, probabilities, context, log_fire_size_range):
        cdf_values = self.cdf(log_fire_size_range, context)
        inverse_cdf = np.interp(probabilities, cdf_values, log_fire_size_range.cpu().numpy())
        return inverse_cdf

    def quantile(self, q, context, log_fire_size_range):
        return self.inverse_cdf([q], context, log_fire_size_range)[0]

    def probability_above_threshold(self, log_threshold, context, log_fire_size_range):
        cdf_values = self.cdf(log_fire_size_range, context)
        threshold_idx = np.searchsorted(log_fire_size_range.cpu().numpy(), log_threshold)
        return 1 - cdf_values[threshold_idx]

    def calculate_percentile(self, context, percentile=99.9):
        fire_size_range = create_fire_size_range().to(self.device)
        cdf_values = self.cdf(fire_size_range, context)
        
        # Ensure cdf_values is strictly increasing
        cdf_values, unique_indices = np.unique(cdf_values, return_index=True)
        fire_sizes = np.expm1(fire_size_range[unique_indices].cpu().numpy())
        
        percentile_value = np.interp(percentile/100, cdf_values, fire_sizes)
        return percentile_value

    def calculate_percentiles_batch(self, context_batch, percentile=99.9):
        fire_size_range = create_fire_size_range().to(self.device)
        
        fire_sizes_log_repeated = fire_size_range.repeat(len(context_batch), 1)
        context_repeated = context_batch.unsqueeze(1).repeat(1, fire_size_range.shape[0], 1)
        
        log_probs = self.log_prob(fire_sizes_log_repeated.view(-1, 1), context_repeated.view(-1, context_batch.shape[1]))
        log_probs = log_probs.view(len(context_batch), -1)
        
        fire_sizes = np.expm1(fire_size_range.cpu().numpy()).flatten()
        pdf_values = torch.exp(log_probs).cpu().numpy()
        cdf_values = np.cumsum(pdf_values, axis=1)
        cdf_values /= cdf_values[:, -1:]
        
        percentile_values = []
        for cdf in cdf_values:
            # Ensure cdf is strictly increasing
            cdf, unique_indices = np.unique(cdf, return_index=True)
            unique_fire_sizes = fire_sizes[unique_indices]
            percentile_value = np.interp(percentile/100, cdf, unique_fire_sizes)
            percentile_values.append(percentile_value)
        
        return np.array(percentile_values)
    def plot_pdf_cdf_for_context(self, context_point, percentile=0.99, num_steps=1000):
        """
        Plot the PDF and CDF of wildfire sizes for a given input context.

        Args:
        context_point (torch.Tensor): Context point tensor.
        percentile (float): Percentile to highlight on the plot.
        num_steps (int): Number of steps in the fire size range.

        Returns:
        None
        """
        fire_sizes_log = torch.linspace(0, 10, steps=num_steps).unsqueeze(1).to(self.device)  # Create a range of log-scaled sizes
        context_repeated = context_point.repeat(fire_sizes_log.size(0), 1)

        self.model.eval()
        with torch.no_grad():
            log_probs = self.model.log_prob(fire_sizes_log, context=context_repeated)

        fire_sizes = np.expm1(fire_sizes_log.cpu().numpy())  # Inverse of log1p using numpy
        pdf_values = torch.exp(log_probs.cpu()).numpy()  # Convert log probabilities to probabilities

        cdf_values = np.cumsum(pdf_values)
        cdf_values /= cdf_values[-1]  # Normalize to [0, 1]
        percentile_value = fire_sizes[np.searchsorted(cdf_values, percentile)]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))

        # Plot PDF
        ax1.plot(fire_sizes, pdf_values, label='PDF', color='blue')
        ax1.axvline(x=percentile_value, color='red', linestyle='--', label=f'{percentile*100:.1f} Percentile: {percentile_value[0]:.2f} Acres')
        ax1.set_xlabel('Wildfire Size (Acres)')
        ax1.set_ylabel('Probability Density')
        ax1.set_yscale('log')
        ax1.set_ylim(1e-3, ax1.get_ylim()[1])  # Set the minimum y limit to 10^-3
        ax1.set_title('PDF of Wildfire Sizes for Given Input Condition')
        ax1.legend()
        ax1.grid(True)

        # Plot CDF
        ax2.plot(fire_sizes, cdf_values, label='CDF', color='blue')
        ax2.axvline(x=percentile_value, color='red', linestyle='--', label=f'{percentile*100:.1f} Percentile: {percentile_value[0]:.2f} Acres')
        ax2.set_xlabel('Wildfire Size (Acres)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('CDF of Wildfire Sizes for Given Input Condition')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


    def plot_distribution_comparison(self, X_val, y_val, y_train, num_samples=3000):
        """
        Evaluate the model, generate samples, and plot histograms comparing
        the generated samples with the validation and training data.

        Args:
        X_val (torch.Tensor): Validation set features.
        y_val (torch.Tensor): Validation set targets.
        y_train (torch.Tensor): Training set targets.
        num_samples (int): Number of samples to generate.

        Returns:
        None
        """
        # Evaluate the model and generate samples
        self.model.eval()
        with torch.no_grad():
            context_samples = X_val[:num_samples]
            context_samples = torch.tensor(context_samples, dtype=torch.float32).to(self.device)
            print(f"Context samples shape: {context_samples.shape}")
            samples, _ = self.model.sample(len(context_samples), context=context_samples)
            samples = torch.clamp(samples, min=4)

        samples = np.squeeze(samples.cpu().numpy())
        print(f"Samples shape: {samples.shape}")
        print(f"Validation targets shape: {y_val.cpu().numpy().shape}")

        # Plot histograms
        plt.figure(figsize=(12, 8))
        plt.hist(y_val.cpu().numpy(), bins=50, density=True, alpha=0.6, color='g', label='Validation Data Size')
        plt.hist(y_train, bins=50, density=True, alpha=0.6, color='r', label='Original Data Size')
        plt.hist(samples, bins=50, density=True, alpha=0.6, color='b', label='Predicted Data Size')

        plt.title('Comparison of Learned Distribution to Original Data')
        plt.xlabel('Wildfire Size Log-Acres')
        plt.ylabel('Density')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_qq_residuals_intervals(self, X_val, y_val, num_samples=3000, quantiles=np.linspace(0, 99, 50)):
        """
        Generate Q-Q plot, histogram of residuals, and prediction intervals.

        Args:
        X_val (torch.Tensor): Validation set features.
        y_val (torch.Tensor): Validation set targets.
        num_samples (int): Number of samples to generate.
        quantiles (np.ndarray): Quantiles to calculate for the Q-Q plot.

        Returns:
        None
        """
        # Transform y_val back from log scale
        y_val_exp = torch.expm1(y_val)

        # Generate predictions (samples) from the model
        self.model.eval()
        with torch.no_grad():
            context_samples = X_val[:num_samples]
            samples, _ = self.model.sample(len(context_samples), context=context_samples)

        # Transform samples back from log scale
        samples_exp = torch.expm1(samples)

        # Convert to numpy arrays for easier handling
        y_val_np = y_val_exp.cpu().numpy()
        samples_np = samples_exp.cpu().numpy()

        # Function to calculate quantiles
        def calculate_quantiles(data, quantiles):
            return np.percentile(data, quantiles)

        # Calculate quantiles for the actual data and predicted samples
        actual_quantiles = calculate_quantiles(y_val_np, quantiles)
        predicted_quantiles = calculate_quantiles(samples_np, quantiles)

        # Plot Q-Q plot
        plt.figure(figsize=(10, 6))
        plt.plot(actual_quantiles, predicted_quantiles, 'o')
        plt.plot([min(actual_quantiles), max(actual_quantiles)], [min(actual_quantiles), max(actual_quantiles)], 'r--')
        plt.xlabel('Actual Quantiles')
        plt.ylabel('Predicted Quantiles')
        plt.title('Q-Q Plot')
        plt.grid(True)
        plt.show()

def create_fire_size_range(min_size=4, max_size=1e6, num_steps=1000):
    return torch.linspace(np.log1p(min_size), np.log1p(max_size), num_steps).unsqueeze(1)