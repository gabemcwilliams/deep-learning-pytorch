"""
experiment.py

This module defines the ExperimentManager class, which centralizes logging, tracking,
and model export functionality for ML experiments using MLflow. It supports:
- Stage-based logging with Loguru
- Metric and artifact tracking with MLflow
- Model exporting in both Torch and ONNX formats
- Runtime diagnostics and environment validation

Intended for use in training scripts that require repeatable experiment management,
deployment readiness, and production-grade observability.

Example usage:
    manager = ExperimentManager("my_experiment")
    manager.log_params({...})
    manager.log_metrics({...})
    manager.end_run()
"""


# --- Standard Library ---
import os
import time
import datetime as dt
from pathlib import Path
import subprocess
import functools
import io
import re

# --- Third-Party Libraries ---
from dotenv import load_dotenv

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

import onnx

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from colorama import Fore, Style
from loguru import logger

import json
import tempfile


# --- ExperimentManager Class ---
class ExperimentManager:
    """
    Manages experiments with MLflow, logging, and ONNX model export.
    """

    def __init__(self,
                 experiment_name: str,
                 log_dir: str = "/mnt/mls/logs",
                 tls_enabled: bool = True,
                 requests_ca_bundle: str = '/mnt/mls/certs/mlflow/ca.crt'
                 ):
        """
        Initializes the ExperimentManager with logging, MLflow, and TLS setup.

        :param experiment_name: Name of the experiment.
        :param log_dir: Directory for storing logs.
        :param tls_enabled: Flag to enable TLS.
        :param requests_ca_bundle: Path to the certificate for secure connections.
        """

        self.input_example = None
        self.signature = None
        self.experiment_name = experiment_name
        self.timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y_%m_%d_%H%M%S")
        self.log_dir = Path(log_dir) / experiment_name
        self.log_file = self.log_dir / f"train_{experiment_name}_{self.timestamp}.log"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # --- MLflow Setup ---
        self.__setup_mlflow(tls_enabled, requests_ca_bundle)

        # --- Loguru Setup ---
        self.__setup_logger()

        self.stage = 'setup'

    # +-----------------------------+
    # |           Loguru            |
    # +-----------------------------+
    def __setup_logger(self):
        """
        Configures the Loguru logger with run details and log file settings.
        """
        logger.remove()
        self.logger = logger.bind(run_id=self.__run_id, timestamp=self.timestamp)
        self.logger.add(
            str(self.log_file),
            format="{extra[timestamp]} | {level} | {extra[run_id]} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="00:00",
            retention="14 days",
            compression="zip",
            enqueue=True,
            backtrace=True,
            diagnose=False,
            mode="a",
            filter=lambda record: record["extra"].get("run_id") == self.__run_id
        )
        self.logger.info("ExperimentManager initialized.")

    def log_time(self, stage: str = "unspecified"):
        """
        Decorator for logging the execution time of functions.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.logger.opt(depth=1).info(f"[{stage.upper()}] Starting {func.__name__}...")
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                self.logger.opt(depth=1).info(
                    f"[{stage.upper()}] {func.__name__} took {end_time - start_time:.4f} seconds")
                return result

            return wrapper

        return decorator

    # +-----------------------------+
    # |           MlFlow            |
    # +-----------------------------+

    def __setup_mlflow(self, tls_enabled, requests_ca_bundle):
        """
        Configures MLflow environment, including TLS and experiment setup.
        """
        if tls_enabled:
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "false"
            if not os.path.exists(requests_ca_bundle):
                raise FileNotFoundError(f"TLS cert not found at {requests_ca_bundle}")
            os.environ["REQUESTS_CA_BUNDLE"] = requests_ca_bundle
        else:
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

        mlflow.set_experiment(self.experiment_name)
        if mlflow.active_run():
            mlflow.end_run()

        # enable system_metrics logging
        mlflow.system_metrics.enable_system_metrics_logging()
        mlflow.set_system_metrics_sampling_interval = 1

        self.__run = \
            mlflow.start_run(
                log_system_metrics=True
            )

        self.__run_id = self.__run.info.run_id
        self.run_name = self.__run.data.tags.get("mlflow.runName", "unnamed-run")

    @property
    def run_id(self):
        """Returns the run ID of the current MLflow run."""
        return self.__run_id

    def log_params(self, params: dict):
        """Logs parameters to MLflow and the logger."""
        self.logger.debug(f"Logging parameters: {params}")
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int = None):
        """Logs metrics to MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path: str, artifact_path: str = None):
        """Logs an artifact to MLflow."""
        self.logger.debug(f"Logging artifact: {path}")
        mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_final_results(self, results: dict):
        """
        Logs final metrics and parameters to MLflow and saves artifacts for model evaluation (like loss plots).
        - Logs essential metrics (accuracy, loss, etc.).
        - Saves arrays/tensors as artifacts (e.g., CSVs or plots).
        """
        # Iterate through the results dictionary
        for key, value in results.items():
            # Skip tensors and arrays, handle them as artifacts if needed
            if isinstance(value, (torch.Tensor, np.ndarray)):
                # If it's an array or tensor, save it as an artifact (e.g., CSV)
                file_path = self.save_tensor_as_csv(value, key)
                self.log_artifact(file_path, artifact_path=f"artifacts/{key}")
            elif isinstance(value, (int, float, str)):  # Scalars (metrics or hyperparameters)
                # Log metrics (e.g., final test accuracy) or parameters (e.g., learning rate)
                if 'accuracy' in key or 'loss' in key:
                    mlflow.log_metric(key, value)  # Log metrics
                else:
                    mlflow.log_param(key, value)  # Log hyperparameters

        # Optionally log the final model accuracy and loss metrics to MLflow
        if 'final_test_accuracy' in results:
            mlflow.log_metric("final_test_accuracy", results['final_test_accuracy'])
        if 'final_loss' in results:
            mlflow.log_metric("final_loss", results['final_loss'])

        print("[INFO] Final results logged to MLflow.")

    def save_tensor_as_csv(self, tensor, key, save_path='/mnt/mls/experiments'):
        """
        Converts tensors or arrays into CSV files and returns the file path for artifact logging.
        """
        # Convert tensor to numpy array (if it's a PyTorch tensor)
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()

        # Create the file path
        save_path = Path(f'{save_path}/{self.run_name}')
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / f"{key}_{time.strftime('%Y%m%d_%H%M%S')}.csv"

        # Save tensor/array as a CSV using pandas
        df = pd.DataFrame(tensor)
        df.to_csv(file_path, index=False)

        return file_path

    def log_plots(self, results, plot_dir="/mnt/mls/plots"):
        """
        Generates and logs training and evaluation plots to MLflow.

        Args:
            results (dict): Dictionary containing the training results including losses and accuracies.
            plot_dir (str): Directory where plots will be saved. Default is "/mnt/mls/plots".
        """
        # Extract data from results
        train_losses = results.get('train_losses', [])
        test_losses = results.get('eval_losses', [])
        accuracies = results.get('eval_accuracies', [])

        # Ensure the plot directory exists
        plot_dir = Path(f'{plot_dir}/{self.run_name}')
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        # --- Plot Losses ---
        if train_losses and test_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label="Training Loss")
            plt.plot(test_losses, label="Test Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Loss Curve")
            plt.legend()
            loss_plot_path = Path(plot_dir) / f"loss_plot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(loss_plot_path)
            plt.close()

            # Log loss plot as artifact
            self.log_artifact(loss_plot_path, artifact_path="plots")

        # --- Plot Accuracy ---
        if accuracies:
            plt.figure(figsize=(10, 6))
            plt.plot(accuracies, label="Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy (%)")
            plt.title("Accuracy Curve")
            plt.legend()
            accuracy_plot_path = Path(plot_dir) / f"accuracy_plot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(accuracy_plot_path)
            plt.close()

            # Log accuracy plot as artifact
            self.log_artifact(accuracy_plot_path, artifact_path="plots")

        # Optionally, log additional metrics to MLflow
        if accuracies:
            mlflow.log_metric("final_accuracy", accuracies[-1])
        if test_losses:
            mlflow.log_metric("final_test_loss", test_losses[-1])

        print(f"Plots saved and logged to MLflow under 'plots'.")

    def end_run(self):
        """Ends the current MLflow run."""
        self.logger.info("Ending MLflow run.")
        mlflow.end_run()

    # +-----------------------------+
    # |       Export Model          |
    # +-----------------------------+

    def log_json_labels_artifact(self,
                                 class_labels: list,
                                 artifact_filename: str = "class_labels.json",
                                 artifact_path: str = 'meta'
                                 ) -> None:
        """
        Logs a JSON-serializable object to MLflow as an artifact and deletes the temp file afterward.
        """
        tmp_dir = tempfile.mkdtemp()
        tmp_file_path = Path(tmp_dir) / artifact_filename

        try:
            with open(tmp_file_path, "w") as f:
                json.dump(class_labels, f)

            self.logger.debug(f"Logging JSON artifact: {tmp_file_path} → {artifact_path}/{artifact_filename}")
            mlflow.log_artifact(str(tmp_file_path), artifact_path=artifact_path)

        finally:
            try:
                os.remove(tmp_file_path)
                os.rmdir(tmp_dir)
                self.logger.debug(f"Deleted temp file and directory: {tmp_file_path}")
            except Exception as e:
                self.logger.warning(f"Could not delete temp file: {tmp_file_path} — {e}")

    def check_accuracy_thresh(self, accuracy: float, threshold_percent: float = 90.0) -> bool:
        """
        Determines if model accuracy is high enough to export.

        Parameters:
        - accuracy (float): Final test accuracy in percentage (e.g., 94.67)
        - threshold_percent (float): Minimum accuracy required to trigger export

        Returns:
        - bool: True if accuracy is acceptable, False if export should be skipped.
        """

        acc = round(accuracy, 4)

        # Classify tier
        if acc < 70:
            tier, desc, color = 0, "Poor", Fore.RED
        elif acc < 80:
            tier, desc, color = 1, "Fair", Fore.YELLOW
        elif acc < 90:
            tier, desc, color = 2, "Good", Fore.GREEN
        elif acc < 95:
            tier, desc, color = 3, "Very Good", Fore.GREEN
        elif acc < 99.5:
            tier, desc, color = 4, "Excellent", Fore.GREEN
        else:
            tier, desc, color = -1, "⚠ Suspiciously High (possible overfit)", Fore.RED

        # Decision logic
        if acc < threshold_percent:
            print(
                f"\n{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Accuracy below threshold ({acc} < {threshold_percent})\nSkipping model export!\n")
            return False
        elif tier >= 1:
            print(f"{color}[INFO]{Style.RESET_ALL} Accuracy {acc}% is {desc}")
            return True
        else:
            print(f"{color}[FAIL]{Style.RESET_ALL} Accuracy {acc}% is {desc}")
            return False

    def get_signature_from_loader(self, model, data_loader):
        """Captures the input and output shapes from the first batch in the data loader using infer_signature."""
        # # Get the first batch from the data loader
        # sample_input, sample_target = next(iter(data_loader))
        #
        # # Convert torch tensors to numpy arrays for better compatibility with infer_signature
        # sample_input_np = sample_input.cpu().numpy()
        # sample_target_np = sample_target.cpu().numpy()
        #
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # # Use infer_signature to automatically determine the input/output schema
        # signature = infer_signature(sample_input_np, sample_target_np)
        # Get a single batch of data from train_dataloader
        model.to(device)

        X_batch, y_batch = next(iter(data_loader))

        # Move the input batch to the appropriate device (e.g., GPU or CPU)
        X_tensor = X_batch.to(device)

        # Run prediction to get output
        with torch.no_grad():
            y_pred = model(X_tensor).cpu().numpy()

        # Infer MLflow signature and input example

        # Save the inferred signature to class-level attributes
        self.signature = infer_signature(X_tensor.cpu().numpy(), y_pred)
        self.input_example = X_tensor[0:1].cpu().numpy()

    # ---------- Model Registration --------- #

    def model_registration_onnx(self, model, sample_input, registered_name: str, save_path: str = '/mnt/mls/models'):
        """Exports the model to ONNX and registers it with MLflow."""

        # TODO: Make this one match pytorch version for order of operations
        model_format = 'onnx'

        # Ensure the model is in eval mode before exporting
        model.eval()

        # Move model and sample input to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        sample_input = sample_input.to(device)

        # Export the model to ONNX format and save it to the provided path
        torch.onnx.export(
            model,
            args=sample_input,
            f=save_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=18  # Use the appropriate ONNX opset version
        )

        # Log the ONNX model file as an artifact in MLflow
        save_path = Path(f"{save_path}/{registered_name}")
        save_path.mkdir(parents=True, exist_ok=True)
        self.log_artifact(str(save_path), artifact_path="onnx_model")

        # Register the ONNX model with MLflow's model registry
        model_uri = f"runs:/{self.run_id}/{model_format}_{registered_name}_model"
        result = mlflow.register_model(model_uri=model_uri, name=registered_name)

        # Use MLflow client to set the alias and tags for the registered model
        client = mlflow.tracking.MlflowClient()

        client.set_registered_model_alias(
            name=registered_name,
            version=result.version,
            alias='Training')

        client.set_model_version_tag(
            name=registered_name,
            version=result.version,
            key="created_by",
            value="gabe mcwilliams")

        client.set_model_version_tag(
            name=registered_name,
            version=result.version,
            key="model_format",
            value="onnx")

        self.logger.info(f"ONNX model registered as '{registered_name}' v{result.version}")

    def model_artifact_torch(self, model, registered_name: str, save_path: str = '/mnt/mls/models', alias='Training'):
        """
        Exports the model to Torch format and stores it as an artifact in MLflow.
        """

        model_format = 'torch'

        # Save the model state_dict (recommended way)
        save_path = Path(f"{save_path}/{self.run_name}")
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_file_path = save_dir / f"{self.experiment_name}_{registered_name}_{self.timestamp}.pth"

        # Save the model state_dict
        torch.save(model.state_dict(), str(model_file_path))
        self.log_artifact(str(model_file_path), artifact_path="models")

        model_info = mlflow.pytorch.log_model(
            model,
            registered_name,
            signature=self.signature,
            input_example=self.input_example,
            registered_model_name=None
        )

        # Register the model as an artifact in the model registry
        result = \
            mlflow.register_model(
                model_uri=model_info.model_uri,
                name=registered_name
            )

        # Use MLflow client to set the alias and tags for the registered model
        client = mlflow.tracking.MlflowClient()

        client.set_registered_model_alias(
            name=registered_name,
            version=result.version,
            alias=alias)

        client.set_model_version_tag(
            name=registered_name,
            version=result.version,
            key="created_by",
            value="gabe mcwilliams")

        client.set_model_version_tag(
            name=registered_name,
            version=result.version,
            key="model_format",
            value="torch")

        self.logger.info(f"Torch model saved and registered as '{registered_name}' version {result.version}")

    def get_sample_input_from_loader(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Extracts a batch from the data loader to be used as a sample input for the ONNX model.

        Args:
            data_loader (DataLoader): The data loader from which to get a sample input.

        Returns:
            torch.Tensor: A batch of data from the loader, which can be used as a sample input.
        """
        # Get a single batch (usually the first batch in the loader)
        inputs, _ = next(iter(data_loader))

        # Move the inputs to the same device as your model (typically CPU)
        # For your case, assuming the model is on CPU or GPU, you can modify this accordingly.
        inputs = inputs.to(torch.device('cpu'))  # Move to CPU if needed

        # You can also print the shape of the input for debugging
        print(f"Sample Input Shape: {inputs.shape}")

        return inputs

    # +-----------------------------+
    # |     Environment             |
    # +-----------------------------+

    def check_who_am_i(self):
        # for the current user using the `whoami` command
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}--- Current User (whoami) ---{Style.RESET_ALL}")
        try:
            user = subprocess.check_output("whoami", universal_newlines=True).strip()
            print(f"{Fore.LIGHTBLUE_EX}Current User: {Fore.CYAN}{user}")
        except Exception as e:
            print(f"{Fore.RED}Could not retrieve user info. Error: {e}")

    def check_hostname(self):
        # for the server hostname using the `hostname` command
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}--- Server Hostname (hostname) ---{Style.RESET_ALL}")
        try:
            hostname = subprocess.check_output("hostname", universal_newlines=True).strip()
            print(f"{Fore.LIGHTBLUE_EX}Server Hostname: {Fore.CYAN}{hostname}")
        except Exception as e:
            print(f"{Fore.RED}Could not retrieve hostname. Error: {e}")

    @staticmethod
    def __validate_env_var(var_name, default_value=None):
        value = os.getenv(var_name, default_value)

        # if the environment variable is AWS credentials and mask them
        if var_name in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
            if value:
                print(f"{Fore.GREEN}[PASS] {var_name}: {Fore.CYAN}*****")
            else:
                print(f"{Fore.YELLOW}[WARN] {var_name} is not set or missing value.")
        else:
            # For other environment variables, show the actual value
            if value:
                print(f"{Fore.GREEN}[PASS] {var_name}: {Fore.CYAN}{value}")
            else:
                print(f"{Fore.YELLOW}[WARN] {var_name} is not set or missing value.")

    def load_env(self, env_file_path='.env'):
        """
        Loads environment variables from a .env file into the Python environment.
        Validates the path, expands the user directory if needed, and continues if neither path is valid.
        """
        # First, try validating the original path
        if os.path.exists(env_file_path):
            print(f"[INFO] Found .env file at {env_file_path}")
            load_dotenv(env_file_path)
        else:
            # If the original path is not valid, expand it
            expanded_path = os.path.expanduser(env_file_path)
            if os.path.exists(expanded_path):
                print(f"[INFO] Found .env file at expanded path {expanded_path}")
                load_dotenv(expanded_path)
            else:
                print(
                    f"[WARN] .env file not found at {env_file_path} or expanded path {expanded_path}. Continuing without loading environment.")

    def check_env_vars(self):
        """
        Checks critical environment variables and prints diagnostic information.
        """

        # Diagnostic Block for MLflow and Artifact Storage
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}--- MLflow and Artifact Storage ---{Style.RESET_ALL}")
        self.__validate_env_var("MLFLOW_TRACKING_URI")
        self.__validate_env_var("MLFLOW_S3_ENDPOINT_URL")
        self.__validate_env_var("AWS_ACCESS_KEY_ID")
        self.__validate_env_var("AWS_SECRET_ACCESS_KEY")

        # Diagnostic Block for SSL Certs
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}--- SSL Certs for Secure Connections ---{Style.RESET_ALL}")
        self.__validate_env_var("SSL_CERT_FILE")
        self.__validate_env_var("REQUESTS_CA_BUNDLE")

        # Diagnostic Block for Vault Configuration
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}--- Vault Configuration ---{Style.RESET_ALL}")
        self.__validate_env_var("VAULT_ADDR")
        self.__validate_env_var("VAULT_CACERT")
        self.__validate_env_var("VAULT_CLIENT_CERT")
        self.__validate_env_var("VAULT_CLIENT_KEY")
        self.__validate_env_var("VAULT_NAMESPACE")

        # Diagnostic Block for CUDA-related Environment Variables
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}--- CUDA-related Environment Variables ---{Style.RESET_ALL}")
        self.__validate_env_var("PATH")
        self.__validate_env_var("LD_LIBRARY_PATH")
        self.__validate_env_var("CUDA_HOME")

        # Diagnostic Block for OMP_NUM_THREADS
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}--- OpenMP Threads ---{Style.RESET_ALL}")
        self.__validate_env_var("OMP_NUM_THREADS", "Not Set")
