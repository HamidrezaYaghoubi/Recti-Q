"""
Checkpoint management for saving and loading experiment state.

This module provides utilities for saving model checkpoints,
predictions, and experiment results in various formats.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

from src.utils.config import ExperimentConfig, OutputConfig
from src.utils.logging import get_logger


class CheckpointManager:
    """
    Manages saving and loading of checkpoints and results.
    
    Provides a unified interface for saving:
    - Model checkpoints (weights, optimizer state)
    - Predictions (logits, labels, confidences)
    - Experiment results (metrics, configurations)
    """
    
    def __init__(
        self,
        config: Union[ExperimentConfig, OutputConfig],
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            config: Experiment or output configuration.
            experiment_name: Name of the experiment for directory naming.
        """
        if isinstance(config, ExperimentConfig):
            self.output_config = config.output
            self.experiment_name = experiment_name or config.name
        else:
            self.output_config = config
            self.experiment_name = experiment_name or "experiment"
        
        self._logger = get_logger("qda.checkpoint")
        
        # Create directories
        self.checkpoint_dir = Path(self.output_config.checkpoint_dir) / self.experiment_name
        self.results_dir = Path(self.output_config.results_dir) / self.experiment_name
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self._logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self._logger.info(f"Results directory: {self.results_dir}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        model_name: str,
        precision: str = "fp32",
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a model checkpoint.
        
        Args:
            model: PyTorch model to save.
            model_name: Name of the model.
            precision: Precision level (fp32, int8, etc.).
            optimizer: Optional optimizer to save.
            epoch: Optional epoch number.
            metrics: Optional metrics dictionary.
            extra_data: Optional additional data to save.
            
        Returns:
            Path to the saved checkpoint.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{precision}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            "model_name": model_name,
            "precision": precision,
            "timestamp": timestamp,
            "model_state_dict": model.state_dict(),
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint["epoch"] = epoch
        
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        if extra_data is not None:
            checkpoint["extra_data"] = extra_data
        
        torch.save(checkpoint, checkpoint_path)
        self._logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
            model: Optional model to load weights into.
            optimizer: Optional optimizer to load state into.
            device: Device to load the checkpoint to.
            
        Returns:
            Dictionary containing checkpoint data.
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            self._logger.info(f"Loaded model weights from: {checkpoint_path}")
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._logger.info(f"Loaded optimizer state from: {checkpoint_path}")
        
        return checkpoint
    
    def save_predictions(
        self,
        predictions: Dict[str, Any],
        model_name: str,
        dataset_name: str,
        precision: str = "fp32",
        format: Optional[str] = None,
    ) -> Path:
        """
        Save model predictions.
        
        Args:
            predictions: Dictionary containing predictions data.
                Expected keys: predictions, labels, logits, confidences, etc.
            model_name: Name of the model.
            dataset_name: Name of the dataset.
            precision: Precision level.
            format: Output format (pickle, json, hdf5). Uses config default if None.
            
        Returns:
            Path to the saved predictions file.
        """
        format = format or self.output_config.prediction_format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine file extension
        ext_map = {"pickle": ".pkl", "json": ".json", "hdf5": ".h5"}
        ext = ext_map.get(format, ".pkl")
        
        filename = f"{model_name}_{precision}_{dataset_name}_{timestamp}{ext}"
        output_path = self.results_dir / filename
        
        if format == "pickle":
            self._save_pickle(predictions, output_path)
        elif format == "json":
            self._save_json(predictions, output_path)
        elif format == "hdf5":
            self._save_hdf5(predictions, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        self._logger.info(f"Saved predictions: {output_path}")
        return output_path
    
    def _save_pickle(self, data: Dict[str, Any], path: Path) -> None:
        """Save data as pickle file."""
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save data as JSON file (converts numpy arrays to lists)."""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            return obj
        
        # Convert all values
        converted_data = {}
        for key, value in data.items():
            converted_data[key] = convert(value)
        
        with open(path, "w") as f:
            json.dump(converted_data, f, indent=2)
    
    def _save_hdf5(self, data: Dict[str, Any], path: Path) -> None:
        """Save data as HDF5 file."""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")
        
        with h5py.File(path, "w") as f:
            for key, value in data.items():
                if isinstance(value, (np.ndarray, list)):
                    f.create_dataset(key, data=np.array(value))
                elif isinstance(value, torch.Tensor):
                    f.create_dataset(key, data=value.cpu().numpy())
                elif isinstance(value, (int, float, str)):
                    f.attrs[key] = value
    
    def load_predictions(
        self,
        path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Load predictions from a file.
        
        Args:
            path: Path to the predictions file.
            
        Returns:
            Dictionary containing predictions data.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found: {path}")
        
        ext = path.suffix.lower()
        
        if ext == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        elif ext == ".json":
            with open(path, "r") as f:
                return json.load(f)
        elif ext in (".h5", ".hdf5"):
            if not HDF5_AVAILABLE:
                raise ImportError("h5py is required for HDF5 format")
            data = {}
            with h5py.File(path, "r") as f:
                for key in f.keys():
                    data[key] = f[key][:]
                for key, value in f.attrs.items():
                    data[key] = value
            return data
        else:
            raise ValueError(f"Unknown file format: {ext}")
    
    def save_metrics(
        self,
        metrics: Dict[str, float],
        model_name: str,
        dataset_name: str,
        precision: str = "fp32",
    ) -> Path:
        """
        Save evaluation metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metric names and values.
            model_name: Name of the model.
            dataset_name: Name of the dataset.
            precision: Precision level.
            
        Returns:
            Path to the saved metrics file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{model_name}_{precision}_{dataset_name}_{timestamp}.json"
        output_path = self.results_dir / filename
        
        output_data = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "precision": precision,
            "timestamp": timestamp,
            "metrics": metrics,
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        self._logger.info(f"Saved metrics: {output_path}")
        return output_path
    
    def get_latest_checkpoint(
        self,
        model_name: Optional[str] = None,
        precision: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Get the most recent checkpoint file.
        
        Args:
            model_name: Filter by model name.
            precision: Filter by precision.
            
        Returns:
            Path to the latest checkpoint, or None if not found.
        """
        pattern = "*.pt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        # Filter by model name and precision
        if model_name:
            checkpoints = [c for c in checkpoints if model_name in c.name]
        if precision:
            checkpoints = [c for c in checkpoints if precision in c.name]
        
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    def list_checkpoints(
        self,
        model_name: Optional[str] = None,
    ) -> List[Path]:
        """
        List all available checkpoints.
        
        Args:
            model_name: Filter by model name.
            
        Returns:
            List of checkpoint paths.
        """
        pattern = f"{model_name}*.pt" if model_name else "*.pt"
        return sorted(self.checkpoint_dir.glob(pattern))
    
    def list_predictions(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> List[Path]:
        """
        List all saved prediction files.
        
        Args:
            model_name: Filter by model name.
            dataset_name: Filter by dataset name.
            
        Returns:
            List of prediction file paths.
        """
        files = []
        for ext in [".pkl", ".json", ".h5"]:
            files.extend(self.results_dir.glob(f"*{ext}"))
        
        # Filter
        if model_name:
            files = [f for f in files if model_name in f.name]
        if dataset_name:
            files = [f for f in files if dataset_name in f.name]
        
        return sorted(files)
