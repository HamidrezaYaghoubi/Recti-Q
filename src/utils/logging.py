"""
Logging utilities with Weights & Biases integration.

This module provides structured logging to console, files, and wandb.
"""

import logging
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.utils.config import ExperimentConfig, LoggingConfig


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        
        # Add color to level name
        record.levelname = f"{color}{record.levelname}{reset}"
        
        return super().format(record)


def setup_logging(
    config: Union[ExperimentConfig, LoggingConfig],
    experiment_name: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging with console, file, and optional wandb handlers.
    
    Args:
        config: Experiment or logging configuration.
        experiment_name: Name for the experiment (used in log file names).
        
    Returns:
        Configured root logger.
    """
    if isinstance(config, ExperimentConfig):
        log_config = config.logging
        exp_name = experiment_name or config.name
    else:
        log_config = config
        exp_name = experiment_name or "experiment"
    
    # Create root logger
    logger = logging.getLogger("qda")  # quantization-decision-analysis
    logger.setLevel(getattr(logging, log_config.level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = ColoredFormatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_config.log_to_file:
        log_dir = Path(log_config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{exp_name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_logger(name: str = "qda") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (hierarchical with dots).
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def _sanitize_wandb_job_type(job_type: Optional[str]) -> Optional[str]:
    """
    Ensure WandB job_type stays within backend limits.

    WandB enforces a 64-character limit for job_type.
    """
    if not job_type:
        return job_type
    if len(job_type) <= 64:
        return job_type
    suffix = hashlib.sha1(job_type.encode("utf-8")).hexdigest()[:8]
    head_len = max(1, 64 - 1 - len(suffix))
    return f"{job_type[:head_len]}_{suffix}"


class WandbLogger:
    """
    Wrapper for Weights & Biases logging.
    
    Provides a consistent interface for logging metrics, artifacts,
    and other data to wandb.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        resume: bool = False,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        run_group: Optional[str] = None,
        run_job_type: Optional[str] = None,
        extra_tags: Optional[List[str]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize wandb logging.
        
        Args:
            config: Experiment configuration.
            resume: Whether to resume a previous run.
            run_id: Specific run ID to resume.
            run_name: Optional explicit run name.
            run_group: Optional wandb run group.
            run_job_type: Optional wandb job type.
            extra_tags: Optional tags appended to base config tags.
            extra_config: Optional key-values merged into wandb config.
        """
        self.config = config
        self.enabled = config.logging.wandb.enabled and WANDB_AVAILABLE
        self._run = None
        self._logger = get_logger("qda.wandb")
        
        if not WANDB_AVAILABLE and config.logging.wandb.enabled:
            self._logger.warning(
                "wandb is not installed. Install with: pip install wandb"
            )
        
        if self.enabled:
            self._init_wandb(
                resume=resume,
                run_id=run_id,
                run_name=run_name,
                run_group=run_group,
                run_job_type=run_job_type,
                extra_tags=extra_tags,
                extra_config=extra_config,
            )
    
    def _init_wandb(
        self,
        resume: bool,
        run_id: Optional[str],
        run_name: Optional[str],
        run_group: Optional[str],
        run_job_type: Optional[str],
        extra_tags: Optional[List[str]],
        extra_config: Optional[Dict[str, Any]],
    ) -> None:
        """Initialize wandb run."""
        wandb_config = self.config.logging.wandb
        
        # Prepare config dict for wandb
        config_dict = {
            "experiment_name": self.config.name,
            "seed": self.config.seed,
            "device": self.config.device,
            "models": [m.name for m in self.config.models],
            "datasets": list(self.config.datasets.keys()),
            "quantization_enabled": self.config.quantization.enabled,
        }
        if extra_config:
            config_dict.update(extra_config)

        tags = list(wandb_config.tags)
        if extra_tags:
            for tag in extra_tags:
                if tag not in tags:
                    tags.append(tag)

        safe_job_type = _sanitize_wandb_job_type(run_job_type)
        init_kwargs = {
            "project": wandb_config.project,
            "entity": wandb_config.entity,
            "name": run_name or self.config.name,
            "group": run_group,
            "job_type": safe_job_type,
            "config": config_dict,
            "tags": tags,
            "notes": wandb_config.notes,
            "resume": "allow" if resume else None,
            "id": run_id,
            "reinit": True,
        }

        try:
            self._run = wandb.init(**init_kwargs)
        except Exception as e:
            msg = str(e)
            if "64 limit exceeded for JobType" not in msg:
                raise

            self._logger.warning(
                "WandB rejected job_type length; retrying run init without job_type."
            )
            init_kwargs["job_type"] = None
            self._run = wandb.init(**init_kwargs)
        
        self._logger.info(f"Initialized wandb run: {self._run.name}")
    
    def log(
        self, 
        metrics: Dict[str, Any], 
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: Global step (optional).
            commit: Whether to commit the log immediately.
        """
        if not self.enabled or self._run is None:
            return
        
        wandb.log(metrics, step=step, commit=commit)
    
    def log_table(
        self,
        name: str,
        columns: list,
        data: list,
    ) -> None:
        """
        Log a table to wandb.
        
        Args:
            name: Table name.
            columns: Column names.
            data: Table data (list of rows).
        """
        if not self.enabled or self._run is None:
            return
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({name: table})
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an artifact (file or directory) to wandb.
        
        Args:
            artifact_path: Path to the artifact.
            name: Artifact name.
            artifact_type: Type of artifact (model, dataset, etc.).
            metadata: Optional metadata dictionary.
        """
        if not self.enabled or self._run is None:
            return
        
        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
        artifact_path = Path(artifact_path)
        
        if artifact_path.is_dir():
            artifact.add_dir(str(artifact_path))
        else:
            artifact.add_file(str(artifact_path))
        
        self._run.log_artifact(artifact)
        self._logger.info(f"Logged artifact: {name}")
    
    def log_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Log summary metrics (final metrics at end of experiment).
        
        Args:
            metrics: Dictionary of summary metrics.
        """
        if not self.enabled or self._run is None:
            return
        
        for key, value in metrics.items():
            self._run.summary[key] = value
    
    def finish(self) -> None:
        """Finish the wandb run."""
        if self.enabled and self._run is not None:
            self._run.finish()
            self._logger.info("Finished wandb run")
            self._run = None
    
    @property
    def run(self):
        """Get the wandb run object."""
        return self._run


class MetricsLogger:
    """
    Unified metrics logger that writes to console, file, and wandb.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        wandb_logger: Optional[WandbLogger] = None,
    ):
        """
        Initialize metrics logger.
        
        Args:
            config: Experiment configuration.
            wandb_logger: Optional WandbLogger instance.
        """
        self.config = config
        self.wandb_logger = wandb_logger
        self._logger = get_logger("qda.metrics")
        self._step = 0
        self._metrics_history: Dict[str, list] = {}
    
    def log(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log metrics to all outputs.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: Global step (uses internal counter if not provided).
            prefix: Prefix to add to metric names.
        """
        if step is None:
            step = self._step
            self._step += 1
        
        # Add prefix
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self._logger.info(f"Step {step} | {metrics_str}")
        
        # Store in history
        for name, value in metrics.items():
            if name not in self._metrics_history:
                self._metrics_history[name] = []
            self._metrics_history[name].append((step, value))
        
        # Log to wandb
        if self.wandb_logger is not None:
            self.wandb_logger.log(metrics, step=step)
    
    def get_history(self, metric_name: str) -> list:
        """Get the history of a specific metric."""
        return self._metrics_history.get(metric_name, [])
    
    def get_all_history(self) -> Dict[str, list]:
        """Get the history of all metrics."""
        return self._metrics_history.copy()
