"""
Quantization Decision Analysis
==============================

A research framework for analyzing how neural network quantization
affects model decisions, particularly in edge cases.

Author: PhD Student, University of Maryland
Project: ECCV 2026 Submission
"""

__version__ = "0.1.0"
__author__ = "University of Maryland"

from src.utils.config import load_config, ExperimentConfig
from src.utils.logging import setup_logging, get_logger

__all__ = [
    "load_config",
    "ExperimentConfig",
    "setup_logging",
    "get_logger",
]
