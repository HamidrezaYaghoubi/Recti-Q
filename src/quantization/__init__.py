"""
Quantization module for applying quantization to models.

This module will be fully implemented in Week 2-3.
"""

from src.quantization.quantizer import (
    Quantizer,
    QuantizationConfig,
    quantize_model,
)

__all__ = [
    "Quantizer",
    "QuantizationConfig",
    "quantize_model",
]
