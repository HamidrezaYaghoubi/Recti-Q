"""
INT8 Post-Training Quantization using torchao.

Applies GPU-native quantization via torchao's quantize_() API.
Two modes:
  - weight_only:          INT8 weight-only (activations stay FP32)
  - weight_and_activation: INT8 weights + INT8 dynamic activations

Both run entirely on GPU – no CPU transfer needed.

Usage:
    from src.quantization import quantize_model

    q_model, stats = quantize_model(model, mode="weight_only")
    q_model, stats = quantize_model(model, mode="weight_and_activation")
"""

import copy
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchao.quantization import (
    quantize_,
    Int8WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
)

from src.utils.logging import get_logger

logger = get_logger("qda.quantization")


# Valid quantization modes
QUANT_MODES = {
    "weight_only": "INT8 weight-only (activations FP32)",
    "weight_and_activation": "INT8 weights + INT8 dynamic activations",
}


# ============================================================================
# Helpers
# ============================================================================

def get_model_size_mb(model: nn.Module) -> float:
    """
    Compute serialised model size in MB.

    Uses torch.save to a BytesIO buffer so that torchao's quantized
    tensor subclasses (which store int8 data internally) are measured
    correctly — plain p.element_size() always reports 4 for them.
    """
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 ** 2)


def count_layers(model: nn.Module) -> Dict[str, int]:
    """
    Count layer types in a model.
    Returns dict with counts for Linear, Conv2d, and total modules.
    """
    counts = {"Linear": 0, "Conv2d": 0, "total": 0}
    for module in model.modules():
        counts["total"] += 1
        if isinstance(module, nn.Linear):
            counts["Linear"] += 1
        elif isinstance(module, nn.Conv2d):
            counts["Conv2d"] += 1
    return counts


# ============================================================================
# Main API
# ============================================================================

def quantize_model(
    model: nn.Module,
    mode: str = "weight_only",
    device: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Quantize a model using torchao.

    The model is deep-copied so the original stays untouched.
    quantize_() is applied in-place on the copy.

    Args:
        model:  The nn.Module to quantize (e.g. model.backbone).
        mode:   "weight_only" or "weight_and_activation".
        device: Device for quantization (default: keep current device).

    Returns:
        (quantized_model, stats_dict)
    """
    if mode not in QUANT_MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose from: {list(QUANT_MODES.keys())}"
        )

    logger.info(f"Quantizing model – mode={mode}")

    # Measure original size
    original_size = get_model_size_mb(model)
    original_layers = count_layers(model)

    # Deep copy so the original model is not mutated
    q_model = copy.deepcopy(model)
    q_model.eval()

    # Move to target device if specified
    if device is not None:
        q_model = q_model.to(device)

    # Pick config
    if mode == "weight_only":
        config = Int8WeightOnlyConfig()
    else:
        config = Int8DynamicActivationInt8WeightConfig()

    # Apply quantization (in-place, targets nn.Linear by default)
    t0 = time.time()
    quantize_(q_model, config)
    quant_time = time.time() - t0

    # Measure quantized size
    quantized_size = get_model_size_mb(q_model)
    quantized_layers = count_layers(q_model)

    # Build stats
    stats = {
        "mode": mode,
        "mode_description": QUANT_MODES[mode],
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "compression_ratio": original_size / max(quantized_size, 1e-6),
        "size_reduction_pct": (1 - quantized_size / max(original_size, 1e-6)) * 100,
        "quantization_time_s": quant_time,
        "original_layers": original_layers,
        "quantized_layers": quantized_layers,
        "target_layers": "nn.Linear (torchao default)",
    }

    logger.info(
        f"  Done – {original_size:.1f} MB → {quantized_size:.1f} MB "
        f"({stats['compression_ratio']:.2f}x) in {quant_time:.1f}s"
    )

    return q_model, stats
