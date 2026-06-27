"""Post-Training Quantization using torchao.

Applies GPU-native quantization via torchao's quantize_() API.
All modes run entirely on GPU – no CPU transfer needed.

Supported modes (short name → torchao config):
  - W4     : Int4WeightOnlyConfig(use_hqq=True)  ← paper PTQ baseline
  - W8A16  : Int8WeightOnlyConfig              (INT8 weights, FP16/32 activations)
  - W8A8   : Int8DynamicActivationInt8WeightConfig (INT8 weights + INT8 dyn. activations)
  - W4A16  : Int4WeightOnlyConfig              (INT4 group-wise weights, FP16/32 act.)
  - FP8wo  : Float8WeightOnlyConfig            (FP8 E4M3 weights only)
  - FP8    : Float8DynamicActivationFloat8WeightConfig (FP8 weights + FP8 dyn. act.)
  - W4A8fp : Float8DynamicActivationInt4WeightConfig   (INT4 weights + FP8 dyn. act.)

Usage:
    from src.quantization import quantize_model, QUANT_MODES

    q_model, stats = quantize_model(model)                   # W4 default (paper)
    q_model, stats = quantize_model(model, mode="W8A8")
    q_model, stats = quantize_model(model, mode="W4", group_size=128, use_hqq=True)
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

logger = get_logger("recti_q.quantization")


# ── Quantization mode registry ──
QUANT_MODES = {
    # Paper PTQ baseline
    "W4":     "INT4 weight-only HQQ (paper PTQ baseline)",
    # INT8
    "W8A16":  "INT8 weight-only (activations FP32)",
    "W8A8":   "INT8 weights + INT8 dynamic activations",
    # INT4
    "W4A16":  "INT4 weight-only, group_size=128 (activations FP32)",
    # FP8
    "FP8wo":  "FP8 E4M3 weight-only",
    "FP8":    "FP8 E4M3 weights + FP8 dynamic activations",
    # Mixed
    "W4A8fp": "INT4 weights + FP8 dynamic activations",
}

# Human-friendly / legacy aliases
_ALIASES = {
    "weight_only":           "W8A16",
    "weight_and_activation": "W8A8",
    "dynamic":               "W8A8",
    "int8":                  "W8A8",
    "int4":                  "W4A16",
}


def _get_torchao_config(mode: str, group_size: int = 128, use_hqq: bool = True):
    """Return the torchao config object for a given mode.

    INT4 and FP8 configs are imported lazily so the module can load
    even when fbgemm-gpu-genai is missing or incompatible.

    group_size and use_hqq are only used by W4 (paper baseline).
    """
    # INT8 – always available
    if mode == "W8A16":
        return Int8WeightOnlyConfig()
    if mode == "W8A8":
        return Int8DynamicActivationInt8WeightConfig()

    # Lazy imports for configs that may need extra deps
    if mode == "W4":
        from torchao.quantization import Int4WeightOnlyConfig
        return Int4WeightOnlyConfig(group_size=group_size, use_hqq=use_hqq)
    if mode == "W4A16":
        from torchao.quantization import Int4WeightOnlyConfig
        return Int4WeightOnlyConfig(group_size=128)
    if mode == "FP8wo":
        from torchao.quantization import Float8WeightOnlyConfig
        return Float8WeightOnlyConfig()
    if mode == "FP8":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        return Float8DynamicActivationFloat8WeightConfig()
    if mode == "W4A8fp":
        from torchao.quantization import Float8DynamicActivationInt4WeightConfig
        return Float8DynamicActivationInt4WeightConfig()

    raise ValueError(f"Unknown mode '{mode}'")


# ============================================================================
# Helpers
# ============================================================================

def get_model_size_mb(model: nn.Module) -> float:
    """Compute serialised model size in MB.

    Uses torch.save to a BytesIO buffer so that torchao's quantized
    tensor subclasses (which store int8 data internally) are measured
    correctly — plain p.element_size() always reports 4 for them.
    """
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 ** 2)


def count_layers(model: nn.Module) -> Dict[str, int]:
    """Count layer types in a model.

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

def resolve_mode(mode: str) -> str:
    """Resolve aliases (e.g. 'dynamic' → 'W8A8') and validate."""
    raw_mode = mode.strip()

    # Accept canonical names in any case (e.g., "w4" -> "W4")
    upper_mode = raw_mode.upper()
    if upper_mode in QUANT_MODES:
        return upper_mode

    # Resolve human-friendly aliases (lower case)
    resolved = _ALIASES.get(raw_mode.lower(), raw_mode)
    if resolved not in QUANT_MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose from: {list(QUANT_MODES.keys())}"
        )
    return resolved


def quantize_model(
    model: nn.Module,
    mode: str = "W4",
    device: Optional[str] = None,
    group_size: int = 128,
    use_hqq: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Quantize a model using torchao.

    The model is deep-copied so the original stays untouched.
    quantize_() is applied in-place on the copy.

    Args:
        model:      The nn.Module to quantize (e.g. model.backbone).
        mode:       One of QUANT_MODES keys. Default "W4" (paper PTQ baseline).
        device:     Device for quantization (default: keep current device).
        group_size: Group size for W4 quantization (default 128).
        use_hqq:    Use HQQ for W4 quantization (default True, per paper).

    Returns:
        (quantized_model, stats_dict)
    """
    mode = resolve_mode(mode)

    logger.info(f"Quantizing model – mode={mode} ({QUANT_MODES[mode]})")

    original_size = get_model_size_mb(model)
    original_layers = count_layers(model)

    # Deep copy so the original model is not mutated
    q_model = copy.deepcopy(model)
    q_model.eval()

    if device is not None:
        q_model = q_model.to(device)

    ao_config = _get_torchao_config(mode, group_size=group_size, use_hqq=use_hqq)

    t0 = time.time()
    quantize_(q_model, ao_config)
    quant_time = time.time() - t0

    quantized_size = get_model_size_mb(q_model)
    quantized_layers = count_layers(q_model)

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
