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
import torch.nn.functional as F
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
# 4-bit weight-only Conv2d (for CNNs whose weights torchao's Linear-only
# int4 path leaves untouched, e.g. ResNet50 ~92% Conv2d weights)
# ============================================================================

def _hqq_int4_groups(Wg: torch.Tensor, nbits: int = 4, iters: int = 20,
                     lp_norm: float = 0.7, beta0: float = 1e1, kappa: float = 1.01):
    """HQQ (half-quadratic) group-wise affine int4: W ≈ scale * (q - zero).

    Calibration-free, like the paper's Int4WeightOnly(use_hqq=True). Optimizes
    the per-group zero-point with an Lp (p<1) proximal solver to minimize weight
    error — crucial for CNNs, where plain round-to-nearest int4 desyncs frozen
    BatchNorm stats and collapses accuracy.

    Wg: [out, n_groups, gs]. Returns (q uint8, scale, zero) with q in [0, 2^b-1].
    """
    qmax = (1 << nbits) - 1
    wmin = Wg.min(dim=2, keepdim=True).values
    wmax = Wg.max(dim=2, keepdim=True).values
    scale = ((wmax - wmin) / qmax).clamp_(min=1e-8)
    zero = -wmin / scale  # init = round-to-nearest; q = round(W/scale + zero) in [0, qmax]

    def _err(z):
        q = (Wg / scale + z).round().clamp_(0, qmax)
        return (Wg - scale * (q - z)).abs().mean().item()

    best_zero = zero.clone()
    best_err = _err(zero)  # RTN baseline — HQQ can only improve on this
    beta = beta0
    for _ in range(iters):
        q = (Wg / scale + zero).round().clamp_(0, qmax)
        err = Wg - scale * (q - zero)
        shrunk = torch.sign(err) * F.relu(
            err.abs() - (1.0 / beta) * err.abs().clamp_(min=1e-8).pow(lp_norm - 1)
        )
        zero = (q - (Wg - shrunk) / scale).mean(dim=2, keepdim=True)
        beta *= kappa
        cur = _err(zero)
        if cur < best_err:
            best_err, best_zero = cur, zero.clone()
        else:
            break
    zero = best_zero
    q = (Wg / scale + zero).round().clamp_(0, qmax).to(torch.uint8)
    return q, scale, zero


class Int4Conv2d(nn.Module):
    """Weight-only 4-bit (group-wise HQQ) drop-in for nn.Conv2d.

    Stores conv weights as packed int4 (two values per byte) plus per-group
    fp16 scale/zero, dequantizing W = scale*(q - zero) on the fly in forward().
    Yields a real ~8x reduction of the (dominant) conv weight storage, so CNNs
    actually compress under W4 — torchao's int4 path only covers nn.Linear.
    """

    def __init__(self, conv: nn.Conv2d, group_size: int = 128):
        super().__init__()
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.out_channels = conv.out_channels
        self.orig_shape = tuple(conv.weight.shape)  # [out, in/groups, kH, kW]

        W = conv.weight.detach().reshape(self.out_channels, -1).float()  # [out, K]
        K = W.shape[1]
        gs = K if (group_size is None or group_size <= 0) else min(group_size, K)
        n_groups = (K + gs - 1) // gs
        Kp = n_groups * gs
        self.K, self.gs, self.n_groups, self.Kp = K, gs, n_groups, Kp

        if Kp > K:
            W = F.pad(W, (0, Kp - K))
        Wg = W.reshape(self.out_channels, n_groups, gs)
        q, scale, zero = _hqq_int4_groups(Wg, nbits=4)
        q = q.reshape(self.out_channels, Kp)
        if Kp % 2 == 1:  # ensure an even number of nibbles to pack
            q = F.pad(q, (0, 1))
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()  # [out, ceil(Kp/2)]

        self.register_buffer("weight_packed", packed)
        self.register_buffer("scale", scale.squeeze(-1).half())  # [out, n_groups]
        self.register_buffer("zero", zero.squeeze(-1).half())    # [out, n_groups]
        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.detach().clone())
        else:
            self.bias = None

    def _dequantize_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        packed = self.weight_packed
        low = (packed & 0x0F).float()
        high = (packed >> 4).float()
        q = torch.stack([low, high], dim=2).reshape(self.out_channels, -1)  # interleave
        q = q[:, : self.Kp].reshape(self.out_channels, self.n_groups, self.gs)
        scale = self.scale.float().unsqueeze(-1)
        zero = self.zero.float().unsqueeze(-1)
        W = (scale * (q - zero)).reshape(self.out_channels, self.Kp)[:, : self.K]
        return W.reshape(self.orig_shape).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize_weight(x.dtype, x.device)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)


def _quantize_conv2d_int4(model: nn.Module, group_size: int = 128) -> int:
    """Replace every nn.Conv2d in `model` with Int4Conv2d, in place. Returns count."""
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d):
            device = child.weight.device
            setattr(model, name, Int4Conv2d(child, group_size=group_size).to(device))
            n += 1
        else:
            n += _quantize_conv2d_int4(child, group_size=group_size)
    return n


@torch.no_grad()
def recalibrate_batchnorm(model: nn.Module, loader, device: str, num_batches: int = 200) -> int:
    """Re-estimate BatchNorm running stats to match the quantized weights.

    Frozen BN running mean/var (fitted for FP weights) desync once weights are
    quantized, which can collapse CNN accuracy. Reset them and recompute via a
    cumulative average over a few source batches. Returns #BN layers updated.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    saved = [(bn.momentum, bn.training) for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative moving average over the calibration batches
        bn.train()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        model(batch[0].to(device))
    for bn, (mom, was_training) in zip(bns, saved):
        bn.momentum = mom
        bn.train(was_training)
    return len(bns)


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
    quantize_conv: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Quantize a model using torchao.

    The model is deep-copied so the original stays untouched.
    quantize_() is applied in-place on the copy.

    Args:
        model:         The nn.Module to quantize (e.g. model.backbone).
        mode:          One of QUANT_MODES keys. Default "W4" (paper PTQ baseline).
        device:        Device for quantization (default: keep current device).
        group_size:    Group size for W4 quantization (default 128).
        use_hqq:       Use HQQ for W4 quantization (default True, per paper).
        quantize_conv: For W4, also 4-bit quantize nn.Conv2d via Int4Conv2d.
                       Default False — the paper's ImageNet-C W4 is Linear-only
                       (ResNet50 stays ~91 MB, Table III), so this matches the
                       paper. EXPERIMENTAL: enabling it compresses CNNs heavily
                       but currently degrades ResNet50 accuracy (frozen BatchNorm
                       desyncs from the quantized conv weights). See Int4Conv2d.

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
    quantize_(q_model, ao_config)  # torchao: nn.Linear weights -> int4
    n_conv = 0
    if mode == "W4" and quantize_conv:
        n_conv = _quantize_conv2d_int4(q_model, group_size=group_size)  # nn.Conv2d -> int4
    quant_time = time.time() - t0

    quantized_size = get_model_size_mb(q_model)
    quantized_layers = count_layers(q_model)

    target = "nn.Linear (torchao)" + (f" + {n_conv} nn.Conv2d (int4)" if n_conv else "")
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
        "conv_layers_int4": n_conv,
        "target_layers": target,
    }

    logger.info(
        f"  Done – {original_size:.1f} MB → {quantized_size:.1f} MB "
        f"({stats['compression_ratio']:.2f}x) in {quant_time:.1f}s"
    )

    return q_model, stats
