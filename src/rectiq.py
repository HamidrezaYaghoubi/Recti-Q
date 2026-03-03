"""
Recti-Q adapter for frozen quantized YOLO detection models.

This module provides a lightweight LoRA-style residual adapter that operates on
pre-detect feature maps and can be trained while the quantized model stays
frozen. It follows the Recti-Q idea of feature-space rectification:

    z = z_q + g_phi(u)

where:
  - z_q: frozen quantized model output,
  - u: pre-detect feature maps,
  - g_phi: low-rank adapter trained on source data.

The implementation is intentionally simple and backend-agnostic:
  1) Attach LoRA residual blocks to YOLO Detect-module inputs via a pre-hook.
  2) Freeze all quantized model parameters.
  3) Train only adapter parameters with:
     - optional feature-KD to a frozen teacher,
     - optional task loss callback for supervised objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import weakref
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm


Tensor = torch.Tensor


def _as_scalar_tensor(value: Any, device: Optional[torch.device] = None) -> Tensor:
    """
    Convert input to a scalar tensor (0-dim) for stable backward/logging.
    """
    if isinstance(value, torch.Tensor):
        t = value
    else:
        t = torch.as_tensor(value, device=device)
    if t.ndim == 0:
        return t
    return t.sum()


def _to_attr_namespace(values: Dict[str, Any]) -> Any:
    """
    Convert a dict to an attribute-access namespace expected by Ultralytics.
    """
    try:
        from ultralytics.utils import IterableSimpleNamespace  # type: ignore

        return IterableSimpleNamespace(**values)
    except Exception:
        return SimpleNamespace(**values)


def _ensure_ultralytics_loss_hyp_compat(q_model: nn.Module) -> None:
    """
    Ensure Ultralytics detection loss can read hyperparameters via attributes.

    Some fine-tuned checkpoints load with `model.args` as a minimal dict
    (missing `box`, `cls`, `dfl`) while loss code expects `self.hyp.box`.
    """
    required_defaults = {"box": 7.5, "cls": 0.5, "dfl": 1.5}

    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT  # type: ignore

        base_cfg = dict(DEFAULT_CFG_DICT)
    except Exception:
        base_cfg = {}

    args = getattr(q_model, "args", None)
    if isinstance(args, dict):
        merged = dict(base_cfg)
        merged.update(args)
        for k, v in required_defaults.items():
            merged.setdefault(k, v)
        setattr(q_model, "args", _to_attr_namespace(merged))
    elif args is not None:
        for k, v in required_defaults.items():
            if not hasattr(args, k):
                try:
                    setattr(args, k, v)
                except Exception:
                    pass

    criterion = getattr(q_model, "criterion", None)
    if criterion is None:
        return

    hyp = getattr(criterion, "hyp", None)
    if isinstance(hyp, dict):
        merged = dict(base_cfg)
        merged.update(hyp)
        for k, v in required_defaults.items():
            merged.setdefault(k, v)
        criterion.hyp = _to_attr_namespace(merged)
    elif hyp is not None:
        for k, v in required_defaults.items():
            if not hasattr(hyp, k):
                try:
                    setattr(hyp, k, v)
                except Exception:
                    pass


def _resolve_yolo_runtime(model_like: Any) -> Any:
    """
    Resolve an ultralytics YOLO runtime handle from common wrappers.

    Supported:
      - qda detection model (`model._yolo`)
      - qda wrapper backbone (`model.backbone.yolo`)
      - raw ultralytics `YOLO` object
    """
    backbone = getattr(model_like, "backbone", None)
    if backbone is not None and hasattr(backbone, "yolo"):
        return backbone.yolo

    if hasattr(model_like, "_yolo"):
        return model_like._yolo

    if hasattr(model_like, "yolo"):
        return model_like.yolo

    if hasattr(model_like, "model") and hasattr(model_like, "predict"):
        # Best-effort fallback for raw ultralytics YOLO object.
        return model_like

    raise ValueError(
        "Could not resolve YOLO runtime from input. Expected qda YOLO model "
        "(with `_yolo` or `backbone.yolo`) or a raw ultralytics YOLO object."
    )


def _find_detect_module(yolo_runtime: Any) -> nn.Module:
    """
    Find the YOLO Detect module inside `yolo_runtime.model`.
    """
    model = getattr(yolo_runtime, "model", None)
    if model is None:
        raise ValueError("YOLO runtime has no `.model` attribute.")

    for module in model.modules():
        if module.__class__.__name__.lower() == "detect":
            return module
    raise RuntimeError("Could not find a Detect module in the YOLO model.")


def _extract_conv_in_channels(module: nn.Module) -> Optional[int]:
    """
    Return input channels if this module wraps/contains a Conv2d.
    """
    if isinstance(module, nn.Conv2d):
        return int(module.in_channels)

    conv = getattr(module, "conv", None)
    if isinstance(conv, nn.Conv2d):
        return int(conv.in_channels)

    return None


def _infer_detect_input_channels(detect_module: nn.Module) -> List[int]:
    """
    Infer channel size for each feature map entering Detect forward().
    """
    cv2 = getattr(detect_module, "cv2", None)
    if cv2 is None:
        raise RuntimeError("Detect module has no `cv2` attribute; unsupported YOLO variant.")

    channels: List[int] = []
    for branch in cv2:
        ch = None
        if isinstance(branch, nn.Sequential):
            for sub in branch:
                ch = _extract_conv_in_channels(sub)
                if ch is not None:
                    break
        else:
            ch = _extract_conv_in_channels(branch)

        if ch is None:
            raise RuntimeError("Unable to infer detect input channels from branch.")
        channels.append(ch)

    if not channels:
        raise RuntimeError("No detect input channels inferred.")
    return channels


def _get_yolo_layer_list(yolo_runtime: Any) -> List[nn.Module]:
    """Return Ultralytics detection-model layer list (`model.model`)."""
    model = getattr(yolo_runtime, "model", None)
    layers = getattr(model, "model", None)
    if isinstance(layers, nn.ModuleList):
        return list(layers)
    if isinstance(layers, nn.Sequential):
        return list(layers)
    if isinstance(layers, (list, tuple)) and all(isinstance(m, nn.Module) for m in layers):
        return list(layers)
    if isinstance(layers, nn.Module):
        children = list(layers.children())
        if children and all(isinstance(m, nn.Module) for m in children):
            return children
    raise RuntimeError("Could not resolve YOLO layer list from runtime model.")


def _resolve_graph_indices(f_spec: Any, current_idx: int, n_layers: int) -> List[int]:
    """
    Resolve Ultralytics graph indices (`module.f`) to absolute non-negative indices.
    """
    if isinstance(f_spec, int):
        raw = [f_spec]
    elif isinstance(f_spec, (list, tuple)):
        raw = [int(v) for v in f_spec]
    else:
        return []

    resolved: List[int] = []
    for ridx in raw:
        aidx = current_idx + ridx if ridx < 0 else ridx
        if 0 <= aidx < n_layers:
            resolved.append(int(aidx))
    return resolved


def _infer_neck_pre_detect_indices(yolo_runtime: Any, detect_module: nn.Module) -> List[int]:
    """
    Infer one-block-earlier neck indices for each Detect input branch.

    For each detect source index `s`, we resolve the producer module's own
    inputs (`layers[s].f`) and take the first resolved input as the "earlier"
    hook point. If it cannot be resolved, we fall back to `s`.
    """
    layers = _get_yolo_layer_list(yolo_runtime)
    n_layers = len(layers)
    detect_idx = next((i for i, m in enumerate(layers) if m is detect_module), -1)
    if detect_idx < 0:
        raise RuntimeError("Could not locate Detect module index in model graph.")

    detect_sources = _resolve_graph_indices(getattr(detect_module, "f", []), detect_idx, n_layers)
    if not detect_sources:
        raise RuntimeError("Could not resolve Detect source indices from `Detect.f`.")

    target_indices: List[int] = []
    for src_idx in detect_sources:
        src_module = layers[src_idx]
        earlier = _resolve_graph_indices(getattr(src_module, "f", -1), src_idx, n_layers)
        if earlier:
            target_indices.append(int(earlier[0]))
        else:
            target_indices.append(int(src_idx))

    if len(target_indices) != len(detect_sources):
        raise RuntimeError("Failed inferring neck pre-detect hook indices.")
    return target_indices


def _infer_module_output_channels(
    yolo_runtime: Any,
    module_indices: Sequence[int],
    device: str,
    imgsz: int,
) -> List[int]:
    """
    Infer output channels for selected YOLO graph modules using a dry forward.
    """
    model = getattr(yolo_runtime, "model", None)
    if not isinstance(model, nn.Module):
        raise RuntimeError("Channel inference requires a PyTorch YOLO backend.")
    layers = _get_yolo_layer_list(yolo_runtime)
    n_layers = len(layers)
    indices = [int(i) for i in module_indices]
    for i in indices:
        if i < 0 or i >= n_layers:
            raise RuntimeError(f"Module index out of range for channel inference: {i}")

    outputs: List[Optional[int]] = [None for _ in indices]
    handles: List[RemovableHandle] = []

    def _make_hook(slot: int):
        def _hook(_module: nn.Module, _inp: Tuple[Any, ...], out: Any):
            t: Optional[Tensor] = None
            if isinstance(out, torch.Tensor):
                t = out
            elif isinstance(out, (list, tuple)) and len(out) == 1 and isinstance(out[0], torch.Tensor):
                t = out[0]
            if t is not None and t.ndim >= 2:
                outputs[slot] = int(t.shape[1])
        return _hook

    for slot, idx in enumerate(indices):
        handles.append(layers[idx].register_forward_hook(_make_hook(slot)))

    try:
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros((1, 3, int(imgsz), int(imgsz)), device=device, dtype=torch.float32)
            _ = model(dummy)
    finally:
        for h in handles:
            h.remove()

    channels: List[int] = []
    for idx, ch in zip(indices, outputs):
        if ch is None:
            raise RuntimeError(f"Could not infer output channels for module index {idx}.")
        channels.append(int(ch))
    return channels


def _resolve_rectiq_target_modules(
    yolo_runtime: Any,
    target_indices: Sequence[int],
) -> List[nn.Module]:
    """Resolve YOLO graph module indices to module objects."""
    layers = _get_yolo_layer_list(yolo_runtime)
    modules: List[nn.Module] = []
    for idx in target_indices:
        i = int(idx)
        if i < 0 or i >= len(layers):
            raise RuntimeError(f"Invalid Recti-Q target module index: {i}")
        modules.append(layers[i])
    return modules


def _freeze_module(module: nn.Module) -> None:
    """
    Freeze all parameters in a module.
    """
    for p in module.parameters():
        p.requires_grad_(False)


def _fixed_int8_qdq(x: Tensor, scale: float, use_ste: bool = True) -> Tensor:
    """
    Apply fixed symmetric INT8 quantize-dequantize.
    """
    s = float(max(scale, 1e-8))
    q = torch.clamp(torch.round(x / s), -128, 127)
    dq = q * s
    if use_ste:
        # Keep forward as quantized value while preserving gradients.
        return x + (dq - x).detach()
    return dq


class FixedInt8DetectInputQuantizer:
    """
    Fixed INT8 Q/DQ hook for YOLO Detect input feature maps.

    This is PTQ-style quantization: scales are pre-calibrated and frozen.
    """

    def __init__(self, scales: Sequence[float], use_ste: bool = True):
        self.scales = [float(max(s, 1e-8)) for s in scales]
        self.use_ste = bool(use_ste)
        self._handle: Optional[RemovableHandle] = None
        self._detect_module: Optional[nn.Module] = None

    def attach(self, detect_module: nn.Module) -> None:
        if self._handle is not None:
            return
        self._detect_module = detect_module
        self._handle = detect_module.register_forward_pre_hook(self._pre_hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._detect_module = None

    def _pre_hook(self, _module: nn.Module, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
        if not inputs:
            return inputs

        feats = inputs[0]
        if isinstance(feats, tuple):
            feats = list(feats)
        elif not isinstance(feats, list):
            raise RuntimeError("Detect pre-hook expected a list/tuple of feature maps.")

        if len(feats) != len(self.scales):
            raise RuntimeError(
                f"Detect features/scales mismatch: got {len(feats)} features, "
                f"but quantizer has {len(self.scales)} scales."
            )

        quantized: List[Tensor] = []
        for feat, scale in zip(feats, self.scales):
            quantized.append(_fixed_int8_qdq(feat, scale=scale, use_ste=self.use_ste))

        return (quantized,) + tuple(inputs[1:])


def calibrate_detect_input_ptq_scales(
    model_like: Any,
    source_loader: DataLoader,
    device: str,
    imgsz: int,
    max_batches: Optional[int] = 50,
) -> Dict[str, Any]:
    """
    Calibrate fixed symmetric INT8 scales for YOLO Detect input features.
    """
    yolo_runtime = _resolve_yolo_runtime(model_like)
    model = getattr(yolo_runtime, "model", None)
    if not isinstance(model, nn.Module):
        raise RuntimeError(
            "PTQ scale calibration requires a trainable PyTorch YOLO backend."
        )
    # Ensure calibration forward uses same device for inputs and model weights.
    # This avoids CUDA-input vs CPU-weights mismatch when the surrogate is
    # freshly loaded and has not yet been moved to `device`.
    model = model.to(device)

    detect_module = _find_detect_module(yolo_runtime)
    n_features = len(_infer_detect_input_channels(detect_module))
    absmax = [0.0 for _ in range(n_features)]

    def _calib_hook(_module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        if not inputs:
            return
        feats = inputs[0]
        if isinstance(feats, tuple):
            feats = list(feats)
        if not isinstance(feats, list):
            return
        if len(feats) != len(absmax):
            return
        for i, feat in enumerate(feats):
            cur = float(feat.detach().abs().max().item())
            if cur > absmax[i]:
                absmax[i] = cur

    handle = detect_module.register_forward_pre_hook(_calib_hook)
    n_seen_batches = 0
    try:
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, _targets) in enumerate(source_loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                batch = _as_bchw(images=images, device=device, imgsz=imgsz)
                _ = model(batch)
                n_seen_batches += 1
    finally:
        handle.remove()

    scales = [max(v / 127.0, 1e-8) for v in absmax]
    return {
        "scales": scales,
        "absmax": absmax,
        "num_features": n_features,
        "num_batches": n_seen_batches,
        "quant_min": -128,
        "quant_max": 127,
    }


def attach_fixed_int8_detect_input_quantizer(
    model_like: Any,
    scales: Sequence[float],
    use_ste: bool = True,
) -> FixedInt8DetectInputQuantizer:
    """
    Attach fixed PTQ Q/DQ hook to YOLO Detect input features.
    """
    yolo_runtime = _resolve_yolo_runtime(model_like)
    detect_module = _find_detect_module(yolo_runtime)
    quantizer = FixedInt8DetectInputQuantizer(scales=scales, use_ste=use_ste)
    quantizer.attach(detect_module)
    return quantizer


def _as_bchw(
    images: Sequence[Any],
    device: str,
    imgsz: int,
) -> Tensor:
    """
    Convert a batch of images to BCHW float tensor in [0, 1] and resize.
    """
    tensors: List[Tensor] = []
    for img in images:
        if isinstance(img, np.ndarray):
            t = torch.from_numpy(img).permute(2, 0, 1).float()
        elif isinstance(img, torch.Tensor):
            t = img.float()
            if t.dim() == 3 and t.shape[0] != 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        if t.max() > 1.0:
            t = t / 255.0

        t = t.unsqueeze(0)
        if t.shape[-2:] != (imgsz, imgsz):
            t = F.interpolate(t, size=(imgsz, imgsz), mode="bilinear", align_corners=False)
        tensors.append(t.squeeze(0))

    return torch.stack(tensors, dim=0).to(device, non_blocking=True)


def _image_hw(image: Any) -> Tuple[int, int]:
    """Infer (height, width) from numpy HWC or tensor CHW/HWC image."""
    if isinstance(image, np.ndarray):
        if image.ndim < 2:
            raise RuntimeError(f"Expected HWC numpy image, got shape={image.shape}")
        return int(image.shape[0]), int(image.shape[1])

    if isinstance(image, torch.Tensor):
        if image.dim() != 3:
            raise RuntimeError(f"Expected 3D tensor image, got shape={tuple(image.shape)}")
        if image.shape[0] in {1, 3}:  # CHW
            return int(image.shape[1]), int(image.shape[2])
        return int(image.shape[0]), int(image.shape[1])  # HWC

    raise TypeError(f"Unsupported image type for shape inference: {type(image)}")


def _normalize_xyxy_to_xywh(
    boxes_xyxy: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """Convert absolute xyxy boxes to normalized xywh in [0, 1]."""
    boxes = boxes_xyxy.float().clone()
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    x1 = boxes[:, 0].clamp(0.0, float(width))
    y1 = boxes[:, 1].clamp(0.0, float(height))
    x2 = boxes[:, 2].clamp(0.0, float(width))
    y2 = boxes[:, 3].clamp(0.0, float(height))

    xc = ((x1 + x2) * 0.5) / max(float(width), 1.0)
    yc = ((y1 + y2) * 0.5) / max(float(height), 1.0)
    bw = (x2 - x1).clamp(min=0.0) / max(float(width), 1.0)
    bh = (y2 - y1).clamp(min=0.0) / max(float(height), 1.0)

    out = torch.stack([xc, yc, bw, bh], dim=1)
    return out.clamp_(0.0, 1.0)


def _build_ultralytics_loss_batch(
    raw_images: Sequence[Any],
    targets: Sequence[Dict[str, Any]],
    batch_tensor: Tensor,
    num_classes: int,
) -> Dict[str, Tensor]:
    """
    Build Ultralytics detection-loss batch dict from torchvision-style targets.

    Expected output keys:
      - img: BCHW tensor
      - batch_idx: [N, 1]
      - cls: [N, 1] class indices in [0, num_classes)
      - bboxes: [N, 4] normalized xywh
    """
    batch_idx_list: List[Tensor] = []
    cls_list: List[Tensor] = []
    bbox_list: List[Tensor] = []

    for image_idx, (image, target) in enumerate(zip(raw_images, targets)):
        boxes = target.get("boxes")
        labels = target.get("labels")
        if boxes is None or labels is None:
            continue
        if not isinstance(boxes, torch.Tensor) or not isinstance(labels, torch.Tensor):
            continue
        if boxes.numel() == 0 or labels.numel() == 0:
            continue

        labels_cpu = labels.detach().cpu().long()
        if labels_cpu.min().item() < 0 or labels_cpu.max().item() >= int(num_classes):
            raise RuntimeError(
                "Recti-Q detection task loss expects contiguous class labels in "
                f"[0, {num_classes - 1}]. Got range="
                f"[{int(labels_cpu.min().item())}, {int(labels_cpu.max().item())}]."
            )

        height, width = _image_hw(image)
        boxes_xywh = _normalize_xyxy_to_xywh(
            boxes.detach().cpu(),
            height=height,
            width=width,
        )
        if boxes_xywh.numel() == 0:
            continue

        n = int(boxes_xywh.shape[0])
        batch_idx_list.append(torch.full((n, 1), float(image_idx), dtype=torch.float32))
        cls_list.append(labels_cpu.float().view(-1, 1))
        bbox_list.append(boxes_xywh.float())

    if bbox_list:
        batch_idx = torch.cat(batch_idx_list, dim=0)
        cls = torch.cat(cls_list, dim=0)
        bboxes = torch.cat(bbox_list, dim=0)
    else:
        batch_idx = torch.zeros((0, 1), dtype=torch.float32)
        cls = torch.zeros((0, 1), dtype=torch.float32)
        bboxes = torch.zeros((0, 4), dtype=torch.float32)

    return {
        "img": batch_tensor,
        "batch_idx": batch_idx.to(batch_tensor.device, non_blocking=True),
        "cls": cls.to(batch_tensor.device, non_blocking=True),
        "bboxes": bboxes.to(batch_tensor.device, non_blocking=True),
    }


def _detection_task_loss_from_ultralytics(
    q_model: nn.Module,
    preds: Any,
    raw_images: Sequence[Any],
    targets: Sequence[Dict[str, Any]],
    batch_tensor: Tensor,
) -> Tensor:
    """Compute supervised detection loss using Ultralytics' native criterion."""
    if not hasattr(q_model, "loss"):
        raise RuntimeError("YOLO model does not expose `.loss(...)` for Recti-Q task supervision.")

    detect_head = getattr(q_model, "model", None)
    if isinstance(detect_head, (list, tuple)) and detect_head:
        detect_head = detect_head[-1]
    elif isinstance(detect_head, nn.Module):
        try:
            detect_head = detect_head[-1]
        except Exception:
            pass

    num_classes = int(getattr(detect_head, "nc", 0))
    if num_classes <= 0:
        raise RuntimeError("Could not infer YOLO class count from detection head.")

    loss_batch = _build_ultralytics_loss_batch(
        raw_images=raw_images,
        targets=targets,
        batch_tensor=batch_tensor,
        num_classes=num_classes,
    )
    _ensure_ultralytics_loss_hyp_compat(q_model)
    loss_out = q_model.loss(loss_batch, preds=preds)
    if isinstance(loss_out, tuple):
        total_loss = loss_out[0]
    else:
        total_loss = loss_out
    total_loss = _as_scalar_tensor(total_loss, device=batch_tensor.device)
    batch_size = max(int(batch_tensor.shape[0]), 1)
    return total_loss / float(batch_size)


def _as_per_scale_list(
    value: Union[int, float, Sequence[Union[int, float]]],
    n_scales: int,
    cast_type: type,
    name: str,
) -> List[Union[int, float]]:
    """Normalize scalar/list config values to a per-scale list."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        out = [cast_type(v) for v in value]
        if len(out) != n_scales:
            raise ValueError(
                f"{name} length mismatch: got {len(out)} values for {n_scales} detect scales."
            )
        return out
    return [cast_type(value) for _ in range(n_scales)]


class FeatureLoRA2d(nn.Module):
    """
    LoRA-style residual adapter for one feature map:
      delta = up(dw(down(x))) * (alpha / rank)
    """

    def __init__(
        self,
        channels: int,
        rank: int,
        alpha: float,
        use_dwconv: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1). Got {dropout}.")

        self.channels = int(channels)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.use_dwconv = bool(use_dwconv)

        self.down = nn.Conv2d(self.channels, self.rank, kernel_size=1, bias=False)
        self.dw = (
            nn.Conv2d(self.rank, self.rank, kernel_size=3, stride=1, padding=1, groups=self.rank, bias=False)
            if self.use_dwconv
            else nn.Identity()
        )
        self.dropout = nn.Dropout2d(p=float(dropout)) if dropout > 0.0 else nn.Identity()
        self.up = nn.Conv2d(self.rank, self.channels, kernel_size=1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        if isinstance(self.dw, nn.Conv2d):
            nn.init.kaiming_uniform_(self.dw.weight, a=5 ** 0.5)
        # Zero-init up-proj so initial residual is near zero.
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        y = self.down(x)
        y = self.dw(y)
        y = self.dropout(y)
        return self.up(y) * self.scaling


class RectiQPreDetectAdapter(nn.Module):
    """
    Multi-scale LoRA residual adapter applied before YOLO Detect head.

    This module is attached through a Detect forward-pre-hook so the residual
    is injected as:
      x_i <- x_i + adapter_i(x_i)
    for each detect scale i.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        rank: Union[int, Sequence[int]] = 8,
        alpha: Union[float, Sequence[float]] = 16.0,
        use_dwconv: bool = True,
        dropout: float = 0.0,
        scales: Optional[Sequence[float]] = None,
        use_ste: bool = True,
        requantize_after_residual: bool = True,
    ):
        super().__init__()
        self.in_channels = [int(c) for c in in_channels]
        self.ranks = [int(v) for v in _as_per_scale_list(rank, len(self.in_channels), int, "rank")]
        self.alphas = [float(v) for v in _as_per_scale_list(alpha, len(self.in_channels), float, "alpha")]
        self.rank = self.ranks[0] if len(set(self.ranks)) == 1 else -1
        self.alpha = self.alphas[0] if len(set(self.alphas)) == 1 else -1.0
        self.use_ste = bool(use_ste)
        self.requantize_after_residual = bool(requantize_after_residual)
        self.scales: Optional[List[float]] = None
        if scales is not None:
            self.set_scales(scales)

        self.adapters = nn.ModuleList(
            [
                FeatureLoRA2d(
                    channels=c,
                    rank=r,
                    alpha=a,
                    use_dwconv=bool(use_dwconv),
                    dropout=float(dropout),
                )
                for c, r, a in zip(self.in_channels, self.ranks, self.alphas)
            ]
        )

        self._handle: Optional[RemovableHandle] = None
        self._detect_module_ref: Optional[weakref.ReferenceType[nn.Module]] = None
        self._collect_absmax_enabled = False
        self._collect_absmax: List[float] = [0.0 for _ in self.in_channels]

        # Cached tensors for loss computation.
        self.last_input_features: List[Tensor] = []
        self.last_rectified_features: List[Tensor] = []
        self.last_residuals: List[Tensor] = []

    def clear_cache(self) -> None:
        self.last_input_features = []
        self.last_rectified_features = []
        self.last_residuals = []

    def set_scales(self, scales: Sequence[float]) -> None:
        vals = [float(max(s, 1e-8)) for s in scales]
        if len(vals) != len(self.in_channels):
            raise ValueError(
                f"Scale length mismatch: got {len(vals)} scales for {len(self.in_channels)} feature maps."
            )
        self.scales = vals

    def get_scales(self) -> Optional[List[float]]:
        return list(self.scales) if self.scales is not None else None

    def has_scales(self) -> bool:
        return self.scales is not None

    def start_absmax_collection(self) -> None:
        self._collect_absmax_enabled = True
        self._collect_absmax = [0.0 for _ in self.in_channels]

    def finish_absmax_collection(self) -> List[float]:
        self._collect_absmax_enabled = False
        return list(self._collect_absmax)

    def attach(self, detect_module: nn.Module) -> None:
        if self._handle is not None:
            return

        # Keep only a weak reference to avoid creating a module cycle
        # (Detect -> adapter via add_module, adapter -> Detect as strong attr),
        # which breaks state_dict() with recursive traversal.
        self._detect_module_ref = weakref.ref(detect_module)
        detect_module.add_module("_rectiq_adapter", self)
        self._handle = detect_module.register_forward_pre_hook(self._pre_hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        detect_module = self._detect_module_ref() if self._detect_module_ref is not None else None
        if detect_module is not None:
            detect_module._modules.pop("_rectiq_adapter", None)
        self._detect_module_ref = None

    def _pre_hook(
        self,
        _module: nn.Module,
        inputs: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        if not inputs:
            return inputs

        feats = inputs[0]
        if isinstance(feats, tuple):
            feats = list(feats)
        elif not isinstance(feats, list):
            raise RuntimeError("Detect pre-hook expected a list/tuple of feature maps.")

        if len(feats) != len(self.adapters):
            raise RuntimeError(
                f"Detect features/adapters mismatch: got {len(feats)} features, "
                f"but adapter has {len(self.adapters)} branches."
            )

        self.last_input_features = [f.detach() for f in feats]
        self.last_rectified_features = []
        self.last_residuals = []

        rectified: List[Tensor] = []
        for idx, (feat, adapter) in enumerate(zip(feats, self.adapters)):
            x = feat
            if self.scales is not None:
                x = _fixed_int8_qdq(x, scale=self.scales[idx], use_ste=self.use_ste)

            residual = adapter(x)
            rect = x + residual
            if self._collect_absmax_enabled:
                cur = float(rect.detach().abs().max().item())
                if cur > self._collect_absmax[idx]:
                    self._collect_absmax[idx] = cur

            if self.scales is not None and self.requantize_after_residual:
                rect = _fixed_int8_qdq(rect, scale=self.scales[idx], use_ste=self.use_ste)

            self.last_residuals.append(residual)
            self.last_rectified_features.append(rect)
            rectified.append(rect)

        return (rectified,) + tuple(inputs[1:])


class RectiQModuleOutputAdapter(nn.Module):
    """
    Multi-scale LoRA residual adapter applied at selected YOLO module outputs.

    This variant is used for "one block earlier" correction inside the neck.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        target_module_indices: Sequence[int],
        rank: Union[int, Sequence[int]] = 8,
        alpha: Union[float, Sequence[float]] = 16.0,
        use_dwconv: bool = True,
        dropout: float = 0.0,
        scales: Optional[Sequence[float]] = None,
        use_ste: bool = True,
        requantize_after_residual: bool = True,
    ):
        super().__init__()
        self.in_channels = [int(c) for c in in_channels]
        self.target_module_indices = [int(i) for i in target_module_indices]
        if len(self.in_channels) != len(self.target_module_indices):
            raise ValueError(
                "RectiQModuleOutputAdapter mismatch: "
                f"in_channels={len(self.in_channels)} vs targets={len(self.target_module_indices)}."
            )

        self.ranks = [int(v) for v in _as_per_scale_list(rank, len(self.in_channels), int, "rank")]
        self.alphas = [float(v) for v in _as_per_scale_list(alpha, len(self.in_channels), float, "alpha")]
        self.rank = self.ranks[0] if len(set(self.ranks)) == 1 else -1
        self.alpha = self.alphas[0] if len(set(self.alphas)) == 1 else -1.0
        self.use_ste = bool(use_ste)
        self.requantize_after_residual = bool(requantize_after_residual)
        self.scales: Optional[List[float]] = None
        if scales is not None:
            self.set_scales(scales)

        self.adapters = nn.ModuleList(
            [
                FeatureLoRA2d(
                    channels=c,
                    rank=r,
                    alpha=a,
                    use_dwconv=bool(use_dwconv),
                    dropout=float(dropout),
                )
                for c, r, a in zip(self.in_channels, self.ranks, self.alphas)
            ]
        )

        self._handles: List[RemovableHandle] = []
        self._target_refs: List[weakref.ReferenceType[nn.Module]] = []
        self._collect_absmax_enabled = False
        self._collect_absmax: List[float] = [0.0 for _ in self.in_channels]

        self._last_input_slots: List[Optional[Tensor]] = [None for _ in self.in_channels]
        self._last_rect_slots: List[Optional[Tensor]] = [None for _ in self.in_channels]
        self._last_residual_slots: List[Optional[Tensor]] = [None for _ in self.in_channels]
        self.last_input_features: List[Tensor] = []
        self.last_rectified_features: List[Tensor] = []
        self.last_residuals: List[Tensor] = []

    def clear_cache(self) -> None:
        self._last_input_slots = [None for _ in self.in_channels]
        self._last_rect_slots = [None for _ in self.in_channels]
        self._last_residual_slots = [None for _ in self.in_channels]
        self.last_input_features = []
        self.last_rectified_features = []
        self.last_residuals = []

    def set_scales(self, scales: Sequence[float]) -> None:
        vals = [float(max(s, 1e-8)) for s in scales]
        if len(vals) != len(self.in_channels):
            raise ValueError(
                f"Scale length mismatch: got {len(vals)} scales for {len(self.in_channels)} feature maps."
            )
        self.scales = vals

    def get_scales(self) -> Optional[List[float]]:
        return list(self.scales) if self.scales is not None else None

    def has_scales(self) -> bool:
        return self.scales is not None

    def start_absmax_collection(self) -> None:
        self._collect_absmax_enabled = True
        self._collect_absmax = [0.0 for _ in self.in_channels]

    def finish_absmax_collection(self) -> List[float]:
        self._collect_absmax_enabled = False
        return list(self._collect_absmax)

    def attach(self, target_modules: Sequence[nn.Module]) -> None:
        if self._handles:
            return
        if len(target_modules) != len(self.adapters):
            raise RuntimeError(
                f"Target/adapters mismatch: {len(target_modules)} modules vs {len(self.adapters)} adapters."
            )

        self._target_refs = [weakref.ref(m) for m in target_modules]
        for i, module in enumerate(target_modules):
            self._handles.append(module.register_forward_hook(self._make_hook(i)))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._target_refs = []

    def _make_hook(self, idx: int):
        def _hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any):
            out_tensor: Optional[Tensor] = None
            wrap_kind = "tensor"
            if isinstance(output, torch.Tensor):
                out_tensor = output
            elif isinstance(output, tuple) and len(output) == 1 and isinstance(output[0], torch.Tensor):
                out_tensor = output[0]
                wrap_kind = "tuple1"
            elif isinstance(output, list) and len(output) == 1 and isinstance(output[0], torch.Tensor):
                out_tensor = output[0]
                wrap_kind = "list1"
            else:
                return output

            x = out_tensor
            if self.scales is not None:
                x = _fixed_int8_qdq(x, scale=self.scales[idx], use_ste=self.use_ste)

            residual = self.adapters[idx](x)
            rect = x + residual
            if self._collect_absmax_enabled:
                cur = float(rect.detach().abs().max().item())
                if cur > self._collect_absmax[idx]:
                    self._collect_absmax[idx] = cur

            if self.scales is not None and self.requantize_after_residual:
                rect = _fixed_int8_qdq(rect, scale=self.scales[idx], use_ste=self.use_ste)

            self._last_input_slots[idx] = x.detach()
            self._last_rect_slots[idx] = rect
            self._last_residual_slots[idx] = residual
            self.last_input_features = [t for t in self._last_input_slots if t is not None]
            self.last_rectified_features = [t for t in self._last_rect_slots if t is not None]
            self.last_residuals = [t for t in self._last_residual_slots if t is not None]

            if wrap_kind == "tuple1":
                return (rect,)
            if wrap_kind == "list1":
                return [rect]
            return rect

        return _hook


class DetectInputFeatureTap:
    """
    Capture teacher pre-detect features via forward-pre-hook.
    """

    def __init__(self, detect_module: nn.Module):
        self.features: List[Tensor] = []
        self._handle = detect_module.register_forward_pre_hook(self._pre_hook)

    def clear(self) -> None:
        self.features = []

    def _pre_hook(self, _module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        if not inputs:
            self.features = []
            return
        feats = inputs[0]
        if isinstance(feats, tuple):
            feats = list(feats)
        if not isinstance(feats, list):
            self.features = []
            return
        self.features = [f.detach() for f in feats]

    def remove(self) -> None:
        self._handle.remove()


class ModuleOutputFeatureTap:
    """
    Capture selected module outputs (one tensor per target module).
    """

    def __init__(self, target_modules: Sequence[nn.Module]):
        self.features: List[Tensor] = []
        self._slots: List[Optional[Tensor]] = [None for _ in target_modules]
        self._handles: List[RemovableHandle] = []
        for i, module in enumerate(target_modules):
            self._handles.append(module.register_forward_hook(self._make_hook(i)))

    def clear(self) -> None:
        self.features = []
        self._slots = [None for _ in self._slots]

    def _make_hook(self, idx: int):
        def _hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any):
            t: Optional[Tensor] = None
            if isinstance(output, torch.Tensor):
                t = output
            elif isinstance(output, (tuple, list)) and len(output) == 1 and isinstance(output[0], torch.Tensor):
                t = output[0]
            if t is None:
                return output
            self._slots[idx] = t.detach()
            self.features = [v for v in self._slots if v is not None]
            return output

        return _hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def _feature_alignment_loss(student_feats: List[Tensor], teacher_feats: List[Tensor]) -> Tensor:
    if not student_feats or not teacher_feats:
        raise RuntimeError("Missing student/teacher feature caches for alignment loss.")
    if len(student_feats) != len(teacher_feats):
        raise RuntimeError(
            f"Feature branch mismatch: student={len(student_feats)}, teacher={len(teacher_feats)}"
        )

    loss = student_feats[0].new_tensor(0.0)
    for s_feat, t_feat in zip(student_feats, teacher_feats):
        if s_feat.shape != t_feat.shape:
            raise RuntimeError(
                f"Feature shape mismatch for KD: student={tuple(s_feat.shape)} "
                f"teacher={tuple(t_feat.shape)}"
            )
        loss = loss + F.mse_loss(s_feat, t_feat)
    return loss / len(student_feats)


def _residual_regularization(residuals: List[Tensor]) -> Tensor:
    if not residuals:
        return torch.tensor(0.0)
    reg = residuals[0].new_tensor(0.0)
    for r in residuals:
        reg = reg + r.pow(2).mean()
    return reg / len(residuals)


@dataclass
class RectiQConfig:
    rank: int = 8
    rank_per_scale: Optional[List[int]] = None
    alpha: float = 16.0
    alpha_per_scale: Optional[List[float]] = None
    imgsz: int = 640
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    feature_kd_weight: float = 1.0
    residual_reg_weight: float = 1e-4
    task_loss_weight: float = 1.0
    # Where to attach adapters:
    # - "detect_input": current detect-input features
    # - "neck_pre_detect": one block earlier (producers of detect-input producers)
    adapter_target: str = "detect_input"
    # Adapter architecture / quantized interface behavior.
    adapter_use_dwconv: bool = True
    adapter_dropout: float = 0.0
    requantize_after_residual: bool = True
    ptq_scales: Optional[List[float]] = None
    ptq_use_ste: bool = True
    # Optional light head adaptation.
    unfreeze_detect_cv3: bool = False
    cv3_lr: Optional[float] = None
    # Optional scale refresh after warmup.
    recalibration_epoch: Optional[int] = None
    recalibration_batches: Optional[int] = None
    max_batches_per_epoch: Optional[int] = None
    val_final_only: bool = False


@dataclass
class RectiQTrainResult:
    adapter: nn.Module
    best_state_dict: Dict[str, Tensor]
    best_epoch: int
    best_val_loss: float
    history: List[Dict[str, float]] = field(default_factory=list)


def attach_rectiq_adapter(
    quantized_model: Any,
    rank: Union[int, Sequence[int]] = 8,
    alpha: Union[float, Sequence[float]] = 16.0,
    adapter_target: str = "detect_input",
    imgsz: int = 640,
    device: str = "cuda",
    use_dwconv: bool = True,
    dropout: float = 0.0,
    scales: Optional[Sequence[float]] = None,
    use_ste: bool = True,
    requantize_after_residual: bool = True,
) -> nn.Module:
    """
    Attach Recti-Q adapter to a quantized YOLO model and return the adapter.
    """
    q_yolo = _resolve_yolo_runtime(quantized_model)
    detect_module = _find_detect_module(q_yolo)
    target = str(adapter_target).strip().lower()
    if target not in {"detect_input", "neck_pre_detect"}:
        raise ValueError(
            f"Unknown Recti-Q adapter_target='{adapter_target}'. "
            "Use one of: detect_input, neck_pre_detect."
        )

    if target == "detect_input":
        in_channels = _infer_detect_input_channels(detect_module)
        adapter = RectiQPreDetectAdapter(
            in_channels=in_channels,
            rank=rank,
            alpha=alpha,
            use_dwconv=use_dwconv,
            dropout=dropout,
            scales=scales,
            use_ste=use_ste,
            requantize_after_residual=requantize_after_residual,
        )
        adapter.attach(detect_module)
        setattr(adapter, "adapter_target", target)
        return adapter

    target_indices = _infer_neck_pre_detect_indices(q_yolo, detect_module)
    target_modules = _resolve_rectiq_target_modules(q_yolo, target_indices)
    in_channels = _infer_module_output_channels(
        yolo_runtime=q_yolo,
        module_indices=target_indices,
        device=device,
        imgsz=imgsz,
    )
    adapter = RectiQModuleOutputAdapter(
        in_channels=in_channels,
        target_module_indices=target_indices,
        rank=rank,
        alpha=alpha,
        use_dwconv=use_dwconv,
        dropout=dropout,
        scales=scales,
        use_ste=use_ste,
        requantize_after_residual=requantize_after_residual,
    )
    adapter.attach(target_modules)
    setattr(adapter, "adapter_target", target)
    return adapter


def _enable_detect_cv3_training(yolo_runtime: Any) -> Tuple[List[nn.Module], List[nn.Parameter]]:
    """
    Unfreeze YOLO Detect classification branch (`cv3`) and return train modules/params.
    """
    detect_module = _find_detect_module(yolo_runtime)
    cv3 = getattr(detect_module, "cv3", None)
    if cv3 is None:
        return [], []
    modules = [cv3] if isinstance(cv3, nn.Module) else []
    params: List[nn.Parameter] = []
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(True)
            params.append(p)
    return modules, params


def recalibrate_rectiq_output_scales(
    model_like: Any,
    adapter: nn.Module,
    source_loader: DataLoader,
    device: str,
    imgsz: int,
    max_batches: Optional[int] = 50,
) -> Dict[str, Any]:
    """
    Re-estimate per-scale quantization scales from adapter-corrected features.
    """
    yolo_runtime = _resolve_yolo_runtime(model_like)
    model = getattr(yolo_runtime, "model", None)
    if not isinstance(model, nn.Module):
        raise RuntimeError("Scale recalibration requires a trainable PyTorch YOLO backend.")

    model = model.to(device)
    model.eval()
    adapter.start_absmax_collection()
    n_seen_batches = 0
    try:
        with torch.no_grad():
            for batch_idx, (images, _targets) in enumerate(source_loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                batch = _as_bchw(images=images, device=device, imgsz=imgsz)
                _ = model(batch)
                n_seen_batches += 1
    finally:
        absmax = adapter.finish_absmax_collection()

    scales = [max(v / 127.0, 1e-8) for v in absmax]
    adapter.set_scales(scales)
    return {
        "scales": scales,
        "absmax": absmax,
        "num_features": len(scales),
        "num_batches": n_seen_batches,
        "quant_min": -128,
        "quant_max": 127,
    }


def train_rectiq_adapter(
    quantized_model: Any,
    source_loader: DataLoader,
    device: str,
    config: Optional[RectiQConfig] = None,
    teacher_model: Optional[Any] = None,
    val_loader: Optional[DataLoader] = None,
    task_loss_fn: Optional[Callable[..., Tensor]] = None,
) -> RectiQTrainResult:
    """
    Train a Recti-Q adapter on top of a frozen quantized YOLO model.

    Args:
        quantized_model:
            Quantized YOLO model handle (qda wrapper or raw ultralytics YOLO).
        source_loader:
            Training loader yielding `(images, targets)`.
        device:
            Device string (`cuda`, `cuda:0`, `cpu`).
        config:
            Recti-Q hyper-parameters.
        teacher_model:
            Optional frozen teacher for feature-KD.
        val_loader:
            Optional validation loader for model selection.
        task_loss_fn:
            Optional callable for supervised detection loss. It receives:
                `(student_outputs, targets, quantized_yolo_runtime)`
            and returns a scalar tensor.

            If omitted and `task_loss_weight > 0`, native YOLO detection loss
            is used as supervised task loss.

    Returns:
        RectiQTrainResult with best adapter state and training history.
    """
    cfg = config or RectiQConfig()
    q_yolo = _resolve_yolo_runtime(quantized_model)
    q_model_raw = getattr(q_yolo, "model", None)
    if not isinstance(q_model_raw, nn.Module):
        raise ValueError(
            "Recti-Q training requires a trainable PyTorch YOLO model backend. "
            f"Got backend model type: {type(q_model_raw)}"
        )
    q_model = q_model_raw.to(device)
    q_model.eval()
    _freeze_module(q_model)

    rank_value: Union[int, Sequence[int]] = (
        cfg.rank_per_scale if cfg.rank_per_scale is not None else cfg.rank
    )
    alpha_value: Union[float, Sequence[float]] = (
        cfg.alpha_per_scale if cfg.alpha_per_scale is not None else cfg.alpha
    )
    adapter = attach_rectiq_adapter(
        quantized_model=quantized_model,
        rank=rank_value,
        alpha=alpha_value,
        adapter_target=str(getattr(cfg, "adapter_target", "detect_input")),
        imgsz=int(cfg.imgsz),
        device=device,
        use_dwconv=bool(cfg.adapter_use_dwconv),
        dropout=float(cfg.adapter_dropout),
        scales=cfg.ptq_scales,
        use_ste=bool(cfg.ptq_use_ste),
        requantize_after_residual=bool(cfg.requantize_after_residual),
    )
    # Adapter is created after q_model.to(device), so move it explicitly.
    adapter = adapter.to(device)
    for p in adapter.parameters():
        p.requires_grad_(True)

    extra_train_modules: List[nn.Module] = []
    extra_train_params: List[nn.Parameter] = []
    if bool(cfg.unfreeze_detect_cv3):
        extra_train_modules, extra_train_params = _enable_detect_cv3_training(q_yolo)

    use_teacher_kd = teacher_model is not None and float(cfg.feature_kd_weight) > 0.0

    teacher_tap: Optional[Any] = None
    t_model: Optional[nn.Module] = None
    if use_teacher_kd:
        t_yolo = _resolve_yolo_runtime(teacher_model)
        t_model_raw = getattr(t_yolo, "model", None)
        if not isinstance(t_model_raw, nn.Module):
            raise ValueError(
                "Teacher model for Recti-Q must expose a trainable PyTorch backend. "
                f"Got backend model type: {type(t_model_raw)}"
            )
        t_model = t_model_raw.to(device)
        t_model.eval()
        _freeze_module(t_model)
        if str(getattr(cfg, "adapter_target", "detect_input")).strip().lower() == "detect_input":
            teacher_tap = DetectInputFeatureTap(_find_detect_module(t_yolo))
        else:
            target_indices = list(getattr(adapter, "target_module_indices", []))
            teacher_target_modules = _resolve_rectiq_target_modules(t_yolo, target_indices)
            teacher_tap = ModuleOutputFeatureTap(teacher_target_modules)

    has_task_signal = task_loss_fn is not None or float(cfg.task_loss_weight) > 0.0
    if not use_teacher_kd and not has_task_signal:
        raise ValueError(
            "At least one training signal is required: either "
            "(1) teacher + feature_kd_weight > 0, or "
            "(2) task_loss_fn, or "
            "(3) task_loss_weight > 0 for native YOLO detection loss."
        )

    optimizer_groups: List[Dict[str, Any]] = [
        {"params": list(adapter.parameters()), "lr": float(cfg.lr)}
    ]
    if extra_train_params:
        optimizer_groups.append(
            {
                "params": extra_train_params,
                "lr": float(cfg.cv3_lr if cfg.cv3_lr is not None else cfg.lr),
            }
        )
    optimizer = AdamW(optimizer_groups, weight_decay=cfg.weight_decay)
    history: List[Dict[str, float]] = []
    best_state: Dict[str, Tensor] = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    best_epoch = 0
    best_val_loss = float("inf")

    def _run_epoch(loader: DataLoader, training: bool) -> Dict[str, float]:
        if training:
            adapter.train()
            for module in extra_train_modules:
                module.train()
        else:
            adapter.eval()
            for module in extra_train_modules:
                module.eval()

        total_loss = 0.0
        total_feat = 0.0
        total_task = 0.0
        total_reg = 0.0
        n_steps = 0

        it = tqdm(loader, desc="RectiQ train" if training else "RectiQ val", leave=False)
        for batch_idx, (images, targets) in enumerate(it):
            if cfg.max_batches_per_epoch is not None and batch_idx >= cfg.max_batches_per_epoch:
                break

            batch = _as_bchw(images=images, device=device, imgsz=cfg.imgsz)
            adapter.clear_cache()

            with torch.no_grad():
                if t_model is not None and teacher_tap is not None:
                    if hasattr(teacher_tap, "clear"):
                        teacher_tap.clear()
                    _ = t_model(batch)

            if training:
                optimizer.zero_grad(set_to_none=True)

            student_outputs = q_model(batch)

            feat_loss = batch.new_tensor(0.0)
            if teacher_tap is not None:
                feat_loss = _feature_alignment_loss(adapter.last_rectified_features, teacher_tap.features)
            feat_loss = _as_scalar_tensor(feat_loss, device=batch.device)

            task_loss = batch.new_tensor(0.0)
            if task_loss_fn is not None:
                try:
                    task_loss = task_loss_fn(
                        student_outputs=student_outputs,
                        targets=targets,
                        quantized_yolo_runtime=q_yolo,
                        batch_tensor=batch,
                        raw_images=images,
                    )
                except TypeError:
                    # Backward-compatible callback signature:
                    # (student_outputs, targets, quantized_yolo_runtime)
                    task_loss = task_loss_fn(student_outputs, targets, q_yolo)
                if not isinstance(task_loss, torch.Tensor):
                    task_loss = torch.as_tensor(float(task_loss), device=batch.device)
            elif float(cfg.task_loss_weight) > 0.0:
                task_loss = _detection_task_loss_from_ultralytics(
                    q_model=q_model,
                    preds=student_outputs,
                    raw_images=images,
                    targets=targets,
                    batch_tensor=batch,
                )
            task_loss = _as_scalar_tensor(task_loss, device=batch.device)

            reg_loss = _residual_regularization(adapter.last_residuals).to(batch.device)
            reg_loss = _as_scalar_tensor(reg_loss, device=batch.device)
            total = (
                cfg.feature_kd_weight * feat_loss
                + cfg.task_loss_weight * task_loss
                + cfg.residual_reg_weight * reg_loss
            )
            total = _as_scalar_tensor(total, device=batch.device)

            if training:
                total.backward()
                optimizer.step()

            total_loss += float(total.item())
            total_feat += float(feat_loss.item())
            total_task += float(task_loss.item())
            total_reg += float(reg_loss.item())
            n_steps += 1

        denom = max(n_steps, 1)
        return {
            "loss": total_loss / denom,
            "feat_loss": total_feat / denom,
            "task_loss": total_task / denom,
            "reg_loss": total_reg / denom,
        }

    try:
        for epoch in range(1, cfg.epochs + 1):
            train_stats = _run_epoch(source_loader, training=True)
            if (
                cfg.recalibration_epoch is not None
                and int(cfg.recalibration_epoch) > 0
                and epoch == int(cfg.recalibration_epoch)
                and adapter.has_scales()
            ):
                recalib_batches = (
                    int(cfg.recalibration_batches)
                    if cfg.recalibration_batches is not None
                    else 50
                )
                _ = recalibrate_rectiq_output_scales(
                    model_like=quantized_model,
                    adapter=adapter,
                    source_loader=source_loader,
                    device=device,
                    imgsz=cfg.imgsz,
                    max_batches=recalib_batches,
                )
            use_val_this_epoch = (
                val_loader is not None
                and (not cfg.val_final_only or epoch == cfg.epochs)
            )
            val_stats = _run_epoch(val_loader, training=False) if use_val_this_epoch else train_stats

            row = {
                "epoch": float(epoch),
                "train_loss": train_stats["loss"],
                "train_feat_loss": train_stats["feat_loss"],
                "train_task_loss": train_stats["task_loss"],
                "train_reg_loss": train_stats["reg_loss"],
                "val_loss": val_stats["loss"],
                "val_feat_loss": val_stats["feat_loss"],
                "val_task_loss": val_stats["task_loss"],
                "val_reg_loss": val_stats["reg_loss"],
            }
            history.append(row)

            should_update_best = use_val_this_epoch or val_loader is None
            if should_update_best and val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    finally:
        if teacher_tap is not None:
            teacher_tap.remove()

    adapter.load_state_dict(best_state, strict=True)
    return RectiQTrainResult(
        adapter=adapter,
        best_state_dict=best_state,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        history=history,
    )


def save_rectiq_adapter(
    adapter: nn.Module,
    save_path: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save Recti-Q adapter state and metadata.
    """
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "adapter_target": str(getattr(adapter, "adapter_target", "detect_input")),
        "rank": adapter.rank,
        "rank_per_scale": list(getattr(adapter, "ranks", [])),
        "alpha": adapter.alpha,
        "alpha_per_scale": list(getattr(adapter, "alphas", [])),
        "in_channels": adapter.in_channels,
        "target_module_indices": list(getattr(adapter, "target_module_indices", [])),
        "ptq_scales": adapter.get_scales(),
        "state_dict": adapter.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, out)
    return out


def load_rectiq_adapter(
    adapter: nn.Module,
    ckpt_path: str | Path,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load Recti-Q adapter checkpoint into an attached adapter instance.
    """
    payload = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    adapter.load_state_dict(state_dict, strict=strict)
    if isinstance(payload, dict) and payload.get("ptq_scales") is not None:
        try:
            adapter.set_scales(payload["ptq_scales"])
        except Exception:
            pass
    return payload if isinstance(payload, dict) else {"state_dict": state_dict}
