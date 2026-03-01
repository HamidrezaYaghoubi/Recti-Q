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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
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


class FeatureLoRA2d(nn.Module):
    """
    LoRA-style residual adapter for one feature map:
      delta = up(down(x)) * (alpha / rank)
    """

    def __init__(self, channels: int, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.channels = int(channels)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)

        self.down = nn.Conv2d(self.channels, self.rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(self.rank, self.channels, kernel_size=1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x)) * self.scaling


class RectiQPreDetectAdapter(nn.Module):
    """
    Multi-scale LoRA residual adapter applied before YOLO Detect head.

    This module is attached through a Detect forward-pre-hook so the residual
    is injected as:
      x_i <- x_i + adapter_i(x_i)
    for each detect scale i.
    """

    def __init__(self, in_channels: Sequence[int], rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.in_channels = [int(c) for c in in_channels]
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.adapters = nn.ModuleList(
            [FeatureLoRA2d(channels=c, rank=self.rank, alpha=self.alpha) for c in self.in_channels]
        )

        self._handle: Optional[RemovableHandle] = None
        self._detect_module_ref: Optional[weakref.ReferenceType[nn.Module]] = None

        # Cached tensors for loss computation.
        self.last_input_features: List[Tensor] = []
        self.last_rectified_features: List[Tensor] = []
        self.last_residuals: List[Tensor] = []

    def clear_cache(self) -> None:
        self.last_input_features = []
        self.last_rectified_features = []
        self.last_residuals = []

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
        for feat, adapter in zip(feats, self.adapters):
            residual = adapter(feat)
            rect = feat + residual
            self.last_residuals.append(residual)
            self.last_rectified_features.append(rect)
            rectified.append(rect)

        return (rectified,) + tuple(inputs[1:])


class DetectInputFeatureTap:
    """
    Capture teacher pre-detect features via forward-pre-hook.
    """

    def __init__(self, detect_module: nn.Module):
        self.features: List[Tensor] = []
        self._handle = detect_module.register_forward_pre_hook(self._pre_hook)

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
    alpha: float = 16.0
    imgsz: int = 640
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    feature_kd_weight: float = 1.0
    residual_reg_weight: float = 1e-4
    task_loss_weight: float = 1.0
    max_batches_per_epoch: Optional[int] = None
    val_final_only: bool = False


@dataclass
class RectiQTrainResult:
    adapter: RectiQPreDetectAdapter
    best_state_dict: Dict[str, Tensor]
    best_epoch: int
    best_val_loss: float
    history: List[Dict[str, float]] = field(default_factory=list)


def attach_rectiq_adapter(
    quantized_model: Any,
    rank: int = 8,
    alpha: float = 16.0,
) -> RectiQPreDetectAdapter:
    """
    Attach Recti-Q adapter to a quantized YOLO model and return the adapter.
    """
    q_yolo = _resolve_yolo_runtime(quantized_model)
    detect_module = _find_detect_module(q_yolo)
    in_channels = _infer_detect_input_channels(detect_module)
    adapter = RectiQPreDetectAdapter(in_channels=in_channels, rank=rank, alpha=alpha)
    adapter.attach(detect_module)
    return adapter


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

    adapter = attach_rectiq_adapter(quantized_model=quantized_model, rank=cfg.rank, alpha=cfg.alpha)
    # Adapter is created after q_model.to(device), so move it explicitly.
    adapter = adapter.to(device)
    for p in adapter.parameters():
        p.requires_grad_(True)

    use_teacher_kd = teacher_model is not None and float(cfg.feature_kd_weight) > 0.0

    teacher_tap: Optional[DetectInputFeatureTap] = None
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
        teacher_tap = DetectInputFeatureTap(_find_detect_module(t_yolo))

    has_task_signal = task_loss_fn is not None or float(cfg.task_loss_weight) > 0.0
    if not use_teacher_kd and not has_task_signal:
        raise ValueError(
            "At least one training signal is required: either "
            "(1) teacher + feature_kd_weight > 0, or "
            "(2) task_loss_fn, or "
            "(3) task_loss_weight > 0 for native YOLO detection loss."
        )

    optimizer = AdamW(adapter.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: List[Dict[str, float]] = []
    best_state: Dict[str, Tensor] = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    best_epoch = 0
    best_val_loss = float("inf")

    def _run_epoch(loader: DataLoader, training: bool) -> Dict[str, float]:
        if training:
            adapter.train()
        else:
            adapter.eval()

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
    adapter: RectiQPreDetectAdapter,
    save_path: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save Recti-Q adapter state and metadata.
    """
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": adapter.rank,
        "alpha": adapter.alpha,
        "in_channels": adapter.in_channels,
        "state_dict": adapter.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, out)
    return out


def load_rectiq_adapter(
    adapter: RectiQPreDetectAdapter,
    ckpt_path: str | Path,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load Recti-Q adapter checkpoint into an attached adapter instance.
    """
    payload = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    adapter.load_state_dict(state_dict, strict=strict)
    return payload if isinstance(payload, dict) else {"state_dict": state_dict}
