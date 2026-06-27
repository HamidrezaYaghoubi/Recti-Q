"""
Recti-Q: LoRA classifier-head adapter on top of a frozen quantized backbone.

Paper method (IROS 2026):
  Let u ∈ R^d be the pre-classifier feature and z_q = f_q(x) ∈ R^C be the
  logits of the frozen quantized backbone. Both come from a single forward pass.

  Adapter:  g_φ(u) = B(A(u)) · (α/r)
    A: Linear(d, r, bias=False)  — kaiming_uniform init
    B: Linear(r, C, bias=False)  — zero init  → g_φ = 0 at start

  Final logits: z = z_q + g_φ(u)

  Training: freeze backbone, train adapter only.
    L = L_CE(z, y) + λ · L_KD
    L_KD = KL(softmax(z/T) ‖ softmax(f_t(x)/T)) · T²
  When λ=0 the teacher forward pass is skipped entirely.

  Ablation: adapter_space="logit" → adapter operates on z_q (A: Linear(C, r)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Backbone forward-split helper
# ---------------------------------------------------------------------------

def _forward_split(backbone: nn.Module, x: Tensor):
    """
    Single forward pass through a timm backbone, returning both the
    pre-classifier feature vector u and the final logits z_q.

    Returns:
        u   : Tensor [B, d]  — pre-logit feature (from forward_head pre_logits=True)
        z_q : Tensor [B, C]  — logits
    """
    feats = backbone.forward_features(x)
    u = backbone.forward_head(feats, pre_logits=True)
    z_q = backbone.forward_head(feats)
    return u, z_q


# ---------------------------------------------------------------------------
# Adapter module
# ---------------------------------------------------------------------------

class ClassifierLoRA(nn.Module):
    """
    Low-rank correction adapter for classifier logits.

        g_φ(u) = B(A(u)) · (α / r)

    A is kaiming-init; B is zero-init so g_φ = 0 at construction.

    Args:
        in_dim     : input feature dimension d (or C for logit-space variant)
        num_classes: output dimension C
        rank       : LoRA rank r
        alpha      : LoRA scaling α
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        rank: int = 64,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, num_classes, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, u: Tensor) -> Tensor:
        """Return correction logits [B, C]."""
        return self.B(self.A(u)) * self.scale


# ---------------------------------------------------------------------------
# RectiQModel
# ---------------------------------------------------------------------------

class RectiQModel(nn.Module):
    """
    Frozen quantized backbone + ClassifierLoRA adapter.

    adapter_space="feature" (default): adapter input is u (pre-classifier feature).
    adapter_space="logit":             adapter input is z_q.

    forward(x) -> z = z_q + g_φ(input)
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        feat_dim: int,
        logit_dim: int,
        rank: int = 64,
        alpha: float = 16.0,
        adapter_space: str = "feature",
    ) -> None:
        super().__init__()
        if adapter_space not in {"feature", "logit"}:
            raise ValueError(f"adapter_space must be 'feature' or 'logit', got '{adapter_space}'")

        self.backbone = backbone
        self.adapter_space = adapter_space
        self.num_classes = num_classes

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        in_dim = feat_dim if adapter_space == "feature" else logit_dim
        self.adapter = ClassifierLoRA(in_dim, num_classes, rank=rank, alpha=alpha)

    def forward(self, x: Tensor) -> Tensor:
        u, z_q = _forward_split(self.backbone, x)
        adapter_input = u if self.adapter_space == "feature" else z_q
        return z_q + self.adapter(adapter_input)


# ---------------------------------------------------------------------------
# Training config and result
# ---------------------------------------------------------------------------

@dataclass
class RectiQTrainConfig:
    rank: int = 64
    alpha: float = 16.0
    kd_lambda: float = 0.0
    temperature: float = 4.0
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    adapter_space: str = "feature"
    max_batches_per_epoch: Optional[int] = None
    val_max_batches: Optional[int] = None

    @property
    def use_teacher(self) -> bool:
        return self.kd_lambda > 0.0


@dataclass
class RectiQTrainResult:
    model: RectiQModel
    adapter: nn.Module
    best_epoch: int
    best_val_acc: float
    history: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_rectiq_adapter(
    backbone: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    num_classes: int,
    feat_dim: int,
    logit_dim: int,
    config: RectiQTrainConfig,
    teacher_model: Optional[nn.Module] = None,
) -> RectiQTrainResult:
    """
    Train the Recti-Q adapter on top of a frozen quantized backbone.

    Args:
        backbone      : frozen (possibly quantized) timm backbone
        train_loader  : DataLoader yielding (images, labels)
        val_loader    : DataLoader yielding (images, labels)
        device        : torch.device
        num_classes   : number of output classes C
        feat_dim      : pre-classifier feature dimension d
        logit_dim     : logit dimension (usually == num_classes)
        config        : RectiQTrainConfig
        teacher_model : optional frozen FP32 timm backbone for KD
                        (required when config.use_teacher is True)

    Returns:
        RectiQTrainResult with the model loaded to the best adapter state.
    """
    use_kd = config.use_teacher and teacher_model is not None

    model = RectiQModel(
        backbone=backbone,
        num_classes=num_classes,
        feat_dim=feat_dim,
        logit_dim=logit_dim,
        rank=config.rank,
        alpha=config.alpha,
        adapter_space=config.adapter_space,
    ).to(device)

    if use_kd:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    optimizer = AdamW(
        model.adapter.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_acc = -1.0
    best_epoch = 0
    best_state: Dict[str, Tensor] = {}
    history: List[Dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if config.max_batches_per_epoch is not None and batch_idx >= config.max_batches_per_epoch:
                break

            images = images.to(device)
            labels = labels.to(device)

            z = model(images)
            loss_ce = F.cross_entropy(z, labels)

            if use_kd:
                T = config.temperature
                with torch.no_grad():
                    _, z_t = _forward_split(teacher_model, images)
                log_student = F.log_softmax(z / T, dim=-1)
                prob_teacher = F.softmax(z_t / T, dim=-1)
                loss_kd = F.kl_div(log_student, prob_teacher, reduction="batchmean") * (T ** 2)
                loss = loss_ce + config.kd_lambda * loss_kd
            else:
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = z.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ---- validate ----
        val_acc = _evaluate(model, val_loader, device, config.val_max_batches)

        entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(entry)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.adapter.state_dict().items()}

    # Restore best adapter weights
    if best_state:
        model.adapter.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return RectiQTrainResult(
        model=model,
        adapter=model.adapter,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        history=history,
    )


def _evaluate(
    model: RectiQModel,
    loader,
    device: torch.device,
    max_batches: Optional[int],
) -> float:
    """Top-1 accuracy on loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            z = model(images)
            preds = z.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Save / load adapter
# ---------------------------------------------------------------------------

def save_adapter(
    adapter: nn.Module,
    path: str | Path,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save adapter state_dict and optional metadata to path."""
    payload = {
        "state_dict": adapter.state_dict(),
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_adapter(adapter: nn.Module, path: str | Path) -> Dict[str, Any]:
    """Load adapter weights from path. Returns the saved meta dict."""
    payload = torch.load(path, map_location="cpu")
    adapter.load_state_dict(payload["state_dict"])
    return payload.get("meta", {})


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    device = torch.device("cpu")

    B, d, C, r = 4, 512, 10, 8

    # Minimal stub backbone
    class _StubBackbone(nn.Module):
        def __init__(self, d, C):
            super().__init__()
            self._feat = nn.Linear(3 * 32 * 32, d, bias=False)
            self._head = nn.Linear(d, C, bias=False)

        def forward_features(self, x):
            return x.flatten(1)

        def forward_head(self, feats, pre_logits=False):
            u = self._feat(feats)
            if pre_logits:
                return u
            return self._head(u)

    backbone = _StubBackbone(d, C)
    x = torch.randn(B, 3, 32, 32)

    # (1) Zero-init check: RectiQModel output == z_q at init
    model = RectiQModel(backbone, num_classes=C, feat_dim=d, logit_dim=C, rank=r, alpha=4.0)
    model.eval()
    with torch.no_grad():
        z_model = model(x)
        _, z_q = _forward_split(backbone, x)
    assert torch.allclose(z_model, z_q, atol=1e-6), \
        f"Zero-init check failed: max diff = {(z_model - z_q).abs().max().item()}"
    print("PASS: zero-init => RectiQModel(x) == z_q")

    # (2) One optimizer step changes adapter params
    model.train()
    optimizer = AdamW(model.adapter.parameters(), lr=1e-3)
    labels = torch.randint(0, C, (B,))
    before = model.adapter.B.weight.clone()
    z = model(x)
    F.cross_entropy(z, labels).backward()
    optimizer.step()
    after = model.adapter.B.weight.clone()
    assert not torch.allclose(before, after), "Adapter B.weight did not change after optimizer step"
    print("PASS: one optimizer step updates adapter params")

    # (3) Only adapter params have requires_grad=True
    for name, p in model.named_parameters():
        if "adapter" in name:
            assert p.requires_grad, f"Adapter param {name} should require grad"
        else:
            assert not p.requires_grad, f"Backbone param {name} should NOT require grad"
    print("PASS: only adapter params are trainable")

    # (4) logit-space ablation variant
    model_logit = RectiQModel(
        backbone, num_classes=C, feat_dim=d, logit_dim=C,
        rank=r, alpha=4.0, adapter_space="logit"
    )
    model_logit.eval()
    with torch.no_grad():
        z_logit = model_logit(x)
        _, z_q2 = _forward_split(backbone, x)
    assert torch.allclose(z_logit, z_q2, atol=1e-6), "logit-space zero-init check failed"
    print("PASS: logit-space variant zero-init => output == z_q")

    # (5) save/load round-trip
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    try:
        save_adapter(model.adapter, tmp, meta={"rank": r, "alpha": 4.0})
        fresh_adapter = ClassifierLoRA(d, C, rank=r, alpha=4.0)
        loaded_meta = load_adapter(fresh_adapter, tmp)
        assert loaded_meta["rank"] == r
        print("PASS: save_adapter / load_adapter round-trip")
    finally:
        os.unlink(tmp)

    print("\nAll smoke tests passed.")
