"""
Recti-Q experiment pipeline.

For each model, runs three phases and reports top-1 on in-distribution (ID) and
out-of-distribution (OOD) data:
  1. FP32        — pretrained / source-trained full-precision model (upper bound).
  2. W4          — 4-bit weight-only PTQ baseline (torchao Int4WeightOnly HQQ).
  3. Recti-Q     — classifier-head LoRA adapter trained source-only on top of the
                   frozen W4 backbone.

Benchmarks (selected by which datasets the config defines):
  - ImageNet-C : adapter trained on a 5% class-balanced ImageNet-1k train subset,
                 validated on ImageNet val, evaluated on corruptions x severities.
  - PACS       : leave-one-domain-out; a short ERM base fine-tune on the source
                 domains produces the FP32 base, then W4 + Recti-Q.

Usage:
    python -m src.main --config configs/imagenet_c_rectiq.yaml
    python -m src.main --config configs/pacs_rectiq.yaml --debug
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, ExperimentConfig, ModelConfig
from src.utils.logging import setup_logging, WandbLogger
from src.utils.formatting import (
    format_experiment_header,
    format_model_header,
    format_phase_comparison,
    format_quantization_stats,
    format_final_summary,
)
from src.models import ModelFactory
from src.datasets import (
    get_imagenet_loader,
    get_imagenet_subset_loader,
    get_all_imagenet_c_loaders,
    get_pacs_loaders,
    PACS_DOMAINS,
)
from src.quantization import quantize_model, get_model_size_mb, recalibrate_batchnorm
from src.rectiq import train_rectiq_adapter, RectiQTrainConfig, save_adapter


# ===========================================================================
# Helpers
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recti-Q experiment pipeline")
    p.add_argument("--config", "-c", required=True, help="Path to YAML config")
    p.add_argument("--debug", action="store_true", help="Tiny subset / 1 epoch smoke run")
    p.add_argument("--device", default=None, help="Override device (e.g. cuda:0, cpu)")
    p.add_argument("--models", nargs="+", default=None, help="Subset of model names")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(module: torch.nn.Module, loader: DataLoader, device: str,
             max_batches: Optional[int] = None, desc: str = "eval") -> Dict[str, float]:
    """Top-1/Top-5 accuracy of a logits-producing module over a loader."""
    module.eval()
    top1 = top5 = total = 0
    for i, (images, labels) in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        logits = module(images)
        _, pred5 = logits.topk(min(5, logits.size(1)), dim=1)
        correct = pred5.eq(labels.view(-1, 1))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        total += labels.size(0)
    total = max(total, 1)
    return {
        "top1_accuracy": 100.0 * top1 / total,
        "top5_accuracy": 100.0 * top5 / total,
        "num_samples": total,
    }


@torch.no_grad()
def evaluate_imagenet_c(module: torch.nn.Module, ood_loaders: Dict[tuple, DataLoader],
                        device: str, max_batches: Optional[int] = None,
                        max_combos: Optional[int] = None) -> Dict[str, float]:
    """Mean top-1 over ImageNet-C (corruption, severity) combinations."""
    per_combo: Dict[str, float] = {}
    items = list(ood_loaders.items())
    if max_combos is not None:
        items = items[:max_combos]
    for (corruption, severity), loader in items:
        m = evaluate(module, loader, device, max_batches=max_batches,
                     desc=f"{corruption}/s{severity}")
        per_combo[f"{corruption}/s{severity}"] = m["top1_accuracy"]
    mean_top1 = sum(per_combo.values()) / max(len(per_combo), 1)
    return {"mean_top1": mean_top1, "per_combo": per_combo}


def train_base_classifier(model, train_loader: DataLoader, val_loader: DataLoader,
                          device: str, epochs: int, lr: float,
                          max_batches: Optional[int] = None) -> float:
    """Short ERM fine-tune of the full backbone (PACS base model). Returns best val top-1."""
    backbone = model.backbone
    for p in backbone.parameters():
        p.requires_grad_(True)
    optimizer = AdamW(backbone.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        backbone.train()
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"base e{epoch}", leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(backbone(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(backbone, val_loader, device, max_batches=max_batches, desc="base val")["top1_accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in backbone.state_dict().items()}

    if best_state:
        backbone.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    return best_acc


def make_wandb(exp: ExperimentConfig, run_name: str, tags: List[str]) -> Optional[WandbLogger]:
    """One wandb run per pipeline invocation (model x benchmark)."""
    if not exp.logging.wandb.enabled:
        return None
    return WandbLogger(exp, run_name=run_name, run_group=exp.name,
                       run_job_type=run_name, extra_tags=tags)


def rectiq_train_config(exp: ExperimentConfig) -> RectiQTrainConfig:
    r = exp.rectiq
    epochs = 1 if exp.debug else r.epochs
    return RectiQTrainConfig(
        rank=r.rank, alpha=r.alpha, kd_lambda=r.kd_lambda, temperature=r.temperature,
        epochs=epochs, lr=r.lr, weight_decay=r.weight_decay, adapter_space=r.adapter_space,
        max_batches_per_epoch=(2 if exp.debug else r.max_batches_per_epoch),
        val_max_batches=(2 if exp.debug else r.val_max_batches),
    )


# ===========================================================================
# Phase runner (shared by both benchmarks)
# ===========================================================================

def run_phases(model, exp: ExperimentConfig, train_loader, val_loader,
               ood_eval, ood_tag: str, dataset_tag: str, run_name: str,
               text_logger) -> Dict[str, Any]:
    """
    Run FP32 -> W4 -> Recti-Q on one (model, source/target) setup.

    ood_eval(module) -> float : returns OOD top-1 for a logits-producing module.
    Returns {precision: {"id": top1, "ood": top1, "size_mb": mb}}.
    """
    device = exp.device
    eval_mb = 2 if exp.debug else None
    d, C = model.classifier_dims
    wb = make_wandb(exp, run_name, tags=[dataset_tag, model.name])
    results: Dict[str, Any] = {}
    rows = []

    # ── Phase 1: FP32 ──
    fp32_id = evaluate(model.backbone, val_loader, device, max_batches=eval_mb, desc="fp32 id")["top1_accuracy"]
    fp32_ood = ood_eval(model.backbone)
    fp32_size = get_model_size_mb(model.backbone)
    results["FP32"] = {"id": fp32_id, "ood": fp32_ood, "size_mb": fp32_size}
    rows.append({"precision": "FP32", "metrics": {"top1_accuracy": fp32_ood}, "size_mb": fp32_size})
    text_logger.info(f"  FP32  | ID {fp32_id:.2f}% | OOD {fp32_ood:.2f}% | {fp32_size:.1f} MB")

    # ── Phase 2: W4 PTQ ──
    q_backbone, q_stats = quantize_model(
        model.backbone, mode="W4", device=device,
        group_size=exp.quantization.group_size, use_hqq=exp.quantization.use_hqq,
        quantize_conv=exp.quantization.quantize_conv, conv_bits=exp.quantization.conv_bits,
    )
    print(format_quantization_stats(model.name, "W4", q_stats))
    # Re-fit BatchNorm to the quantized conv weights (source-only) so CNNs don't collapse.
    if exp.quantization.quantize_conv and exp.quantization.bn_recalib_batches > 0:
        n_bn = recalibrate_batchnorm(q_backbone, train_loader, device,
                                     num_batches=exp.quantization.bn_recalib_batches)
        if n_bn:
            text_logger.info(f"  BN recalibration: {n_bn} layers over "
                             f"{exp.quantization.bn_recalib_batches} source batches")
    w4_size = q_stats["quantized_size_mb"]
    w4_id = evaluate(q_backbone, val_loader, device, max_batches=eval_mb, desc="w4 id")["top1_accuracy"]
    w4_ood = ood_eval(q_backbone)
    results["W4"] = {"id": w4_id, "ood": w4_ood, "size_mb": w4_size}
    rows.append({"precision": "W4", "metrics": {"top1_accuracy": w4_ood}, "size_mb": w4_size})
    text_logger.info(f"  W4    | ID {w4_id:.2f}% | OOD {w4_ood:.2f}% | {w4_size:.1f} MB")

    # ── Phase 3: Recti-Q ──
    if exp.rectiq.enabled:
        cfg = rectiq_train_config(exp)
        teacher = model.backbone if cfg.use_teacher else None
        text_logger.info(
            f"  Recti-Q training: rank={cfg.rank} alpha={cfg.alpha} "
            f"kd_lambda={cfg.kd_lambda} adapter_space={cfg.adapter_space} epochs={cfg.epochs}"
        )
        rq = train_rectiq_adapter(
            backbone=q_backbone, train_loader=train_loader, val_loader=val_loader,
            device=device, num_classes=C, feat_dim=d, logit_dim=C,
            config=cfg, teacher_model=teacher,
        )
        rq_id = evaluate(rq.model, val_loader, device, max_batches=eval_mb, desc="rectiq id")["top1_accuracy"]
        rq_ood = ood_eval(rq.model)
        # Effective size = frozen W4 backbone + tiny adapter.
        adapter_dir = Path(exp.rectiq.output_dir or (Path(exp.output.results_dir) / exp.name / "rectiq_adapters"))
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = adapter_dir / f"{model.name}_{dataset_tag}_rectiq.pt"
        save_adapter(rq.adapter, adapter_path, meta={
            "model": model.name, "dataset": dataset_tag, "rank": cfg.rank, "alpha": cfg.alpha,
            "adapter_space": cfg.adapter_space, "kd_lambda": cfg.kd_lambda,
            "feat_dim": d, "num_classes": C, "best_epoch": rq.best_epoch,
        })
        adapter_mb = adapter_path.stat().st_size / (1024 ** 2)
        rq_size = w4_size + adapter_mb
        results["Recti-Q"] = {"id": rq_id, "ood": rq_ood, "size_mb": rq_size, "adapter_mb": adapter_mb}
        rows.append({"precision": "Recti-Q", "metrics": {"top1_accuracy": rq_ood}, "size_mb": rq_size})
        text_logger.info(
            f"  Recti-Q | ID {rq_id:.2f}% | OOD {rq_ood:.2f}% | {rq_size:.2f} MB "
            f"(adapter {adapter_mb*1024:.1f} KB, best epoch {rq.best_epoch})"
        )

    print(format_phase_comparison(model.name, ood_tag, rows))

    if wb is not None:
        summary = {}
        for prec, r in results.items():
            key = prec.lower().replace("-", "")
            summary[f"{key}_id_top1"] = r["id"]
            summary[f"{key}_ood_top1"] = r["ood"]
            summary[f"{key}_size_mb"] = r["size_mb"]
        wb.log_summary(summary)
        wb.finish()

    return results


# ===========================================================================
# Benchmarks
# ===========================================================================

def run_imagenet_c(model_cfg: ModelConfig, exp: ExperimentConfig, text_logger) -> Dict[str, Any]:
    model = ModelFactory.create(model_cfg, device=exp.device)
    text_logger.info(format_model_header(model_cfg.name, model.get_model_info()))

    imagenet_dcfg = exp.get_dataset("imagenet")
    imagenet_c_dcfg = exp.get_dataset("imagenet_c")
    eval_tf = model.build_transform(train=False)
    train_tf = model.build_transform(train=True)

    val_loader = get_imagenet_loader(
        imagenet_dcfg, model_name=model.name, transform=eval_tf,
        num_workers=exp.num_workers, debug=exp.debug, debug_samples=exp.debug_samples)
    train_loader = get_imagenet_subset_loader(
        imagenet_dcfg, model_name=model.name, transform=train_tf,
        num_workers=exp.rectiq.num_workers, debug=exp.debug, debug_samples=exp.debug_samples)
    ood_loaders = get_all_imagenet_c_loaders(
        imagenet_c_dcfg, model_name=model.name, num_workers=exp.num_workers,
        transform=eval_tf)

    max_combos = 2 if exp.debug else None
    eval_mb = 2 if exp.debug else None

    def ood_eval(module):
        return evaluate_imagenet_c(module, ood_loaders, exp.device,
                                   max_batches=eval_mb, max_combos=max_combos)["mean_top1"]

    return run_phases(
        model, exp, train_loader, val_loader, ood_eval,
        ood_tag="ImageNet-C (mean top-1)", dataset_tag="imagenet_c",
        run_name=f"{model.name}-imagenet_c", text_logger=text_logger)


def run_pacs(model_cfg: ModelConfig, exp: ExperimentConfig, text_logger) -> Dict[str, Any]:
    pacs_dcfg = exp.get_dataset("pacs")
    targets = [pacs_dcfg.target_domain] if pacs_dcfg.target_domain else (pacs_dcfg.domains or PACS_DOMAINS)
    base_epochs = 1 if exp.debug else pacs_dcfg.base_train_epochs
    eval_mb = 2 if exp.debug else None
    out: Dict[str, Any] = {}

    for target in targets:
        text_logger.info(f"\n  ── PACS leave-one-domain-out: target={target} ──")
        # Fresh model per target (source-trained base must not see target).
        model = ModelFactory.create(model_cfg, device=exp.device)
        eval_tf = model.build_transform(train=False)
        train_tf = model.build_transform(train=True)
        loaders = get_pacs_loaders(pacs_dcfg, target_domain=target, model_name=model.name,
                                   transform=eval_tf, train_transform=train_tf)
        train_loader = loaders["source_train"]
        val_loader = loaders["source_val"]
        test_loader = loaders["target_test"]

        if base_epochs > 0:
            best = train_base_classifier(model, train_loader, val_loader, exp.device,
                                         epochs=base_epochs, lr=pacs_dcfg.base_lr,
                                         max_batches=(2 if exp.debug else None))
            text_logger.info(f"  PACS base ERM ({base_epochs} ep) source-val top-1: {best:.2f}%")

        def ood_eval(module):
            return evaluate(module, test_loader, exp.device, max_batches=eval_mb, desc="ood")["top1_accuracy"]

        out[target] = run_phases(
            model, exp, train_loader, val_loader, ood_eval,
            ood_tag=f"PACS target={target} (top-1)", dataset_tag=f"pacs_{target}",
            run_name=f"{model.name}-pacs-{target}", text_logger=text_logger)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    exp = load_config(args.config)
    if args.debug:
        exp.debug = True
    if args.device:
        exp.device = args.device
    if args.seed is not None:
        exp.seed = args.seed
    if args.no_wandb:
        exp.logging.wandb.enabled = False

    set_seed(exp.seed)
    text_logger = setup_logging(exp)

    if exp.device.startswith("cuda") and not torch.cuda.is_available():
        text_logger.warning(f"CUDA requested ({exp.device}) but unavailable; using CPU.")
        exp.device = "cpu"

    model_cfgs = exp.models
    if args.models:
        wanted = set(args.models)
        model_cfgs = [m for m in model_cfgs if m.name in wanted]

    text_logger.info(format_experiment_header(
        exp.name, args.config, exp.device,
        [m.name for m in model_cfgs], list(exp.datasets.keys())))

    is_pacs = "pacs" in exp.datasets
    is_imagenet_c = "imagenet_c" in exp.datasets
    if not (is_pacs or is_imagenet_c):
        raise ValueError("Config must define either 'imagenet_c' (+ 'imagenet') or 'pacs' datasets.")

    all_results: Dict[str, Dict[str, Any]] = {}
    for model_cfg in model_cfgs:
        if is_imagenet_c:
            res = run_imagenet_c(model_cfg, exp, text_logger)
            all_results[model_cfg.name] = {
                f"imagenet_c/{prec}": {"top1_accuracy": r["ood"]} for prec, r in res.items()
            }
        else:
            res = run_pacs(model_cfg, exp, text_logger)
            flat = {}
            for target, phases in res.items():
                for prec, r in phases.items():
                    flat[f"pacs:{target}/{prec}"] = {"top1_accuracy": r["ood"]}
            all_results[model_cfg.name] = flat

    text_logger.info(format_final_summary(all_results))
    text_logger.info("Experiment completed.")
    return all_results


if __name__ == "__main__":
    main()
