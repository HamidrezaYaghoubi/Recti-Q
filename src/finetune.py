"""
YOLO fine-tuning entrypoint (BDD100K-first workflow).

This script implements a practical two-stage fine-tuning schedule:
1) short warmup with partial freeze (head-focused adaptation),
2) full-model fine-tuning.

Usage:
    python -m src.finetune --config configs/finetune_bdd100k_yolo.yaml
"""

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

from src.utils.config import load_config
from src.utils.logging import setup_logging, WandbLogger


@dataclass
class StageSettings:
    epochs: int
    freeze: int
    lr0: Optional[float] = None
    lrf: Optional[float] = None


@dataclass
class EvalSettings:
    run_after_training: bool = True
    split: str = "val"
    batch: Optional[int] = None


@dataclass
class FineTuneSettings:
    dataset: str
    init_mode: str
    custom_weights: Optional[str]
    project_dir: str
    imgsz: int
    batch: int
    workers: int
    optimizer: str
    weight_decay: float
    momentum: float
    patience: int
    cache: bool
    amp: bool
    cos_lr: bool
    close_mosaic: int
    val: bool
    save: bool
    deterministic: bool
    stage1: StageSettings
    stage2: StageSettings
    eval: EvalSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO fine-tuning pipeline")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--models", nargs="+", default=None, help="Optional subset of model names")
    parser.add_argument("--device", type=str, default=None, help="Optional device override (e.g., cuda:0)")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb for this run")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _read_raw_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _parse_finetune_settings(raw: Dict[str, Any]) -> FineTuneSettings:
    ft = raw.get("fine_tuning") or {}
    if not ft:
        raise ValueError("Missing 'fine_tuning' section in config.")

    stage1_raw = ft.get("stage1") or {}
    stage2_raw = ft.get("stage2") or {}
    eval_raw = ft.get("eval") or {}

    stage1 = StageSettings(
        epochs=int(stage1_raw.get("epochs", 0)),
        freeze=int(stage1_raw.get("freeze", 10)),
        lr0=stage1_raw.get("lr0"),
        lrf=stage1_raw.get("lrf"),
    )
    stage2 = StageSettings(
        epochs=int(stage2_raw.get("epochs", 0)),
        freeze=int(stage2_raw.get("freeze", 0)),
        lr0=stage2_raw.get("lr0"),
        lrf=stage2_raw.get("lrf"),
    )
    eval_cfg = EvalSettings(
        run_after_training=bool(eval_raw.get("run_after_training", True)),
        split=str(eval_raw.get("split", "val")),
        batch=eval_raw.get("batch"),
    )

    return FineTuneSettings(
        dataset=str(ft.get("dataset", "bdd100k")),
        init_mode=str(ft.get("init_mode", "pretrained")).lower(),
        custom_weights=ft.get("custom_weights"),
        project_dir=str(ft.get("project_dir", "./runs/finetune")),
        imgsz=int(ft.get("imgsz", 640)),
        batch=int(ft.get("batch", 16)),
        workers=int(ft.get("workers", 4)),
        optimizer=str(ft.get("optimizer", "AdamW")),
        weight_decay=float(ft.get("weight_decay", 0.0005)),
        momentum=float(ft.get("momentum", 0.937)),
        patience=int(ft.get("patience", 30)),
        cache=bool(ft.get("cache", False)),
        amp=bool(ft.get("amp", True)),
        cos_lr=bool(ft.get("cos_lr", True)),
        close_mosaic=int(ft.get("close_mosaic", 10)),
        val=bool(ft.get("val", True)),
        save=bool(ft.get("save", True)),
        deterministic=bool(ft.get("deterministic", True)),
        stage1=stage1,
        stage2=stage2,
        eval=eval_cfg,
    )


def _build_bdd_data_yaml(dataset_root: Path, out_path: Path) -> Path:
    if not (dataset_root / "train" / "images").exists():
        raise FileNotFoundError(f"Expected train images at: {dataset_root / 'train' / 'images'}")
    if not (dataset_root / "val" / "images").exists():
        raise FileNotFoundError(f"Expected val images at: {dataset_root / 'val' / 'images'}")

    names = [
        "person",
        "rider",
        "car",
        "bus",
        "truck",
        "bike",
        "motor",
        "traffic light",
        "traffic sign",
        "train",
    ]
    source_yaml = dataset_root / "data.yaml"
    if source_yaml.exists():
        try:
            with open(source_yaml, "r") as f:
                payload = yaml.safe_load(f) or {}
            yaml_names = payload.get("names")
            if isinstance(yaml_names, list) and yaml_names:
                names = [str(x) for x in yaml_names]
            elif isinstance(yaml_names, dict) and yaml_names:
                names = [str(yaml_names[k]) for k in sorted(yaml_names.keys(), key=lambda v: int(v))]
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": names,
    }
    if (dataset_root / "test" / "images").exists():
        payload["test"] = "test/images"

    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return out_path


def _extract_det_metrics(val_results) -> Dict[str, float]:
    box = getattr(val_results, "box", None)
    results_dict = getattr(val_results, "results_dict", {}) or {}

    def _metric(attr: str, fallback: str) -> float:
        if box is not None and hasattr(box, attr):
            try:
                return float(getattr(box, attr)) * 100.0
            except Exception:
                pass
        return float(results_dict.get(fallback, 0.0)) * 100.0

    return {
        "map": _metric("map", "metrics/mAP50-95(B)"),
        "map_50": _metric("map50", "metrics/mAP50(B)"),
        "map75": _metric("map75", "metrics/mAP75(B)"),
    }


def _resolve_init_source(model_arch: str, model_weights: str, ft: FineTuneSettings) -> str:
    if ft.custom_weights:
        p = Path(ft.custom_weights)
        if not p.exists():
            raise FileNotFoundError(f"custom_weights not found: {p}")
        return str(p.resolve())

    if ft.init_mode == "scratch":
        return f"{model_arch}.yaml"

    if model_weights and model_weights.lower() not in {"coco", "none"}:
        p = Path(model_weights)
        if p.exists():
            return str(p.resolve())
        return model_weights

    return f"{model_arch}.pt"


def _build_common_train_args(
    exp,
    ft: FineTuneSettings,
    data_yaml: Path,
    stage_name: str,
    run_name: str,
) -> Dict[str, Any]:
    project = ft.project_dir or str(Path(exp.output.results_dir) / exp.name / "finetune_runs")
    args = {
        "data": str(data_yaml),
        "task": "detect",
        "imgsz": ft.imgsz,
        "batch": ft.batch,
        "device": exp.device,
        "workers": ft.workers,
        "optimizer": ft.optimizer,
        "weight_decay": ft.weight_decay,
        "momentum": ft.momentum,
        "patience": ft.patience,
        "cache": ft.cache,
        "amp": ft.amp,
        "cos_lr": ft.cos_lr,
        "close_mosaic": ft.close_mosaic,
        "val": ft.val,
        "save": ft.save,
        "deterministic": ft.deterministic,
        "seed": exp.seed,
        "project": project,
        "name": f"{run_name}-{stage_name}",
        "exist_ok": True,
        "verbose": False,
    }
    return args


def _save_metadata(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _path_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def _make_finetune_run_name(model_name: str, dataset_name: str) -> str:
    return f"{model_name}-detection-{dataset_name}-finetune"


def _log_stage_metrics(
    wandb_logger: Optional[WandbLogger],
    stage_name: str,
    stage_metrics: Optional[Dict[str, float]],
) -> None:
    if wandb_logger is None or stage_metrics is None:
        return
    wandb_logger.log(
        {
            f"{stage_name}/map": stage_metrics["map"],
            f"{stage_name}/map_50": stage_metrics["map_50"],
            f"{stage_name}/map75": stage_metrics["map75"],
        }
    )


def main() -> None:
    args = parse_args()
    exp = load_config(args.config)
    raw = _read_raw_yaml(args.config)
    ft = _parse_finetune_settings(raw)

    if args.device:
        exp.device = args.device
    if args.seed is not None:
        exp.seed = args.seed

    if args.no_wandb:
        exp.logging.wandb.enabled = False
        os.environ["WANDB_DISABLED"] = "true"

    set_seed(exp.seed)
    text_logger = setup_logging(exp)
    text_logger.info(f"Starting fine-tuning experiment: {exp.name}")
    text_logger.info(f"Config: {args.config}")

    if exp.device.startswith("cuda") and not torch.cuda.is_available():
        text_logger.warning(f"CUDA requested ({exp.device}) but not available. Falling back to CPU.")
        exp.device = "cpu"

    if ft.stage1.epochs <= 0 and ft.stage2.epochs <= 0:
        raise ValueError("At least one stage must have epochs > 0 (stage1 or stage2).")

    dataset_cfg = exp.get_dataset(ft.dataset)
    dataset_root = Path(dataset_cfg.root).resolve()
    data_yaml = _build_bdd_data_yaml(
        dataset_root=dataset_root,
        out_path=Path(exp.output.results_dir) / exp.name / "finetune_data" / f"{ft.dataset}_train.yaml",
    )
    text_logger.info(f"Using training data yaml: {data_yaml}")

    from ultralytics import YOLO

    model_cfgs = exp.models
    if args.models:
        allowed = set(args.models)
        model_cfgs = [m for m in model_cfgs if m.name in allowed]
        if not model_cfgs:
            raise ValueError(f"No models selected. Requested={args.models}")

    run_manifest = {}

    for m in model_cfgs:
        run_name = _make_finetune_run_name(m.name, ft.dataset)
        text_logger.info(f"\n=== Fine-tuning {m.name} ({run_name}) ===")

        wandb_logger: Optional[WandbLogger] = None
        if exp.logging.wandb.enabled:
            wandb_logger = WandbLogger(
                exp,
                run_name=run_name,
                run_group=exp.name,
                run_job_type="finetune",
                extra_tags=["finetune", "detection", ft.dataset, m.name],
                extra_config={
                    "task": "detection",
                    "dataset": ft.dataset,
                    "model": m.name,
                    "device": exp.device,
                    "stage1_epochs": ft.stage1.epochs,
                    "stage2_epochs": ft.stage2.epochs,
                    "stage1_freeze": ft.stage1.freeze,
                    "stage2_freeze": ft.stage2.freeze,
                    "eval_split": ft.eval.split,
                },
            )
            wandb_logger.log_summary({"status": "running"})

        try:
            init_source = _resolve_init_source(
                model_arch=m.architecture.lower(),
                model_weights=m.weights,
                ft=ft,
            )
            text_logger.info(f"Init source: {init_source}")

            current_source = init_source
            stage_artifacts: Dict[str, Any] = {}

            if wandb_logger is not None:
                wandb_logger.log(
                    {
                        "train/stage1_epochs": float(ft.stage1.epochs),
                        "train/stage2_epochs": float(ft.stage2.epochs),
                        "train/stage1_freeze": float(ft.stage1.freeze),
                        "train/stage2_freeze": float(ft.stage2.freeze),
                        "train/imgsz": float(ft.imgsz),
                        "train/batch": float(ft.batch),
                    }
                )

            if ft.stage1.epochs > 0:
                text_logger.info(
                    f"Stage 1 (warmup): epochs={ft.stage1.epochs}, freeze={ft.stage1.freeze}, lr0={ft.stage1.lr0}"
                )
                model_s1 = YOLO(current_source, task="detect")
                train_args = _build_common_train_args(exp, ft, data_yaml, stage_name="s1-head", run_name=run_name)
                train_args.update({"epochs": ft.stage1.epochs, "freeze": ft.stage1.freeze})
                if ft.stage1.lr0 is not None:
                    train_args["lr0"] = float(ft.stage1.lr0)
                if ft.stage1.lrf is not None:
                    train_args["lrf"] = float(ft.stage1.lrf)

                s1_results = model_s1.train(**train_args)
                s1_dir = Path(s1_results.save_dir)
                s1_best = s1_dir / "weights" / "best.pt"
                s1_last = s1_dir / "weights" / "last.pt"
                if s1_best.exists():
                    current_source = str(s1_best)
                elif s1_last.exists():
                    current_source = str(s1_last)
                else:
                    raise FileNotFoundError(f"Stage-1 did not produce best.pt/last.pt under: {s1_dir}")
                stage_artifacts["stage1_dir"] = str(s1_dir)
                stage_artifacts["stage1_best"] = str(s1_best) if s1_best.exists() else None
                stage_artifacts["stage1_last"] = str(s1_last) if s1_last.exists() else None

                s1_metrics = _extract_det_metrics(s1_results)
                _log_stage_metrics(wandb_logger, "stage1", s1_metrics)

            if ft.stage2.epochs > 0:
                text_logger.info(
                    f"Stage 2 (full): epochs={ft.stage2.epochs}, freeze={ft.stage2.freeze}, lr0={ft.stage2.lr0}"
                )
                model_s2 = YOLO(current_source, task="detect")
                train_args = _build_common_train_args(exp, ft, data_yaml, stage_name="s2-full", run_name=run_name)
                train_args.update({"epochs": ft.stage2.epochs, "freeze": ft.stage2.freeze})
                if ft.stage2.lr0 is not None:
                    train_args["lr0"] = float(ft.stage2.lr0)
                if ft.stage2.lrf is not None:
                    train_args["lrf"] = float(ft.stage2.lrf)

                s2_results = model_s2.train(**train_args)
                s2_dir = Path(s2_results.save_dir)
                s2_best = s2_dir / "weights" / "best.pt"
                s2_last = s2_dir / "weights" / "last.pt"
                if s2_best.exists():
                    current_source = str(s2_best)
                elif s2_last.exists():
                    current_source = str(s2_last)
                else:
                    raise FileNotFoundError(f"Stage-2 did not produce best.pt/last.pt under: {s2_dir}")
                stage_artifacts["stage2_dir"] = str(s2_dir)
                stage_artifacts["stage2_best"] = str(s2_best) if s2_best.exists() else None
                stage_artifacts["stage2_last"] = str(s2_last) if s2_last.exists() else None

                s2_metrics = _extract_det_metrics(s2_results)
                _log_stage_metrics(wandb_logger, "stage2", s2_metrics)

            final_weights = Path(current_source).resolve()
            if not final_weights.exists():
                raise FileNotFoundError(f"Final fine-tuned checkpoint not found: {final_weights}")

            # Copy final best to checkpoint area for stable downstream usage.
            ckpt_dir = Path(exp.output.checkpoint_dir) / exp.name / "finetune"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            final_copy = ckpt_dir / f"{m.name}_{ft.dataset}_finetuned_best.pt"
            shutil.copy2(final_weights, final_copy)
            text_logger.info(f"Saved final fine-tuned weights: {final_copy}")

            metrics = None
            if ft.eval.run_after_training:
                eval_batch = int(ft.eval.batch) if ft.eval.batch is not None else ft.batch
                text_logger.info(
                    f"Running post-train val for {m.name} on {ft.dataset} split={ft.eval.split}, batch={eval_batch}"
                )
                eval_model = YOLO(str(final_weights), task="detect")
                val_results = eval_model.val(
                    data=str(data_yaml),
                    split=ft.eval.split,
                    batch=eval_batch,
                    imgsz=ft.imgsz,
                    device=exp.device,
                    verbose=False,
                )
                metrics = _extract_det_metrics(val_results)
                text_logger.info(
                    f"Post-train metrics | map={metrics['map']:.3f} | map_50={metrics['map_50']:.3f} | map75={metrics['map75']:.3f}"
                )

            final_size_mb = _path_size_mb(final_copy)
            if wandb_logger is not None:
                if metrics is not None:
                    wandb_logger.log(
                        {
                            "eval/map": metrics["map"],
                            "eval/map_50": metrics["map_50"],
                            "eval/map75": metrics["map75"],
                            "model_size_mb": final_size_mb,
                        }
                    )
                    wandb_logger.log_summary(
                        {
                            "map": metrics["map"],
                            "map_50": metrics["map_50"],
                            "map75": metrics["map75"],
                            "model_size_mb": final_size_mb,
                        }
                    )
                else:
                    wandb_logger.log({"model_size_mb": final_size_mb})
                    wandb_logger.log_summary({"model_size_mb": final_size_mb})
                wandb_logger.log_summary(
                    {
                        "final_weights": str(final_weights),
                        "final_weights_copy": str(final_copy),
                        "status": "completed",
                    }
                )
                wandb_logger.log_artifact(
                    final_copy,
                    name=f"{m.name}-{ft.dataset}-finetuned-best",
                    artifact_type="model",
                    metadata={
                        "model": m.name,
                        "dataset": ft.dataset,
                        "seed": exp.seed,
                        "stage1_epochs": ft.stage1.epochs,
                        "stage2_epochs": ft.stage2.epochs,
                    },
                )

            run_manifest[m.name] = {
                "dataset": ft.dataset,
                "init_source": init_source,
                "final_weights": str(final_weights),
                "final_weights_copy": str(final_copy),
                "final_size_mb": final_size_mb,
                "metrics": metrics,
                **stage_artifacts,
            }
        except Exception as e:
            if wandb_logger is not None:
                wandb_logger.log_summary({"status": "failed", "error": str(e)})
            raise
        finally:
            if wandb_logger is not None:
                wandb_logger.finish()

    manifest_path = Path(exp.output.results_dir) / exp.name / "finetune_manifest.json"
    _save_metadata(manifest_path, run_manifest)
    text_logger.info(f"Saved fine-tune manifest: {manifest_path}")
    text_logger.info("Fine-tuning pipeline completed.")


if __name__ == "__main__":
    main()
