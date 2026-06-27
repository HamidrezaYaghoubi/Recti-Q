"""
Terminal output formatting for classification experiments.

Provides compact, consistent tables for per-phase results (FP32 / W4 / Recti-Q)
and quantization size stats.
"""

from typing import Any, Dict, List, Optional


def _top1(metrics: Any) -> float:
    """Read top-1 accuracy from a ClassificationMetrics or dict."""
    if hasattr(metrics, "top1_accuracy"):
        return float(metrics.top1_accuracy)
    return float(metrics.get("top1_accuracy", 0.0))


def _top5(metrics: Any) -> float:
    """Read top-5 accuracy from a ClassificationMetrics or dict."""
    if hasattr(metrics, "top5_accuracy"):
        return float(metrics.top5_accuracy)
    return float(metrics.get("top5_accuracy", 0.0))


def format_experiment_header(
    experiment_name: str,
    config_file: str,
    device: str,
    models: List[str],
    datasets: List[str],
) -> str:
    """Format the experiment banner."""
    width = 64
    lines = [
        "",
        "=" * width,
        f"  EXPERIMENT: {experiment_name}",
        "=" * width,
        f"  Config:   {config_file}",
        f"  Device:   {device}",
        f"  Models:   {', '.join(models)}",
        f"  Datasets: {', '.join(datasets)}",
        "=" * width,
        "",
    ]
    return "\n".join(lines)


def format_model_header(model_name: str, model_info: Dict[str, Any]) -> str:
    """Format a per-model section header."""
    width = 64
    params = model_info.get("total_parameters", 0)
    params_str = f"{params / 1e6:.1f}M" if params > 0 else "N/A"
    return "\n".join(
        [
            "",
            "-" * width,
            f"  MODEL: {model_name}  ({params_str} params)",
            "-" * width,
        ]
    )


def format_classification_results(
    model_name: str,
    dataset_name: str,
    metrics: Any,
    precision: str = "fp32",
) -> str:
    """Format a single classification result line."""
    return (
        f"  [{precision:<16}] {model_name:<20} {dataset_name:<24} "
        f"top1={_top1(metrics):6.2f}%  top5={_top5(metrics):6.2f}%"
    )


def format_quantization_stats(model_name: str, mode: str, stats: Dict[str, Any]) -> str:
    """Format quantization size/compression stats."""
    orig = stats.get("original_size_mb", 0.0)
    quant = stats.get("quantized_size_mb", 0.0)
    compress = stats.get("compression_ratio", 0.0)
    reduction = stats.get("size_reduction_pct", 0.0)
    return (
        f"  [quant {mode}] {model_name}: "
        f"{orig:.1f} MB -> {quant:.1f} MB "
        f"({compress:.2f}x, -{reduction:.1f}%)"
    )


def format_phase_comparison(
    model_name: str,
    dataset_name: str,
    rows: List[Dict[str, Any]],
) -> str:
    """
    Format an FP32 / W4 / Recti-Q comparison table for one (model, dataset).

    Each row dict: {"precision", "metrics", "size_mb"} (+ optional precomputed deltas).
    Deltas are computed against the FP32 row (delta) and the W4 row (recovery).
    """
    fp32 = next((r for r in rows if r["precision"].lower() == "fp32"), None)
    w4 = next((r for r in rows if r["precision"].lower() in {"w4", "ptq", "ptq_w4"}), None)
    fp32_top1 = _top1(fp32["metrics"]) if fp32 else None
    w4_top1 = _top1(w4["metrics"]) if w4 else None

    width = 78
    lines = [
        "",
        "=" * width,
        f"  {model_name}  |  {dataset_name}",
        "-" * width,
        f"  {'Config':<18}{'top1':>9}{'Δ vs FP32 (pp)':>18}{'Recov vs W4 (pp)':>18}{'Size MB':>12}",
        "-" * width,
    ]
    for r in rows:
        top1 = _top1(r["metrics"])
        delta = f"{top1 - fp32_top1:+.2f}" if fp32_top1 is not None else "-"
        recov = f"{top1 - w4_top1:+.2f}" if w4_top1 is not None else "-"
        size = r.get("size_mb")
        size_str = f"{size:.2f}" if size is not None else "-"
        lines.append(
            f"  {r['precision']:<18}{top1:>9.2f}{delta:>18}{recov:>18}{size_str:>12}"
        )
    lines.append("=" * width)
    return "\n".join(lines)


def format_final_summary(all_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Format a final flat summary.

    all_results: {model_name: {f"{dataset}/{precision}": metrics}}.
    """
    width = 78
    lines = ["", "=" * width, f"{'FINAL SUMMARY (OOD top-1)':^{width}}", "=" * width, ""]
    for model_name, entries in all_results.items():
        lines.append(f"  {model_name}")
        for key, metrics in entries.items():
            lines.append(f"    {key:<48} top1={_top1(metrics):6.2f}%")
        lines.append("")
    lines.append("=" * width)
    return "\n".join(lines)
