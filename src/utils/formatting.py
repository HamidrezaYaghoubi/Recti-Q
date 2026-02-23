"""
Output formatting utilities for consistent terminal output.

This module provides unified formatting for metrics display across
classification and detection tasks, with support for different
verbosity levels and quantization comparisons.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


# Box drawing characters for clean tables
BOX_CHARS = {
    "h": "─",  # horizontal
    "v": "│",  # vertical
    "tl": "┌", # top-left
    "tr": "┐", # top-right
    "bl": "└", # bottom-left
    "br": "┘", # bottom-right
    "lj": "├", # left-join
    "rj": "┤", # right-join
    "tj": "┬", # top-join
    "bj": "┴", # bottom-join
    "x": "┼",  # cross
}


def format_classification_results(
    model_name: str,
    dataset_name: str,
    metrics: Dict[str, Any],
    precision: str = "fp32",
    num_samples: Optional[int] = None,
) -> str:
    """
    Format classification results for display.
    
    Args:
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        metrics: Dictionary with top1_accuracy, top5_accuracy, etc.
        precision: Model precision (fp32, int8, etc.).
        num_samples: Number of samples evaluated.
        
    Returns:
        Formatted string for display.
    """
    # Handle both dict and ClassificationMetrics objects
    if hasattr(metrics, 'top1_accuracy'):
        top1 = metrics.top1_accuracy
        top5 = metrics.top5_accuracy
        samples = metrics.num_samples
    else:
        top1 = metrics.get("top1_accuracy", 0.0)
        top5 = metrics.get("top5_accuracy", 0.0)
        samples = metrics.get("num_samples", num_samples or 0)
    
    lines = [
        f"┌{'─' * 50}┐",
        f"│ {'Classification Results':^48} │",
        f"├{'─' * 50}┤",
        f"│ Model:    {model_name:<38} │",
        f"│ Dataset:  {dataset_name:<38} │",
        f"│ Precision: {precision:<37} │",
        f"├{'─' * 50}┤",
        f"│ Top-1 Accuracy: {top1:>29.2f}% │",
        f"│ Top-5 Accuracy: {top5:>29.2f}% │",
        f"│ Samples:        {samples:>29,} │",
        f"└{'─' * 50}┘",
    ]
    return "\n".join(lines)


def format_detection_results(
    model_name: str,
    dataset_name: str,
    metrics: Dict[str, Any],
    precision: str = "fp32",
    verbose: bool = False,
) -> str:
    """
    Format detection results for display.
    
    Args:
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        metrics: Dictionary with mAP, mAP50, mAP75, etc.
        precision: Model precision (fp32, int8, etc.).
        verbose: Include detailed per-size metrics.
        
    Returns:
        Formatted string for display.
    """
    mAP = metrics.get("mAP", 0.0)
    mAP50 = metrics.get("mAP50", 0.0)
    mAP75 = metrics.get("mAP75", 0.0)
    mAP_small = metrics.get("mAP_small", 0.0)
    mAP_medium = metrics.get("mAP_medium", 0.0)
    mAP_large = metrics.get("mAP_large", 0.0)
    AR_100 = metrics.get("AR_100", 0.0)
    num_images = metrics.get("num_images", 0)
    total_dets = metrics.get("total_detections", 0)
    
    lines = [
        f"┌{'─' * 50}┐",
        f"│ {'Detection Results':^48} │",
        f"├{'─' * 50}┤",
        f"│ Model:    {model_name:<38} │",
        f"│ Dataset:  {dataset_name:<38} │",
        f"│ Precision: {precision:<37} │",
        f"├{'─' * 50}┤",
        f"│ mAP@0.5:0.95: {mAP:>31.2f}% │",
        f"│ mAP@0.50:     {mAP50:>31.2f}% │",
        f"│ mAP@0.75:     {mAP75:>31.2f}% │",
    ]
    
    if verbose:
        lines.extend([
            f"├{'─' * 50}┤",
            f"│ By Object Size:                                  │",
            f"│   Small:  {mAP_small:>35.2f}% │",
            f"│   Medium: {mAP_medium:>35.2f}% │",
            f"│   Large:  {mAP_large:>35.2f}% │",
        ])
    
    lines.extend([
        f"├{'─' * 50}┤",
        f"│ AR@100:       {AR_100:>31.2f}% │",
        f"│ Images:       {num_images:>31,} │",
        f"│ Detections:   {total_dets:>31,} │",
        f"└{'─' * 50}┘",
    ])
    
    return "\n".join(lines)


def format_summary_line(
    task: str,
    model_name: str,
    dataset_name: str,
    metrics: Dict[str, Any],
    precision: str = "fp32",
) -> str:
    """
    Format a single-line summary for quick reference.
    
    Args:
        task: Task type ('classification' or 'detection').
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        metrics: Metrics dictionary.
        precision: Model precision.
        
    Returns:
        Single-line summary string.
    """
    if task == "classification":
        if hasattr(metrics, 'top1_accuracy'):
            top1, top5 = metrics.top1_accuracy, metrics.top5_accuracy
        else:
            top1 = metrics.get("top1_accuracy", 0.0)
            top5 = metrics.get("top5_accuracy", 0.0)
        return f"{model_name:20} | {dataset_name:15} | {precision:6} | Top-1: {top1:5.2f}% | Top-5: {top5:5.2f}%"
    
    elif task == "detection":
        mAP = metrics.get("mAP", 0.0)
        mAP50 = metrics.get("mAP50", 0.0)
        return f"{model_name:20} | {dataset_name:15} | {precision:6} | mAP: {mAP:5.2f}% | mAP50: {mAP50:5.2f}%"
    
    else:
        return f"{model_name:20} | {dataset_name:15} | {precision:6} | {metrics}"


def format_comparison_table(
    results: List[Dict[str, Any]],
    task: str = "classification",
) -> str:
    """
    Format a comparison table for multiple models/precisions.
    
    Args:
        results: List of result dicts with model_name, dataset, precision, metrics.
        task: Task type for formatting.
        
    Returns:
        Formatted comparison table.
    """
    if task == "classification":
        header = f"{'Model':20} │ {'Dataset':15} │ {'Prec':6} │ {'Top-1':>7} │ {'Top-5':>7}"
        sep = f"{'─' * 20}─┼─{'─' * 15}─┼─{'─' * 6}─┼─{'─' * 7}─┼─{'─' * 7}"
    else:
        header = f"{'Model':20} │ {'Dataset':15} │ {'Prec':6} │ {'mAP':>7} │ {'mAP50':>7}"
        sep = f"{'─' * 20}─┼─{'─' * 15}─┼─{'─' * 6}─┼─{'─' * 7}─┼─{'─' * 7}"
    
    lines = [
        f"┌{'─' * (len(header))}┐",
        f"│{header}│",
        f"├{sep}┤",
    ]
    
    for r in results:
        model = r.get("model_name", "")[:20]
        dataset = r.get("dataset", "")[:15]
        precision = r.get("precision", "fp32")[:6]
        metrics = r.get("metrics", {})
        
        if task == "classification":
            if hasattr(metrics, 'top1_accuracy'):
                top1, top5 = metrics.top1_accuracy, metrics.top5_accuracy
            else:
                top1 = metrics.get("top1_accuracy", 0.0)
                top5 = metrics.get("top5_accuracy", 0.0)
            row = f"{model:20} │ {dataset:15} │ {precision:6} │ {top1:6.2f}% │ {top5:6.2f}%"
        else:
            mAP = metrics.get("mAP", 0.0)
            mAP50 = metrics.get("mAP50", 0.0)
            row = f"{model:20} │ {dataset:15} │ {precision:6} │ {mAP:6.2f}% │ {mAP50:6.2f}%"
        
        lines.append(f"│{row}│")
    
    lines.append(f"└{'─' * (len(header))}┘")
    
    return "\n".join(lines)


def format_experiment_header(
    experiment_name: str,
    config_file: str,
    device: str,
    models: List[str],
    datasets: List[str],
) -> str:
    """
    Format experiment header for logging.
    
    Args:
        experiment_name: Name of the experiment.
        config_file: Path to config file.
        device: Device being used.
        models: List of model names.
        datasets: List of dataset names.
        
    Returns:
        Formatted header string.
    """
    width = 60
    lines = [
        "",
        "═" * width,
        f"  EXPERIMENT: {experiment_name}",
        "═" * width,
        f"  Config:   {config_file}",
        f"  Device:   {device}",
        f"  Models:   {', '.join(models)}",
        f"  Datasets: {', '.join(datasets)}",
        "═" * width,
        "",
    ]
    return "\n".join(lines)


def format_model_header(model_name: str, model_info: Dict[str, Any]) -> str:
    """
    Format model section header.
    
    Args:
        model_name: Name of the model.
        model_info: Model information dict.
        
    Returns:
        Formatted header string.
    """
    width = 60
    params = model_info.get("total_parameters", 0)
    params_str = f"{params / 1e6:.1f}M" if params > 0 else "N/A"
    
    lines = [
        "",
        "─" * width,
        f"  MODEL: {model_name}",
        f"  Parameters: {params_str} | Task: {model_info.get('task', 'N/A')} | Precision: {model_info.get('precision', 'fp32')}",
        "─" * width,
    ]
    return "\n".join(lines)


def format_quantization_stats(
    model_name: str,
    mode: str,
    stats: Dict[str, Any],
) -> str:
    """
    Format quantization statistics for display.
    
    Args:
        model_name: Name of the model.
        mode: Quantization mode (weight_only, weight_and_activation).
        stats: Dictionary of quantization stats from quantize_model().
        
    Returns:
        Formatted string for display.
    """
    orig = stats.get("original_size_mb", 0)
    quant = stats.get("quantized_size_mb", 0)
    compress = stats.get("compression_ratio", 0)
    reduction = stats.get("size_reduction_pct", 0)
    q_time = stats.get("quantization_time_s", 0)
    calibration_source = stats.get("calibration_source")
    calibration_fraction = stats.get("calibration_fraction")
    total_layers = stats.get("total_layers")
    int8_layers = stats.get("int8_layers")
    fallback_layers = stats.get("fallback_layers")
    int8_ratio = stats.get("int8_ratio")
    fallback_ratio = stats.get("fallback_ratio")
    mode_desc = stats.get("mode_description", mode)
    target = stats.get("target_layers", "nn.Linear")

    orig_layers = stats.get("original_layers", {})
    quant_layers = stats.get("quantized_layers", {})

    lines = [
        f"┌{'─' * 50}┐",
        f"│ {'Quantization Stats':^48} │",
        f"├{'─' * 50}┤",
        f"│ Model:    {model_name:<38} │",
        f"│ Mode:     {mode_desc:<38} │",
        f"│ Target:   {target:<38} │",
        f"├{'─' * 50}┤",
        f"│ Original size:    {orig:>24.1f} MB │",
        f"│ Quantized size:   {quant:>24.1f} MB │",
        f"│ Compression:      {compress:>25.2f}x │",
        f"│ Size reduction:   {reduction:>24.1f} % │",
        f"│ Quant time:       {q_time:>24.1f} s │",
    ]

    if int8_layers is not None and total_layers is not None:
        lines.append(f"│ INT8 layers:      {f'{int8_layers}/{total_layers}':>25} │")
    if int8_ratio is not None:
        lines.append(f"│ INT8 coverage:    {float(int8_ratio) * 100:>24.1f} % │")
    if fallback_layers is not None and total_layers is not None:
        lines.append(f"│ Fallback layers:  {f'{fallback_layers}/{total_layers}':>25} │")
    if fallback_ratio is not None:
        lines.append(f"│ Fallback ratio:   {float(fallback_ratio) * 100:>24.1f} % │")
    if calibration_source is not None:
        lines.append(f"│ Calib source:     {str(calibration_source):>25} │")
    if calibration_fraction is not None:
        lines.append(f"│ Calib fraction:   {float(calibration_fraction):>24.4f} │")

    lines.extend(
        [
            f"├{'─' * 50}┤",
            f"│ Original layers:                                 │",
            f"│   Linear:  {orig_layers.get('Linear', 0):>35d}  │",
            f"│   Conv2d:  {orig_layers.get('Conv2d', 0):>35d}  │",
            f"│ After quantization:                              │",
            f"│   Linear:  {quant_layers.get('Linear', 0):>35d}  │",
            f"│   Conv2d:  {quant_layers.get('Conv2d', 0):>35d}  │",
            f"└{'─' * 50}┘",
        ]
    )
    return "\n".join(lines)


def format_comparison_row(
    model_name: str,
    dataset_name: str,
    task: str,
    fp32_metrics: Any,
    quant_metrics: Any,
    quant_mode: str,
    quant_stats: Dict[str, Any],
) -> str:
    """
    Format a side-by-side FP32 vs quantized comparison.
    
    Args:
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        task: 'classification' or 'detection'.
        fp32_metrics: FP32 baseline metrics.
        quant_metrics: Quantized model metrics.
        quant_mode: 'dynamic' or 'static'.
        quant_stats: Quantization stats dict.
        
    Returns:
        Formatted comparison string.
    """
    precision_label = quant_mode.upper()
    lines = [
        f"┌{'─' * 64}┐",
        f"│ {('FP32 vs ' + precision_label + ' Comparison'):^62} │",
        f"├{'─' * 64}┤",
        f"│ Model:    {model_name:<52} │",
        f"│ Dataset:  {dataset_name:<52} │",
        f"├{'─' * 25}┬{'─' * 18}┬{'─' * 18}┤",
        f"│ {'Metric':<23} │ {'FP32':^16} │ {precision_label:^16} │",
        f"├{'─' * 25}┼{'─' * 18}┼{'─' * 18}┤",
    ]

    if task == "classification":
        fp_t1 = fp32_metrics.top1_accuracy if hasattr(fp32_metrics, 'top1_accuracy') else fp32_metrics.get("top1_accuracy", 0)
        fp_t5 = fp32_metrics.top5_accuracy if hasattr(fp32_metrics, 'top5_accuracy') else fp32_metrics.get("top5_accuracy", 0)
        q_t1 = quant_metrics.top1_accuracy if hasattr(quant_metrics, 'top1_accuracy') else quant_metrics.get("top1_accuracy", 0)
        q_t5 = quant_metrics.top5_accuracy if hasattr(quant_metrics, 'top5_accuracy') else quant_metrics.get("top5_accuracy", 0)
        d_t1 = q_t1 - fp_t1
        d_t5 = q_t5 - fp_t5
        sign1 = "+" if d_t1 >= 0 else ""
        sign5 = "+" if d_t5 >= 0 else ""
        lines.extend([
            f"│ {'Top-1 Accuracy':<23} │ {fp_t1:>13.2f}% │ {q_t1:>13.2f}% │",
            f"│ {'Top-5 Accuracy':<23} │ {fp_t5:>13.2f}% │ {q_t5:>13.2f}% │",
            f"├{'─' * 25}┼{'─' * 18}┼{'─' * 18}┤",
            f"│ {'Δ Top-1':<23} │ {'':16} │ {sign1}{d_t1:>12.2f}pp │",
            f"│ {'Δ Top-5':<23} │ {'':16} │ {sign5}{d_t5:>12.2f}pp │",
        ])
    else:
        fp_map = fp32_metrics.get("mAP", 0) if isinstance(fp32_metrics, dict) else 0
        fp_map50 = fp32_metrics.get("mAP50", 0) if isinstance(fp32_metrics, dict) else 0
        q_map = quant_metrics.get("mAP", 0) if isinstance(quant_metrics, dict) else 0
        q_map50 = quant_metrics.get("mAP50", 0) if isinstance(quant_metrics, dict) else 0
        d_map = q_map - fp_map
        d_map50 = q_map50 - fp_map50
        s_m = "+" if d_map >= 0 else ""
        s_m50 = "+" if d_map50 >= 0 else ""
        lines.extend([
            f"│ {'mAP@0.5:0.95':<23} │ {fp_map:>13.2f}% │ {q_map:>13.2f}% │",
            f"│ {'mAP@0.50':<23} │ {fp_map50:>13.2f}% │ {q_map50:>13.2f}% │",
            f"├{'─' * 25}┼{'─' * 18}┼{'─' * 18}┤",
            f"│ {'Δ mAP':<23} │ {'':16} │ {s_m}{d_map:>12.2f}pp │",
            f"│ {'Δ mAP50':<23} │ {'':16} │ {s_m50}{d_map50:>12.2f}pp │",
        ])

    orig = quant_stats.get("original_size_mb", 0)
    qsize = quant_stats.get("quantized_size_mb", 0)
    lines.extend([
        f"├{'─' * 25}┼{'─' * 18}┼{'─' * 18}┤",
        f"│ {'Model Size':<23} │ {orig:>12.1f} MB │ {qsize:>12.1f} MB │",
        f"│ {'Compression':<23} │ {'':16} │ {quant_stats.get('compression_ratio', 0):>13.2f}x │",
        f"└{'─' * 25}┴{'─' * 18}┴{'─' * 18}┘",
    ])

    return "\n".join(lines)


def format_final_summary(
    all_results: Dict[str, Dict[str, Any]],
    task_types: Dict[str, str],
) -> str:
    """
    Format final summary of all results.
    
    Args:
        all_results: Nested dict of {model_name: {dataset_name: metrics}}.
        task_types: Dict mapping dataset names to task types.
        
    Returns:
        Formatted summary string.
    """
    width = 70
    lines = [
        "",
        "═" * width,
        f"{'FINAL SUMMARY':^{width}}",
        "═" * width,
        "",
    ]
    
    # Collect all results for table format
    classification_results = []
    detection_results = []
    
    for model_name, model_results in all_results.items():
        for dataset_name, metrics in model_results.items():
            task = task_types.get(dataset_name, "classification")
            
            result = {
                "model_name": model_name,
                "dataset": dataset_name,
                "precision": "fp32",  # Will be extended for quantization
                "metrics": metrics,
            }
            
            if task == "detection":
                detection_results.append(result)
            else:
                classification_results.append(result)
    
    if classification_results:
        lines.append("  CLASSIFICATION")
        lines.append("  " + "─" * (width - 4))
        for r in classification_results:
            line = format_summary_line("classification", r["model_name"], r["dataset"], r["metrics"])
            lines.append(f"  {line}")
        lines.append("")
    
    if detection_results:
        lines.append("  DETECTION")
        lines.append("  " + "─" * (width - 4))
        for r in detection_results:
            line = format_summary_line("detection", r["model_name"], r["dataset"], r["metrics"])
            lines.append(f"  {line}")
        lines.append("")
    
    lines.append("═" * width)
    
    return "\n".join(lines)
