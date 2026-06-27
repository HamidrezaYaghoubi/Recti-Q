# Recti-Q: Feature-Space Rectification for OOD-Robust Quantized Perception

**"Recti-Q: Feature-Space Rectification for Out-of-Distribution-Robust Quantized Perception in Edge Robotics"**
*Accepted at IROS 2026*

---

## Overview

Deploying quantized neural networks on edge robots reduces memory and compute cost, but
**4-bit Post-Training Quantization (PTQ) opens a Quantization-Induced Robustness Gap**: models
maintain in-distribution (ID) accuracy while their out-of-distribution (OOD) robustness degrades
substantially — a silent failure mode in safety-critical systems.

**Recti-Q** closes this gap with a small feature-space rectification adapter attached to the frozen
quantized backbone. The adapter is a LoRA residual on pre-classifier features (`u`):

```
z = z_q + B(A(u)) · (α/r)
```

B is zero-initialized (so at init, Recti-Q is identical to the PTQ baseline). With rank r=64 and
α=16, the adapter contains under 1% of model parameters — as small as 6 KB — while recovering the
OOD robustness lost to quantization. Because only the adapter weights need to be transmitted, Recti-Q
enables low-bandwidth Over-The-Air (OTA) model patching on resource-constrained robots.

---

## Method

### Three configurations

| Config | Description |
|--------|-------------|
| **FP32** | Full-precision baseline (upper bound) |
| **PTQ-W4** | 4-bit weight-only PTQ via torchao HQQ (calibration-free); backbone frozen |
| **Recti-Q** | PTQ-W4 backbone + LoRA adapter trained on pre-classifier features |

### Models

Image classification with `timm` pretrained models: `resnet50`, `deit_tiny_patch16_224`,
`deit_small_patch16_224`, `deit_base_patch16_224`. The method is architecture-agnostic (CNN +
Transformer).

### Quantization

4-bit weight-only via torchao `Int4WeightOnlyConfig(use_hqq=True)`. HQQ requires no calibration
data. Only `nn.Linear` layers are quantized; the backbone is then frozen.

### Training

The adapter is trained source-only on a 5% class-balanced subsample of the ImageNet-1k *train*
split. No OOD data is seen during training (leakage-free protocol).

Loss: `L = L_CE + λ·L_KD` where KD distills from a frozen FP32 teacher at temperature T=4.
Setting λ=0 gives the **teacher-free** variant (no FP32 teacher needed at deployment time).

Optimizer: AdamW, lr=3e-4, wd=1e-4, cosine LR schedule.
- ImageNet-C: 5 epochs, train batch 128, val/test batch 256.
- PACS: 5–10 epochs (leave-one-domain-out).

### Ablations

- KD weight λ ∈ {0, 0.5, 1.0} — includes teacher-free setting.
- Adapter space: feature-space (pre-classifier, ours) vs logit-space.
- LoRA rank r ∈ {4, 8, 16, 32, 64}.

---

## Results

Recti-Q recovers the OOD robustness lost to PTQ-W4 across corruption types (ImageNet-C) and
domain shifts (PACS), while retaining more than 99% of PTQ's memory savings relative to FP32.
Ablations confirm that feature-space rectification outperforms logit-space adaptation, and that
higher LoRA rank (r=64) yields the best recovery. The teacher-free variant (λ=0) remains
competitive, removing the need for a live FP32 model during deployment. See the paper tables for
full quantitative results.

---

## Installation

**Requirements:** Python 3.10, CUDA 12.6, PyTorch 2.7.1

```bash
# Create and activate conda environment
conda create -n quant python=3.10
conda activate quant

# Install PyTorch with CUDA 12.6
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126

# Install project dependencies
pip install -r requirements.txt
```

---

## Quick Start

All experiments are config-driven. The pipeline runs FP32 eval → PTQ-W4 eval → Recti-Q
adapter training + eval for each model, logging each phase to wandb.

```bash
# ImageNet-C benchmark (train on 5% ImageNet-1k, eval on corruptions)
python -m src.main --config configs/imagenet_c_rectiq.yaml

# PACS benchmark (leave-one-domain-out)
python -m src.main --config configs/pacs_rectiq.yaml

# Debug mode (small subset, fast sanity check)
python -m src.main --config configs/imagenet_c_rectiq.yaml --debug

# Disable wandb logging
python -m src.main --config configs/imagenet_c_rectiq.yaml --no-wandb
```

### Cluster (UMIACS Nexus Gamma)

```bash
# Submit full pipeline as a batch job
sbatch scripts/slurm_rectiq.sh
```

Partition: `gamma`, account: `gamma`. The script activates the `quant` conda env automatically.

---

## Dataset Layout

Datasets live under `/fs/nexus-projects/pc_driving/yaghoubi/datasets/`:

| Dataset | Path | Used for |
|---------|------|---------|
| ImageNet-1K | `.../imagenet` | 5% training subsample |
| ImageNet-C | `.../imagenet_c` | OOD eval (corruptions × severities) |
| PACS | `.../pacs` | OOD eval (domain shift, LODO) |

Paths are configurable per-dataset inside each YAML file.

---

## Repository Structure

```
Recti-Q/
├── configs/                    # YAML experiment configs
├── src/
│   ├── main.py                 # Pipeline entry point
│   ├── rectiq.py               # ClassifierLoRA, RectiQModel, train_rectiq_adapter
│   ├── models/
│   │   ├── base.py             # BaseModel, ModelOutput
│   │   ├── factory.py          # timm-backed ModelFactory and registry
│   │   └── classification.py  # TimmClassifier, forward_features_logits
│   ├── datasets/
│   │   ├── imagenet.py         # ImageNet loader + 5% balanced subset
│   │   ├── imagenet_c.py       # ImageNet-C corruption loader
│   │   └── pacs.py             # PACS leave-one-domain-out splits
│   ├── quantization/
│   │   └── quantizer.py        # torchao PTQ (W4 = Int4WeightOnly HQQ)
│   ├── evaluation/
│   │   └── metrics.py          # MetricsComputer, ClassificationMetrics (top-1/5)
│   └── utils/
│       ├── config.py           # ExperimentConfig, QuantizationConfig, RectiQConfig
│       └── logging.py          # wandb integration
├── scripts/
│   └── slurm_rectiq.sh         # Nexus Gamma batch job script
├── requirements.txt
└── Recti-Q IROS Submission.pdf # Accepted paper (source of truth)
```

---

## Citation

```bibtex
@inproceedings{rectiq2026,
  title     = {Recti-Q: Feature-Space Rectification for Out-of-Distribution-Robust
               Quantized Perception in Edge Robotics},
  author    = {Yaghoubi, Hamidreza and others},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

---

## License

MIT License
