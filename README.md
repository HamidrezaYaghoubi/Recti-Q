# Recti-Q: Feature-Space Rectification for OOD-Robust Quantized Perception

**"Recti-Q: Feature-Space Rectification for Out-of-Distribution-Robust Quantized Perception in Edge Robotics"** — *Accepted at IROS 2026*

Recti-Q closes the **Quantization-Induced Robustness Gap**: 4-bit Post-Training Quantization (PTQ)
keeps in-distribution (ID) accuracy but degrades out-of-distribution (OOD) robustness. Recti-Q
recovers it with a tiny LoRA adapter on the **pre-classifier features** `u` of the *frozen*
quantized backbone:

```
z = z_q + B(A(u)) · (α/r)      # B zero-initialized, rank r=64, α=16
```

Only the adapter is trained (<1% of parameters, a few hundred KB), so it doubles as a
low-bandwidth Over-The-Air patch. Three configs are compared: **FP32** (upper bound),
**PTQ-W4** (problem baseline), **Recti-Q** (ours).

---

## Install

```bash
conda create -n rectiq python=3.10 && conda activate rectiq
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt      # timm, torchao, pyyaml, ...
```

> **Note:** use `torchao==0.13.x` with PyTorch 2.7.1 (0.15 has a CUDA-extension incompatibility
> with 2.7.1 that disables the int4 kernels).

---

## Data

Set the dataset roots inside the YAML config (`configs/*.yaml`). Expected layout:

```
imagenet/validation/val/<wnid>/*.JPEG     # ImageNet-1k val (ID + adapter validation)
imagenet/train/train/<wnid>/*.JPEG        # ImageNet-1k train (5% class-balanced subset)
imagenet_c/<corruption>/<severity>/<wnid>/*.JPEG   # ImageNet-C (https://github.com/hendrycks/robustness)
pacs/pacs_data/pacs_data/<domain>/<class>/*.jpg    # PACS + official pacs_label/*_kfold.txt splits
```

`timm` backbones (resnet50, deit_tiny/small/base) download automatically on first run.

---

## Reproduce the DeiT results (ImageNet-C)

```bash
# All three DeiT models: FP32 -> PTQ-W4 -> Recti-Q, on ImageNet-C @ severity 5
python -m src.main --config configs/imagenet_c_rectiq.yaml \
    --models deit_tiny deit_small deit_base --no-wandb
```

The pipeline trains each adapter source-only on a 5% class-balanced ImageNet-1k train subset
(5 epochs, AdamW lr=3e-4, cosine, KD to the FP32 teacher at T=4), then reports top-1 on
ImageNet val (ID) and ImageNet-C (OOD).

**Expected numbers** (top-1 %, OOD = mean over contrast + gaussian/shot/impulse noise @ sev 5;
single A5000, seed 42):

| Model | ID FP32→W4 | OOD FP32 | OOD W4 | OOD Recti-Q | Recovery | Size FP32→W4 |
|-------|:----------:|:--------:|:------:|:-----------:|:--------:|:------------:|
| DeiT-tiny  | 72.16 → 71.54 | 18.00 | 17.33 | **17.59** | +0.26 | 21.9 → 16.3 MB |
| DeiT-small | 79.85 → 78.94 | 33.92 | 29.39 | **29.52** | +0.13 | 84.2 → 26.2 MB |
| DeiT-base  | 81.98 → 81.50 | 45.38 | 44.10 | **45.09** | +0.99 | 330.3 → 56.5 MB |

PTQ-W4 preserves ID (≤0.6 pp drop) but opens an OOD gap that grows with corruption severity;
Recti-Q recovers part of it while keeping >99% of the W4 memory savings.

### Pre-trained adapters

Trained Recti-Q adapters (a few hundred KB each) are in [`adapters/`](adapters/) for direct
verification without retraining:

```
adapters/deit_tiny_imagenet_c_rectiq.pt      (301 KB)
adapters/deit_small_imagenet_c_rectiq.pt     (349 KB)
adapters/deit_base_imagenet_c_rectiq.pt      (445 KB)
adapters/resnet50_imagenet_c_rectiq.pt       (764 KB)   # ResNet50, ImageNet-C
adapters/resnet50_pacs_sketch_rectiq.pt      (516 KB)   # ResNet50, PACS (sketch held out)
```

Each holds the LoRA `A`/`B` weights plus metadata (rank, alpha, feat_dim, num_classes); load with
`src.rectiq.load_adapter`.

---

## Reproduce the ResNet50 results

ResNet50 is a CNN: ~92% of its weights are in `nn.Conv2d`, which the paper's **Linear-only** W4
(`Int4WeightOnly`, HQQ) leaves in full precision. So on ResNet50, W4 barely compresses and barely
degrades — but Recti-Q still improves OOD robustness, showing the head-level rectification is
architecture-agnostic.

```bash
# ImageNet-C: all 15 corruptions @ severity 5, Linear-only W4 (add --seed for the seed sweep)
python -m src.main --config configs/imagenet_c_resnet_linear_all.yaml --no-wandb --seed 42

# PACS: leave-one-domain-out over all four domains, Linear-only W4
python -m src.main --config configs/pacs_resnet_linear_all.yaml --no-wandb --seed 42
```

**Expected numbers** (top-1 %, mean over seeds {0,1,2,42} for ImageNet-C; single A5000):

| Benchmark | Split | FP32 | W4 | Recti-Q | Recovery | Size FP32→W4 |
|-----------|-------|:----:|:--:|:-------:|:--------:|:------------:|
| ImageNet-C | impulse_noise @ sev5 | 19.83 | 19.68 | **27.04 ± 0.30** | +7.37 ± 0.30 | 97.79 → 91.02 MB |
| ImageNet-C | fog @ sev5           | 37.67 | 37.72 | **49.24 ± 0.37** | +11.53 ± 0.37 | 97.79 → 91.02 MB |
| PACS (LODO) | sketch held out      | 72.46 ± 1.84 | 72.42 ± 1.78 | **73.30 ± 1.58** | +0.88 ± 0.53 | 90.03 → 90.03 MB |

(ImageNet-C = mean over seeds {0,1,2,42}; PACS = mean over seeds {0,1,2,42}, base ERM retrained per
seed. On PACS, Linear-only W4 leaves ResNet50 size unchanged — only the 7-class head is a Linear.)

The full per-corruption / per-domain breakdown (per-seed and mean±std) is in
[`RESULTS.md`](RESULTS.md). On ResNet50 the W4 gap is negligible (Linear-only), so the recovery is
a genuine OOD head-reshaping gain rather than quantization repair; it is corruption-dependent
(large positive on noise/fog/frost, negative on brightness/blur/pixelate/jpeg/spatter — see
`RESULTS.md` §2b).

### Other runs

```bash
python -m src.main --config configs/pacs_rectiq.yaml --no-wandb      # DeiT PACS leave-one-domain-out
python -m src.main --config configs/imagenet_c_rectiq.yaml --debug   # fast sanity check
sbatch scripts/slurm_rectiq.sh configs/imagenet_c_rectiq.yaml        # SLURM (UMIACS Nexus Gamma)
```

---

## Repository layout

```
src/main.py                    # FP32 -> W4 -> Recti-Q pipeline
src/rectiq.py                  # ClassifierLoRA, RectiQModel, train_rectiq_adapter, save/load_adapter
src/models/classification.py   # TimmClassifier, forward_features_logits(backbone, x) -> (u, z)
src/quantization/quantizer.py  # torchao W4 = Int4WeightOnly(use_hqq=True)
src/datasets/                  # imagenet (5% subset), imagenet_c, pacs (LODO)
configs/                       # experiment YAMLs
adapters/                      # pre-trained Recti-Q adapters
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
