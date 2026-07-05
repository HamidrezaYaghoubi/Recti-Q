# Recti-Q — Reproduction Results

All numbers are top-1 accuracy (%). Single NVIDIA RTX A5000, `saficiency` env
(torch 2.7.1+cu126, torchao 0.13, timm 1.0.20). Recti-Q adapter: rank 64, α 16,
KD λ=1.0 (T=4), AdamW lr 3e-4, cosine, 5 epochs (ImageNet-C) / 10 (PACS), trained
source-only on a 5% class-balanced ImageNet-1k train subset. ImageNet-C is severity 5.

**W4 methods for ResNet50** (a CNN, ~92% Conv2d weights):
- **Linear-only** — `torchao Int4WeightOnly(use_hqq=True)` (the paper's stated method); quantizes
  only `nn.Linear`, so ResNet50 barely compresses (~91 MB) and barely degrades.
- **int4-conv** — additionally 4-bit-quantizes `nn.Conv2d` (`IntConv2d`, HQQ) + BatchNorm
  recalibration; real compression (~13.5 MB) but a genuine OOD gap the head adapter can't recover.

DeiT models are Linear-heavy, so `torchao` Linear-only already compresses them well.

---

## 1. DeiT — ImageNet-C (mean over contrast + gaussian/shot/impulse noise)

| Model | ID FP32→W4 | OOD FP32 | OOD W4 | OOD Recti-Q | Recovery | Size FP32→W4 |
|-------|:----------:|:--------:|:------:|:-----------:|:--------:|:------------:|
| DeiT-tiny  | 72.16 → 71.54 | 18.00 | 17.33 | 17.59 | +0.26 | 21.9 → 16.3 MB |
| DeiT-small | 79.85 → 78.94 | 33.92 | 29.39 | 29.52 | +0.13 | 84.2 → 26.2 MB |
| DeiT-base  | 81.98 → 81.50 | 45.38 | 44.10 | 45.09 | +0.99 | 330.3 → 56.5 MB |

> **ID matches the paper's Table I exactly** (72.16/71.54, 79.85/78.94, 81.98/81.50) and W4 sizes
> match Table II/III. W4 opens the OOD gap; Recti-Q recovers a small, positive slice. *Note:* our
> per-corruption recovery is smaller than the paper's Table III values (see §1a) — the baselines
> reproduce exactly, the adapter recovery is weaker (a recipe-tuning gap, not a baseline error).

### 1a. DeiT-small per-corruption (ImageNet-C @ sev5)

| Corruption | FP32 | W4 | Recti-Q | W4 gap | Recovery | Paper "Ours" (recov) |
|------------|:----:|:--:|:-------:|:------:|:--------:|:--------------------:|
| contrast       | 39.45 | 33.67 | 34.00 | −5.78 | +0.33 | 36.52 (+2.68) |
| gaussian_noise | 33.10 | 28.84 | 28.95 | −4.26 | +0.11 | 29.30 (+0.42) |
| impulse_noise  | 32.80 | 28.64 | 28.68 | −4.16 | +0.04 | 29.19 (+0.27) |
| shot_noise     | 30.32 | 26.39 | 26.44 | −3.93 | +0.05 | 26.67 (+0.37) |

FP32 and W4 match the paper within ≤0.2 pp; Recti-Q recovery is directionally correct but smaller.

---

## 2. ResNet50 — ImageNet-C

### 2a. Summary (mean over contrast/gaussian/shot/impulse; + spatter)

| Method | Corruptions | FP32 | W4 | Recti-Q | W4 gap | Recovery | Size FP32→W4 |
|--------|:-----------:|:----:|:--:|:-------:|:------:|:--------:|:------------:|
| Linear-only | 4-corruption mean | 21.80 | 21.67 | 27.16 | −0.13 | +1.49 | 97.8 → 91.0 MB |
| Linear-only | spatter | 32.45 | 32.21 | 29.62 | −0.24 | **−2.59** | 97.8 → 91.0 MB |
| int4-conv | 4-corruption mean | 21.80 | 18.32 | 18.02 | −3.48 | −0.30 | 97.8 → 13.5 MB |
| int4-conv | spatter | 32.45 | 27.47 | 27.20 | −4.98 | −0.27 | 97.8 → 13.5 MB |

> Under **Linear-only** the W4 gap is negligible (≤0.24), so there is essentially no
> quantization-robustness gap to recover on ResNet50; Recti-Q's effect is a general head
> reshaping that is **corruption-dependent** (see §2b). Under **int4-conv** a real gap appears but
> the head-only adapter cannot recover it. Our FP32/W4 **spatter** matches the paper (32.39/32.19)
> but our Recti-Q **hurts** spatter (−2.59) vs the paper's +0.11.

### 2b. ResNet50 Linear-only per-corruption — per seed and combined (mean ± std over seeds {0,1,2,42})

`FP32`/`W4` are deterministic (adapter-independent). `RQ_s*` = Recti-Q OOD for that seed.

| Corruption | FP32 | W4 | RQ s0 | RQ s1 | RQ s2 | RQ s42 | **Recti-Q (mean±std)** | **Recovery (mean±std)** |
|------------|:----:|:--:|:-----:|:-----:|:-----:|:------:|:----------------------:|:-----------------------:|
| fog               | 37.67 | 37.72 | 48.94 | 49.44 | 48.84 | 49.75 | 49.24 ± 0.37 | **+11.53 ± 0.37** |
| frost             | 31.89 | 31.84 | 39.88 | 40.52 | 39.46 | 39.98 | 39.96 ± 0.38 | +8.12 ± 0.38 |
| impulse_noise     | 19.83 | 19.68 | 26.71 | 27.14 | 26.83 | 27.49 | 27.04 ± 0.30 | +7.37 ± 0.30 |
| gaussian_noise    | 20.68 | 20.54 | 27.13 | 27.52 | 27.22 | 27.94 | 27.45 ± 0.32 | +6.92 ± 0.31 |
| shot_noise        | 21.99 | 21.89 | 27.89 | 28.41 | 28.04 | 28.61 | 28.24 ± 0.29 | +6.35 ± 0.29 |
| glass_blur        |  9.12 |  8.99 |  9.15 |  9.44 |  9.05 |  9.35 |  9.25 ± 0.15 | +0.25 ± 0.16 |
| contrast          | 24.70 | 24.57 | 23.64 | 24.10 | 24.39 | 24.59 | 24.18 ± 0.36 | −0.39 ± 0.36 |
| zoom_blur         | 25.61 | 25.48 | 24.31 | 25.43 | 24.47 | 24.70 | 24.73 ± 0.43 | −0.75 ± 0.43 |
| motion_blur       | 20.09 | 20.00 | 18.99 | 19.03 | 18.98 | 19.11 | 19.03 ± 0.05 | −0.97 ± 0.05 |
| snow              | 29.17 | 29.13 | 27.82 | 28.48 | 27.94 | 28.24 | 28.12 ± 0.26 | −1.01 ± 0.26 |
| elastic_transform | 13.54 | 13.39 | 11.68 | 11.21 | 11.12 | 11.29 | 11.32 ± 0.21 | −2.06 ± 0.21 |
| jpeg_compression  | 49.18 | 49.04 | 46.65 | 46.38 | 46.39 | 46.60 | 46.51 ± 0.12 | −2.54 ± 0.12 |
| pixelate          | 23.66 | 23.49 | 20.40 | 20.16 | 19.85 | 20.00 | 20.10 ± 0.20 | −3.39 ± 0.20 |
| defocus_blur      | 19.57 | 19.49 | 15.99 | 16.24 | 15.52 | 16.01 | 15.94 ± 0.26 | −3.55 ± 0.27 |
| brightness        | 66.41 | 66.22 | 62.10 | 62.13 | 61.81 | 62.21 | 62.06 ± 0.15 | **−4.16 ± 0.15** |
| spatter*          | 32.45 | 32.21 |   —   |   —   |   —   | 29.62 |     29.62    | −2.59 (seed 42) |

\* spatter was run once (seed 42), not in the seed sweep.

> **Seed robustness:** the sign and magnitude of the per-corruption recovery are highly stable
> (std ≤ 0.43 across seeds). Recti-Q consistently **helps** noise/fog/frost (+6 to +12) and
> consistently **hurts** brightness/blur/pixelate/jpeg/spatter (−2 to −4). This is a structural
> property of a clean-trained head adapter on a CNN, not training randomness.

---

## 3. ResNet50 — PACS (leave-one-domain-out; OOD = held-out domain)

Base model: 30-epoch ERM source fine-tune (source-val ≈ 96–98%). *Accuracy is from this
reconstruction, not the paper's original PACS checkpoints.*

| Target domain | Method | FP32 | W4 | Recti-Q | W4 gap | Recovery | Size W4 |
|---------------|:------:|:----:|:--:|:-------:|:------:|:--------:|:-------:|
| photo         | Linear-only | 98.32 | 98.26 | 98.14 | −0.06 | −0.12 | 90.0 MB |
| art_painting  | Linear-only | 82.18 | 82.23 | 82.62 | +0.05 | +0.39 | 90.0 MB |
| cartoon       | Linear-only | 74.53 | 74.53 | 75.34 |  0.00 | +0.81 | 90.0 MB |
| sketch        | Linear-only | 69.99 | 70.04 | 71.39 | +0.05 | **+1.35** | 90.0 MB |
| art_painting  | int4-conv   | 83.20 | 82.71 | 81.88 | −0.49 | −0.83 | 12.4 MB |

> Linear-only W4 gap is negligible on PACS too; Recti-Q recovery is positive on the harder domains
> (sketch +1.35, cartoon +0.81) and ~0 on the easy photo domain.

---

## 4. Model sizes (MB) — corrected

| Model | Head | FP32 | Linear-only W4 | int4-conv W4 | Recti-Q adds |
|-------|:----:|:----:|:--------------:|:------------:|:------------:|
| ResNet50 | 1000-cls (ImageNet) | 97.79 | **91.0** | 13.5 | +0.76 |
| ResNet50 | 7-cls (PACS) | 90.03 | **90.0** | 12.4 | +0.52 |
| DeiT-tiny  | 1000 | 21.9 | 16.3 | — | +0.30 |
| DeiT-small | 1000 | 84.2 | 26.2 | — | +0.35 |
| DeiT-base  | 1000 | 330.3 | 56.5 | — | +0.44 |

> The paper's **Table II ResNet50 W4 = 40.42 MB is a bug**: under the stated Linear-only method it
> is ≈90 MB; the true 4-bit-conv size is ≈12.4 MB. Table III's 91.02 MB (ImageNet) is correct.
