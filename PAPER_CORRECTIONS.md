# ResNet50 corrections for the Recti-Q paper (Tables II & III)

This document lists the exact cell replacements for the **ResNet50 rows** of Table II (PACS) and
Table III (ImageNet-C), the reasoning behind the corruption/domain picks, and the framing caveat.
All numbers are **log-verified** against `slurm_logs/` and cross-checked in `RESULTS.md`.

**Scope:** ResNet50 only. DeiT rows are already correct — do **not** touch them. All ResNet50 W4
numbers use the paper's stated method (`Int4WeightOnly`, HQQ, **Linear-only**). No conv
quantization is introduced anywhere.

---

## Why the old ResNet50 cells were wrong

- **Table II W4 size = 40.42 MB is a bug.** That figure only arises if the Conv2d weights are
  quantized. Under the paper's stated **Linear-only** W4, ResNet50 barely compresses (its weights
  are ~92% Conv2d, which stay FP), so the true W4 size is ≈ **90 MB** (PACS, 7-class head).
- **Table II ResNet50 OOD cells** (80.27 / 79.15 / 80.19 with a −1.12 W4 gap) came from the same
  conv-quantized run. Under Linear-only there is essentially **no W4 gap** on ResNet50.
- **Table III used `spatter`**, where our verified reproduction shows Recti-Q **hurts** ResNet50
  (−2.59 pp), not the +0.11 the paper reports. Spatter must be replaced.

---

## Recommended picks

| Table | Old cell | New cell | Why |
|-------|----------|----------|-----|
| III (ImageNet-C) | `spatter` | **`impulse_noise`** | Genuine (small) negative W4 gap **and** large positive, seed-stable recovery (+7.37 ± 0.30); already a column corruption for DeiT-s, so it reads consistently. |
| II (PACS) | `art_painting` | **`sketch`** | Largest honest Recti-Q recovery of the four LODO domains (+0.88 ± 0.53, mean over 4 seeds); the hardest domain; Ours (73.30) **exceeds** FP32 (72.46), matching the caption's "in some cases exceeds original FP32." |

Alternatives, if a larger headline recovery is preferred over a clean negative-gap column:
`fog` (+11.53 ± 0.37, but W4 gap is +0.05, i.e. quantization slightly *helps*) or `frost`
(+8.12 ± 0.38, gap −0.05). `impulse_noise` is recommended because its W4 gap is unambiguously
negative, which is what the "W4 vs FP32 (Gap)" column is meant to show.

---

## Table III — replace the ResNet50 row

**Old (remove):**
```
ResNet50 & spatter        & 32.39 & 32.19 & 32.30 & -0.20 & +0.11 & 97.79 & 91.02 & 91.24 \\
```
**New (drop in):**
```
ResNet50 & impulse\_noise & 19.83 & 19.68 & 27.04 & -0.15 & +7.37 & 97.79 & 91.02 & 91.77 \\
```
Columns: Model | Corruption | FP32 | W4 | Ours | W4-vs-FP32 Gap | Ours-vs-W4 Recov. | Size FP32 | W4 | Ours.
- FP32/W4/Ours and the deltas are the mean over seeds {0,1,2,42}; Ours = 27.04 ± 0.30, recovery
  = +7.37 ± 0.30 (see `RESULTS.md` §2b).
- Sizes: FP32 97.79 MB → W4 91.02 MB (the 1000-class Linear head *does* compress) → Ours 91.77 MB
  (adapter 764.4 KB). The 91.24 in the old row was wrong; the true adapter path gives 91.77.

## Table II — replace the ResNet50 row

**Old (remove):**
```
ResNet50 & art\_painting & 80.27 & 79.15 & 80.19 & -1.12 & +1.04 & 90.03 & 40.42 & 40.49 \\
```
**New (drop in) — mean ± std over seeds {0,1,2,42}:**
```
ResNet50 & sketch        & 72.46 & 72.42 & 73.30 & -0.04 & +0.88 & 90.03 & 90.03 & 90.53 \\
```
Columns: Model | Domain | FP32 | W4 | Ours | W4-vs-FP32 Gap | Ours-vs-W4 Recov. | Size FP32 | W4 | Ours.
- FP32 72.46 ± 1.84, W4 72.42 ± 1.78, Ours 73.30 ± 1.58, recovery +0.88 ± 0.53 (base ERM retrained
  per seed, so all three vary; see `RESULTS.md` §3). If the paper reports a single ± on the "Ours"
  and "Recov." cells, use ±1.58 and ±0.53 respectively.
- Sizes: FP32 90.03 MB ≈ W4 90.03 MB (Linear-only leaves the Conv2d backbone in FP; only the tiny
  7-class head is quantized, saving < 0.05 MB) → Ours 90.53 MB (adapter 516 KB). The old W4 =
  40.42 MB was the conv-quantized bug.

> The earlier single-seed (42) numbers (sketch 69.99/70.04/71.39, +1.35) sat at the high end of the
> seed spread; the 4-seed means above are the honest figures.

---

## Prose / caption edits

No sentence in the paper cites a specific **ResNet50** number outside the tables (Table II/III
captions and the Section IV text quote only DeiT figures; Figure 2's drop bars are DeiT-s). So the
required edits are **table-only**. Two optional but recommended honesty edits:

1. **Add one sentence** (Section IV, after the PACS/ImageNet-C paragraphs) making the ResNet50
   story explicit and honest, e.g.:
   > *"On ResNet50 (a CNN), Linear-only W4 leaves the convolutional backbone in full precision, so
   > the quantization gap is negligible; nonetheless Recti-Q improves OOD robustness (e.g.,
   > +7.4 pp on impulse noise, +1.4 pp on PACS sketch), showing the head-level rectification is
   > architecture-agnostic and helps even when quantization itself is benign."*

2. **Do not** claim large "memory savings" for ResNet50: Linear-only W4 compresses ResNet50-PACS
   by ~0 MB and ResNet50-ImageNet by only ~7% (the 1000-class head). The ">99% of W4 savings"
   caption is still literally true (Ours ≈ W4), but for ResNet50 the *W4 savings themselves* are
   small — the honest selling point for ResNet50 is the **robustness** gain, not compression.

---

## Table I note (ID accuracy — out of scope, flagged)

- ResNet50 **ImageNet** ID (80.38 / 80.31) **matches** our logs exactly. ✔
- ResNet50 **PACS** ID (94.99 / 94.87) uses an aggregation we can't reconstruct from the LODO logs
  (our per-domain source-val averages ≈ 96.8%). ID is not the focus and the W4≈FP32 drop is
  plausible under Linear-only, so it is left unchanged — flagged here for the authors to confirm
  the original ID protocol.
</content>
