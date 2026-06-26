# Quantization Decision Analysis

**Investigating how neural network quantization alters model decisions in critical and edge cases**

*Research project for IROS 2026*

## Overview

This project demonstrates that quantized models (INT8, INT4) maintain high average accuracy but exhibit systematic decision changes that disproportionately affect:
- Edge cases and difficult examples
- Small objects in detection tasks
- Corrupted/noisy inputs
- Underrepresented classes

## Requirements

- Python 3.10
- CUDA 12.6
- PyTorch 2.7.1

## Installation

### 1. Create conda environment

```bash
conda create -n saficiency python=3.10
conda activate saficiency
```

### 2. Install PyTorch with CUDA support

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Download COCO dataset

```bash
chmod +x scripts/download_coco.sh
./scripts/download_coco.sh
```

## Project Structure

```
quantization-decision-analysis/
├── configs/                    # Configuration files
│   ├── baseline_classification.yaml
│   ├── baseline_imagenet_c.yaml
│   └── baseline_detection.yaml
├── src/                        # Source code
│   ├── models/                 # Model implementations
│   │   ├── base.py            # Abstract base class
│   │   ├── factory.py         # Model factory
│   │   └── classification.py  # ResNet, MobileNet, ViT
│   ├── datasets/              # Dataset loaders
│   │   ├── imagenet.py        # ImageNet-1K
│   │   ├── imagenet_c.py      # ImageNet-C (corruptions)
│   │   └── coco.py            # COCO 2017
│   ├── quantization/          # Quantization module (Week 2-3)
│   │   └── quantizer.py       # Quantization implementation
│   ├── evaluation/            # Evaluation metrics
│   │   └── metrics.py         # Top-1, Top-5, decision changes
│   ├── utils/                 # Utilities
│   │   ├── config.py          # Configuration management
│   │   ├── logging.py         # Logging with wandb
│   │   └── checkpoint.py      # Checkpoint management
│   └── main.py                # Main entry point
├── scripts/                   # Shell scripts
│   ├── run_baseline.sh        # Run baseline experiments
│   ├── download_coco.sh       # Download COCO dataset
│   └── slurm_run.sh          # SLURM cluster submission
├── checkpoints/               # Saved model checkpoints
├── results/                   # Experiment results
├── logs/                      # Log files
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

### Run baseline FP32 inference on ImageNet

```bash
# Full evaluation
python -m src.main --config configs/baseline_classification.yaml

# Quick debug run (100 samples)
python -m src.main --config configs/baseline_classification.yaml --debug

# Without wandb logging
python -m src.main --config configs/baseline_classification.yaml --no-wandb
```

### Run evaluation on ImageNet-C

```bash
python -m src.main --config configs/baseline_imagenet_c.yaml
```

### Run your detection experiments

```bash
# 1) Quantization on COCO (YOLO FP32 + INT8 export)
python -m src.main --config configs/quantize_yolo.yaml

# 2) Fine-tuning on BDD100K
python -m src.finetune --config configs/finetune_bdd100k_yolo.yaml

# 3) Quantization on BDD100K
python -m src.main --config configs/quantize_bdd100k_yolo.yaml

# 4) Recti-Q on BDD100K (runs quantization + Recti-Q phase when rectiq.enabled=true)
python -m src.main --config configs/quantize_bdd100k_yolo_finetuned.yaml
```

### Using shell scripts

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run baseline
./scripts/run_baseline.sh

# Run in debug mode
./scripts/run_baseline.sh --debug
```

### Submit to SLURM cluster

```bash
sbatch scripts/slurm_run.sh configs/baseline_classification.yaml
```

## Configuration

Configuration files are in YAML format. Here's an example:

```yaml
experiment:
  name: "baseline_classification"
  seed: 42
  device: "cuda"

models:
  - name: "resnet50"
    architecture: "resnet50"
    weights: "IMAGENET1K_V2"
    task: "classification"
    num_classes: 1000

datasets:
  imagenet:
    root: "/path/to/imagenet"
    split: "val"
    batch_size: 64

quantization:
  enabled: false  # Enable for quantization experiments

logging:
  wandb:
    enabled: true
    project: "quantization-decision-analysis"

output:
  save_predictions: true
  results_dir: "./results"
```

### Command-line overrides

```bash
# Override device
python -m src.main --config configs/baseline_classification.yaml --device cpu

# Evaluate specific models only
python -m src.main --config configs/baseline_classification.yaml --models resnet50 vit_base

# Evaluate specific datasets only
python -m src.main --config configs/baseline_classification.yaml --datasets imagenet
```

## Adding New Models

1. Create a new model class in `src/models/`:

```python
from src.models.base import BaseModel, ModelOutput
from src.models.factory import register_model

@register_model("my_model")
class MyModel(BaseModel):
    def __init__(self, weights: str = "DEFAULT", num_classes: int = 1000):
        super().__init__("my_model", task="classification", num_classes=num_classes)
        # Initialize your model...
    
    def forward(self, x):
        # Forward pass...
        pass
    
    def predict(self, x):
        # Return ModelOutput...
        pass
    
    def get_preprocessing_config(self):
        return {"input_size": 224, "mean": [...], "std": [...]}
```

2. Add to config:

```yaml
models:
  - name: "my_model"
    architecture: "my_model"
    weights: "DEFAULT"
```

## Adding New Datasets

1. Create a dataset class in `src/datasets/`:

```python
from src.datasets.base import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        # Load your data...
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # Return (image, label)...
        pass
```

2. Create a loader function:

```python
def get_my_dataset_loader(config, **kwargs):
    # Create and return DataLoader...
    pass
```

## Dataset Locations

Default dataset paths (configurable in YAML):

| Dataset | Path |
|---------|------|
| ImageNet-1K | `/fs/nexus-projects/pc_driving/yaghoubi/datasets/imagenet` |
| ImageNet-C | `/fs/nexus-projects/pc_driving/yaghoubi/datasets/imagenet_c` |
| COCO 2017 | `/fs/nexus-projects/pc_driving/yaghoubi/datasets/coco` |

## Output Files

Results are saved in the `results/` directory:

```
results/
└── baseline_classification/
    ├── resnet50_fp32_imagenet_20260203_120000.pkl      # Predictions
    ├── metrics_resnet50_fp32_imagenet_20260203_120000.json  # Metrics
    └── ...
```

## Logging

- **Console**: Colored output with progress bars
- **File**: Detailed logs in `logs/`
- **Weights & Biases**: Experiment tracking (optional)

Configure wandb:
```bash
wandb login
```

Or disable:
```bash
python -m src.main --config ... --no-wandb
```

## Week 1 Features (Implemented)

- [x] Project structure and configuration system
- [x] Dataset loaders (ImageNet, ImageNet-C, COCO)
- [x] Model interface with factory pattern
- [x] Classification models (ResNet50, MobileNetV2, ViT-Base)
- [x] Evaluation metrics (Top-1, Top-5 accuracy)
- [x] Logging with wandb integration
- [x] Checkpoint management
- [x] Main inference pipeline

## Week 2-4 Features (TODO)

- [ ] Post-Training Quantization (PTQ) implementation
- [ ] Calibration methods (MinMax, Percentile, Entropy)
- [ ] Decision change analysis between FP32 and quantized models
- [ ] Per-class and per-sample analysis
- [ ] Object detection support
- [ ] Visualization tools
- [ ] Novel metrics for edge case detection

## Citation

```bibtex
@inproceedings{quantization-decision-analysis,
  title={How Quantization Affects Neural Network Decisions: 
         A Systematic Analysis of Edge Cases},
  author={Your Name},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## License

MIT License
