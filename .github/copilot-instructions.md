# MoCo v3 Solar Panel Segmentation Project

## Project Overview

This is a computer vision research project implementing **MoCo v3** (Momentum Contrast v3) for self-supervised pretraining on the PV03 satellite imagery dataset. The pretrained encoder will later be used as a backbone for U-Net-based solar panel segmentation.

**Key characteristics:**
- Self-supervised learning on unlabeled satellite images
- Designed for comparative SSL studies (SimCLR, MoCo v3, BYOL)
- Strict reproducibility requirements (fixed seeds, explicit hyperparameters)
- Research-grade code quality

## Running the Project

### Setup
```bash
# Install dependencies (using uv)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Training Commands

**Single GPU:**
```bash
python main.py
```

**Multi-GPU with DDP:**
```bash
python main.py --multiprocessing-distributed --world-size 1 --rank 0
```

**Verification script:**
```bash
python -c "from src.utils.verify_training import verify_pipeline; verify_pipeline()"
```

### Evaluation

**Linear probing (evaluate spatial representation quality):**
```bash
python -m src.evaluation.linear_segmentation --checkpoint checkpoints/moco_pv03.pth
```

**Fine-tuning for segmentation:**
```bash
python -m src.evaluation.finetune_segmentation --checkpoint checkpoints/moco_pv03.pth
```

## Architecture Overview

### MoCo v3 Pipeline
The self-supervised learning follows this flow:
1. **Two augmented views** of each image are created
2. **Query encoder** (base + projector + predictor) processes view 1
3. **Momentum encoder** (base + projector, no gradients) processes view 2
4. **Contrastive loss** computed between query predictions and momentum keys
5. Momentum encoder updated via EMA (exponential moving average)

**Hybrid mode:** Optionally uses a memory queue (MoCo v2 style) in addition to batch negatives.

### Project Structure

```
src/
├── config.py              # Single source of truth for all hyperparameters
├── datasets/
│   ├── pv03_ssl.py       # Self-supervised dataset (images only, no labels)
│   └── pv03_seg.py       # Supervised dataset (images + masks) for evaluation
├── models/
│   ├── backbones.py      # ResNet-50, ViT-Small, ViT-Base implementations
│   ├── moco_v3.py        # Core MoCo v3 model (query/key encoders, predictor)
│   └── unet.py           # U-Net for downstream segmentation (not used in SSL)
├── training/
│   └── train_moco.py     # Main training loop with DDP support
├── utils/
│   ├── augmentations.py  # MoCo v3 augmentations (satellite-specific)
│   ├── checkpoints.py    # Save/load utilities
│   ├── logging.py        # AverageMeter, ProgressMeter
│   └── verify_training.py # Sanity check script
└── evaluation/
    ├── linear_segmentation.py  # Linear probe (frozen encoder + 1x1 conv)
    └── finetune_segmentation.py # Full fine-tuning
```

## Key Conventions

### Dataset Organization
- **SSL training:** Only uses `dataset/original/` (RGB images)
- **Evaluation:** Uses both `dataset/original/` and `dataset/labels/` (masks)
- **Never hardcode paths:** Always use `Config.dataset_path`

### Configuration Management
All hyperparameters live in `src/config.py` as a dataclass:
- Training: batch_size, epochs, learning_rate, warmup, etc.
- Model: backbone choice, feature_dim, mlp_dim
- Hardware: num_workers, seed, use_amp, use_grad_checkpointing
- Queue: use_queue (hybrid mode), queue_size

**Override via CLI if needed**, but defaults should be production-ready.

### Reproducibility
- Always set `seed=42` in Config
- Use deterministic CUDA operations where possible
- Document exact hyperparameters in experiment tags: `Config.get_experiment_tag()`
- No "latest" or "current" references in code

### Backbone Abstractions
`get_backbone(name)` returns `(model, output_dim)` tuple:
- `"resnet50"`: Returns ResNet-50 without final FC layer
- `"vit_small"`, `"vit_base"`: Vision Transformers with CLS token extraction
- Backbones are reusable across MoCo, U-Net encoder, and evaluation scripts

### Checkpoint Format
Saved checkpoints contain:
```python
{
    'epoch': int,
    'state_dict': model.state_dict(),  # Full MoCo v3 model
    'optimizer': optimizer.state_dict(),
    'config': config.__dict__,
    'encoder_state_dict': encoder_only_weights  # Just backbone for U-Net reuse
}
```

The `encoder_state_dict` key is critical for downstream transfer to U-Net.

### Augmentation Strategy
MoCo v3 uses two strong augmentations per image:
- `RandomResizedCrop(224)` with scale=(0.2, 1.0)
- `RandomHorizontalFlip` and `RandomVerticalFlip` (satellite images are orientation-agnostic)
- `ColorJitter(0.4, 0.4, 0.2, 0.1)` tuned for satellite RGB
- `RandomGrayscale(p=0.2)`
- `GaussianBlur` with varying kernel sizes

These are satellite-specific (e.g., vertical flip is valid, unlike natural images).

### DDP Training
- Use `torch.distributed` with `DistributedDataParallel`
- Launch via `--multiprocessing-distributed` flag
- Effective batch size = `Config.batch_size * world_size * num_gpus_per_node`
- Learning rate may need scaling based on effective batch size

### Framework Constraints
- **PyTorch only** (no TensorFlow, fastai, Lightning)
- **No external SSL libraries** (VISSL, Lightly, etc.)
- Implement MoCo v3 from first principles for research transparency

## Common Tasks

### Adding a New Backbone
1. Add factory function in `src/models/backbones.py`
2. Return tuple: `(model, output_feature_dim)`
3. Update `Config.backbone` options in docstring
4. Test with verification script

### Switching SSL Methods (SimCLR, BYOL)
The architecture is designed for easy swapping:
- Create `src/models/simclr.py` or `src/models/byol.py`
- Follow same interface: `__init__`, `forward`, `momentum_update`
- Swap import in `train_moco.py`
- All other infrastructure (datasets, augmentations, checkpoints) is reusable

### Debugging Training
Check these in order:
1. Run `verify_training.py` with dummy data
2. Inspect loss trends (should decrease steadily)
3. Check queue pointer wrapping if using hybrid mode
4. Verify augmentations visually (save crops and inspect)
5. Ensure momentum update is running (check `m` parameter)

### Loading Pretrained Weights for U-Net
```python
checkpoint = torch.load('checkpoints/moco_pv03.pth')
encoder_weights = checkpoint['encoder_state_dict']
unet.encoder.load_state_dict(encoder_weights)  # Freeze or fine-tune as needed
```

## Important Constraints

### What NOT to Include
- ❌ No U-Net training in this codebase (SSL pretraining only)
- ❌ No segmentation loss during MoCo training
- ❌ No automatic dataset downloading
- ❌ No hardcoded dataset paths (always use Config)

### Code Quality Requirements
- Type hints for function signatures
- Docstrings at file and class level (explain "why", not "what")
- PEP8 compliance
- Modular design: no 500+ line files
- Separation of concerns: datasets ≠ models ≠ training ≠ utils

## Dataset Details

**PV03:** RGB satellite imagery of solar panel installations
- Images: 224x224 (resized from various sources)
- Format: JPEG or PNG
- Labels: Binary masks (used only for evaluation, not SSL)

The SSL pipeline treats this as an unlabeled dataset.
