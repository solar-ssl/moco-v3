# Training Output & Results Location Guide

## 📁 Where Are Results Saved?

### Directory Structure

```
moco-v3/
├── checkpoints/                    # Root checkpoint directory
│   ├── r50_bs768_lr0.0045_ep600_optimized/   # Experiment subdirectory
│   │   ├── last.pth               # Most recent epoch
│   │   └── best.pth               # Best model (lowest loss)
│   ├── r50_bs512_lr0.003_ep300_lowvram/      # Different config
│   │   ├── last.pth
│   │   └── best.pth
│   └── vbase_bs32_lr0.00015_ep100_hybrid/    # Another config
│       ├── last.pth
│       └── best.pth
└── logs/ (if using TensorBoard - optional)
    └── <experiment_tag>/
        └── events.out.tfevents.*
```

---

## 🏷️ Experiment Tags (Automatic Naming)

Each training run gets a **unique experiment tag** based on its configuration:

### Format:
```
{backbone}_{batch}_{lr}_{epochs}_{config_type}
```

### Examples:

**config_optimized_16gb.py:**
```
r50_bs768_lr0.0045_ep600_optimized
```
- `r50`: ResNet-50 backbone
- `bs768`: Effective batch size = 768
- `lr0.0045`: Learning rate = 0.0045
- `ep600`: Total epochs = 600
- `optimized`: Config type

**config_low_vram.py:**
```
r50_bs512_lr0.003_ep300_lowvram
```

**config.py (original):**
```
vbase_bs32_lr0.00015_ep100_hybrid
```
- `vbase`: ViT-Base backbone
- `bs32`: Batch size = 32
- (Note: This has the OLD incorrect learning rate!)

---

## 📊 What's Inside Each Checkpoint?

### `last.pth` (Saved Every Epoch)

```python
{
    'epoch': 150,                    # Current epoch number
    'arch': 'resnet50',              # Backbone architecture
    'state_dict': model.state_dict(), # Full MoCo v3 model weights
    'optimizer': optimizer.state_dict(), # Optimizer state (for resuming)
    'loss': 4.23                     # Average loss for this epoch
}
```

**File size:** ~200-500 MB (depends on backbone)

### `best.pth` (Saved When Loss Improves)

Same structure as `last.pth`, but only saved when:
```python
current_loss < previous_best_loss
```

This is your **best performing model** based on training loss.

---

## 🔍 Finding Your Results

### Method 1: List All Experiment Directories

```bash
ls -lh checkpoints/
```

Output:
```
drwxr-xr-x 2 user user 4.0K Feb  5 21:00 r50_bs768_lr0.0045_ep600_optimized
```

### Method 2: Find Specific Experiment

```bash
# Find your current config's experiment
python3 << EOF
from src.config_optimized_16gb import Config
c = Config()
print(f"Your checkpoints: checkpoints/{c.get_experiment_tag()}/")
EOF
```

Output:
```
Your checkpoints: checkpoints/r50_bs768_lr0.0045_ep600_optimized/
```

### Method 3: Check Most Recent Training

```bash
# Find most recently modified checkpoint
find checkpoints/ -name "last.pth" -type f -exec ls -lth {} + | head -5
```

---

## 📈 Monitoring Training Progress

### Check Latest Checkpoint

```bash
python3 << EOF
import torch

# Load latest checkpoint
checkpoint_path = "checkpoints/r50_bs768_lr0.0045_ep600_optimized/last.pth"
ckpt = torch.load(checkpoint_path, map_location='cpu')

print(f"Epoch: {ckpt['epoch']}")
print(f"Loss: {ckpt['loss']:.4f}")
print(f"Backbone: {ckpt['arch']}")
EOF
```

### Compare Last vs Best

```bash
python3 << EOF
import torch

exp_dir = "checkpoints/r50_bs768_lr0.0045_ep600_optimized"

last = torch.load(f"{exp_dir}/last.pth", map_location='cpu')
best = torch.load(f"{exp_dir}/best.pth", map_location='cpu')

print(f"Last checkpoint:")
print(f"  Epoch: {last['epoch']}, Loss: {last['loss']:.4f}")
print(f"\nBest checkpoint:")
print(f"  Epoch: {best['epoch']}, Loss: {best['loss']:.4f}")
print(f"\nImprovement: {(last['loss'] - best['loss']):.4f}")
EOF
```

---

## 🔄 Resuming Training

### Load Last Checkpoint to Continue Training

```python
from src.utils.checkpoints import load_checkpoint

# In your training script
checkpoint_path = "checkpoints/r50_bs768_lr0.0045_ep600_optimized/last.pth"
start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

# Continue from start_epoch
for epoch in range(start_epoch, config.epochs):
    train(...)
```

### Load Best Checkpoint for Evaluation

```python
import torch

checkpoint_path = "checkpoints/r50_bs768_lr0.0045_ep600_optimized/best.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Load just the encoder (for U-Net downstream task)
encoder_state_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                     if k.startswith('base_model')}
unet_encoder.load_state_dict(encoder_state_dict, strict=False)
```

---

## 🗂️ Multiple Training Runs

### Scenario: Testing Different Configs

If you train with different configs, you get **separate experiment directories**:

```
checkpoints/
├── r50_bs512_lr0.003_ep300_lowvram/        # First run
│   ├── last.pth  (Epoch 300, Loss: 3.2)
│   └── best.pth  (Epoch 245, Loss: 3.1)
├── r50_bs768_lr0.0045_ep600_optimized/     # Second run (after fixes)
│   ├── last.pth  (Epoch 600, Loss: 2.2)
│   └── best.pth  (Epoch 580, Loss: 2.1)
└── r50_bs768_lr0.00045_ep600_optimized/    # Third run (old LR bug)
    ├── last.pth  (Epoch 600, Loss: 21.3)   # ← Broken training!
    └── best.pth  (Epoch 15, Loss: 22.0)
```

**Each run is isolated** - no overwriting!

---

## 🧹 Cleaning Up Failed Runs

### Delete Broken Training (Before Fix)

```bash
# Remove the broken run with incorrect LR
rm -rf "checkpoints/r50_bs768_lr0.00045_ep600_optimized"

# Or keep for comparison (rename it)
mv "checkpoints/r50_bs768_lr0.0045_ep600_optimized" \
   "checkpoints/r50_bs768_lr0.0045_ep600_optimized_BROKEN"
```

### Delete All Checkpoints (Fresh Start)

```bash
rm -rf checkpoints/*
```

---

## 📊 Checkpoint Contents Detail

### What Can You Extract?

```python
import torch

checkpoint = torch.load("checkpoints/<experiment>/best.pth")

# Available keys
print(checkpoint.keys())
# Output: ['epoch', 'arch', 'state_dict', 'optimizer', 'loss']

# 1. Get training metadata
epoch = checkpoint['epoch']          # 580
loss = checkpoint['loss']            # 2.14
arch = checkpoint['arch']            # 'resnet50'

# 2. Extract full model
model.load_state_dict(checkpoint['state_dict'])

# 3. Extract just encoder (for downstream tasks)
encoder_dict = {k.replace('base_model.', ''): v 
                for k, v in checkpoint['state_dict'].items() 
                if 'base_model' in k and 'base_model_k' not in k}

# 4. Resume training
optimizer.load_state_dict(checkpoint['optimizer'])
```

---

## 🎯 Quick Reference

### Current Training Session

**On your GPU machine, check:**
```bash
# What experiment is running?
python3 -c "from src.config_optimized_16gb import Config; print(Config().get_experiment_tag())"

# Where are results?
ls -lh checkpoints/r50_bs768_lr0.0045_ep600_optimized/

# Latest progress?
python3 -c "import torch; c=torch.load('checkpoints/r50_bs768_lr0.0045_ep600_optimized/last.pth'); print(f'Epoch {c[\"epoch\"]}, Loss {c[\"loss\"]:.4f}')"
```

### Expected Paths (After Restarting Training)

**With config_optimized_16gb.py:**
```
checkpoints/r50_bs768_lr0.0045_ep600_optimized/
  ├── last.pth    (updated every epoch)
  └── best.pth    (updated when loss improves)
```

**File sizes:**
- ResNet-50 checkpoint: ~200 MB
- ViT-Base checkpoint: ~350 MB

---

## 🚨 Important Notes

### 1. Checkpoints Are Local

Checkpoints are saved **on the machine running training**:
- If training on GPU PC → checkpoints are THERE
- Not synced to this development machine automatically

### 2. Experiment Tag Changes

If you modify the config, the experiment tag changes:
```python
# Change learning rate
learning_rate = 4.5e-3  →  5.0e-3

# New experiment tag
r50_bs768_lr0.0045_ep600_optimized  →  r50_bs768_lr0.005_ep600_optimized
#                ^^^^                                ^^^^ (changed)
```

This creates a **NEW directory** - previous run is preserved.

### 3. Best vs Last

- `last.pth`: Always the most recent epoch
- `best.pth`: Best performing model (might be from epoch 245, even if you're at epoch 600)

**For downstream tasks:** Use `best.pth` (better representations)

### 4. Disk Space

With 600 epochs:
- `last.pth` overwrites itself: ~200 MB
- `best.pth` updates occasionally: ~200 MB
- Total: ~400 MB per training run

If disk space is limited, you can delete older experiment directories.

---

## 🔧 Customizing Save Behavior

### Change Save Frequency

**In config file:**
```python
save_freq: int = 20  # Save every 20 epochs (default for optimized)
```

**To save every epoch:**
```python
save_freq: int = 1
```

**To save every 50 epochs:**
```python
save_freq: int = 50
```

Note: `last.pth` is saved **every epoch** regardless. `save_freq` doesn't affect this currently (could be added).

### Change Checkpoint Directory

**In config file:**
```python
checkpoint_dir: str = "/data/moco_checkpoints"  # Custom path
```

Or via environment variable (add to train_moco.py):
```python
config.checkpoint_dir = os.getenv('CHECKPOINT_DIR', config.checkpoint_dir)
```

Then:
```bash
CHECKPOINT_DIR=/mnt/ssd/checkpoints python main.py ...
```

---

## 📋 Summary

| Item | Location | Frequency | Size |
|------|----------|-----------|------|
| Checkpoints | `checkpoints/<experiment_tag>/` | Every epoch | ~400MB |
| Latest model | `last.pth` | Every epoch | ~200MB |
| Best model | `best.pth` | When improved | ~200MB |
| Experiment tag | Auto-generated from config | Per run | - |
| Training logs | Terminal output (can redirect) | Real-time | - |

**Quick check on GPU machine:**
```bash
ls -lh checkpoints/r50_bs768_lr0.0045_ep600_optimized/
```

**Your results are there!** 🎉
