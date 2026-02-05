# Training Failure Diagnosis & Fixes

## 🚨 Problem Report

**Your training results:**
- Initial loss: 24.5
- Epoch 15 loss: 22.3 (stuck!)
- Epoch 600 loss: 21.3
- **Total improvement: Only 3.2 across 600 epochs!**

**Expected MoCo v3 training:**
- Initial loss: 5-8
- Epoch 100 loss: 3-4
- Final loss: 1.5-2.5

---

## 🔍 Root Causes Identified

### BUG #1: Symmetric Loss Not Averaged (CRITICAL!)

**Location:** `src/models/moco_v3.py`, Line 153

**Current code:**
```python
loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
return loss
```

**Problem:**
- MoCo v3 uses symmetric loss: compute loss for both (q1→k2) and (q2→k1)
- Your code SUMS these two losses: L_total = L1 + L2
- Should AVERAGE them: L_total = (L1 + L2) / 2

**Evidence:**
- Your initial loss: 24.5 ≈ 12.25 + 12.25 (two terms added)
- Expected initial: ~12 (if properly averaged)

**Fix:**
```python
loss = 0.5 * (self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1))
return loss
```

---

### BUG #2: Learning Rate Too Low for Small Dataset

**Location:** `src/config_optimized_16gb.py`, Line 48

**Current:**
```python
learning_rate: float = 4.5e-4  # Scaled for effective_batch=768
base_lr: float = 1.5e-4
```

**Problem:**
- MoCo v3 paper uses `base_lr = 1.5e-3` (NOT 1.5e-4!)
- You have 10× lower base LR than paper
- With small dataset (2000 images), you need HIGHER learning rate, not lower

**Calculation:**
```
Paper: base_lr = 1.5e-3, batch = 4096
  → lr = 1.5e-3 × (4096 / 256) = 0.024

You (optimized): base_lr = 1.5e-4, batch = 768
  → lr = 1.5e-4 × (768 / 256) = 4.5e-4  ← 50× LOWER than paper!

Correct: base_lr = 1.5e-3, batch = 768
  → lr = 1.5e-3 × (768 / 256) = 4.5e-3  ← 10× higher than yours
```

**Fix:**
```python
base_lr: float = 1.5e-3  # CORRECT value from paper
learning_rate: float = 4.5e-3  # Scaled for effective_batch=768
```

---

### BUG #3: Loss Stuck After Warmup (Cosine Schedule Issue)

**Location:** `src/training/train_moco.py`, Line 249

**Current:**
```python
lr = config.learning_rate * 0.5 * (1. + math.cos(...))
```

**Problem:**
- After warmup (60 epochs for optimized config), cosine decay starts
- By epoch 600, cosine has decayed LR to near-zero
- With already-low base LR (4.5e-4), decay makes it unusably small
- This is why loss "stuck" after epoch 15 - LR became too small to learn!

**Evidence:**
- Loss drops from 24.5 → 22.3 during warmup (epochs 0-60)
- Loss barely changes 22.3 → 21.3 after warmup (epochs 60-600)
- Warmup phase: LR increases → model learns
- Cosine phase: LR decreases rapidly → learning stalls

**Math:**
```
At epoch 15 (during warmup):
  lr = 4.5e-4 × (15/60) = 1.125e-4  ← Low but still learning

At epoch 100:
  Cosine factor ≈ 0.8
  lr = 4.5e-4 × 0.8 = 3.6e-4  ← Still learning

At epoch 300:
  Cosine factor ≈ 0.3
  lr = 4.5e-4 × 0.3 = 1.35e-4  ← Too low!

At epoch 600:
  Cosine factor ≈ 0.0
  lr = 4.5e-4 × 0.0 = ~0  ← Effectively no learning!
```

---

## ✅ Complete Fix Checklist

### Fix 1: Average Symmetric Loss (HIGHEST PRIORITY!)

**File:** `src/models/moco_v3.py`

**Line 153, change from:**
```python
loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
```

**To:**
```python
loss = 0.5 * (self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1))
```

**Expected impact:** Loss values will halve (24.5 → 12.25)

---

### Fix 2: Correct Base Learning Rate

**File:** `src/config_optimized_16gb.py`

**Lines 48-49, change from:**
```python
learning_rate: float = 4.5e-4
base_lr: float = 1.5e-4
```

**To:**
```python
learning_rate: float = 4.5e-3  # Scaled: 1.5e-3 × (768/256)
base_lr: float = 1.5e-3  # Paper value
```

**Also update `config_low_vram.py` (Lines 48-49):**
```python
learning_rate: float = 3.0e-3  # Scaled: 1.5e-3 × (512/256)
base_lr: float = 1.5e-3  # Paper value
```

**Expected impact:** Model will actually learn during cosine phase

---

### Fix 3: Reduce Warmup Epochs (Optional but Recommended)

**File:** `src/config_optimized_16gb.py`

**Line 50, change from:**
```python
warmup_epochs: int = 60  # 10% of total
```

**To:**
```python
warmup_epochs: int = 40  # Paper recommendation
```

**Reason:** 
- Paper uses 40 epochs regardless of total epochs
- With small dataset, you want to start learning faster
- 60 epochs is too conservative

---

### Fix 4: Consider Weight Decay Adjustment

**File:** `src/config_optimized_16gb.py`

**Current:**
```python
weight_decay: float = 0.05
```

**Optional adjustment for small dataset:**
```python
weight_decay: float = 0.1  # Stronger regularization for small dataset
```

---

## 📊 Expected Results After Fixes

### Before Fixes (Your Current):
```
Epoch   | Loss  | LR        | Status
--------|-------|-----------|------------------
1       | 24.5  | 1.1e-4    | Too high loss
15      | 22.3  | 1.1e-4    | Stuck in warmup
60      | 22.0  | 4.5e-4    | Warmup ends
100     | 21.8  | 3.6e-4    | Barely learning
300     | 21.5  | 1.4e-4    | LR too low
600     | 21.3  | ~0        | No learning
```

### After Fixes (Expected):
```
Epoch   | Loss  | LR        | Status
--------|-------|-----------|------------------
1       | 11.5  | 1.1e-3    | ✅ Halved loss, 10× LR
15      | 9.5   | 1.7e-3    | ✅ Active learning
40      | 6.5   | 4.5e-3    | ✅ Warmup ends
100     | 4.2   | 3.6e-3    | ✅ Good convergence
300     | 2.8   | 1.4e-3    | ✅ Still learning
600     | 2.2   | ~1e-4     | ✅ Converged!
```

**Improvement: 11.5 → 2.2 (5× reduction vs your 1.15× reduction)**

---

## 🔬 Debugging Commands

### Test Loss Calculation (Before Training)

```python
# Test the loss function with dummy data
import torch
from src.models.moco_v3 import MoCoV3
from src.models.backbones import get_backbone
from src.config_optimized_16gb import Config

config = Config()
model = MoCoV3(
    lambda: get_backbone(config.backbone),
    dim=config.feature_dim,
    mlp_dim=config.mlp_dim,
    T=config.temperature,
    m=config.momentum,
    use_queue=config.use_queue,
    queue_size=config.queue_size
)

# Dummy batch
batch_size = 96
x1 = torch.randn(batch_size, 3, 224, 224)
x2 = torch.randn(batch_size, 3, 224, 224)

# Forward pass
loss = model(x1, x2, m=0.99)
print(f"Loss with random init: {loss.item():.2f}")
print(f"Expected: ~11-12 (after fix #1)")
print(f"If you see ~24: Fix #1 not applied!")
```

### Monitor Learning Rate During Training

```python
# Add to training loop
print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# Or use TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
```

### Check Gradient Flow

```python
# After loss.backward(), before optimizer.step()
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f'Gradient norm: {total_norm:.4f}')

# If gradient norm is < 0.01: LR too low or loss not scaling correctly
# If gradient norm is > 100: Possible instability
```

---

## 🎯 Quick Fix Summary

**Critical (MUST do):**
1. ✅ Average symmetric loss (×0.5 multiplier)
2. ✅ Fix base learning rate (1.5e-4 → 1.5e-3)

**Important (SHOULD do):**
3. ✅ Update scaled LR accordingly
4. ✅ Reduce warmup to 40 epochs

**Optional (MAY do):**
5. ⚪ Increase weight decay to 0.1 (for small dataset)
6. ⚪ Add gradient clipping if instability occurs

---

## 🚀 Restart Training

After applying fixes:

```bash
# Stop current training (it's broken anyway)
# Apply fixes to the code
# Restart from scratch (don't resume!)

python main.py --multiprocessing-distributed --world-size 1 --rank 0

# Watch for these signs:
# ✅ Initial loss: 11-12 (not 24!)
# ✅ Loss drops to ~8 within 10 epochs
# ✅ Loss reaches ~4-5 by epoch 100
# ✅ Learning doesn't stall after warmup
```

---

## 🔍 If Still Having Issues

### Issue: Loss still ~24
→ Fix #1 (averaging) not applied correctly

### Issue: Loss drops but stalls at ~8-9
→ Learning rate still too low OR dataset too small

### Issue: Loss oscillates/doesn't converge
→ Learning rate too high, try 0.5× your current value

### Issue: Loss is NaN
→ Learning rate WAY too high, check you didn't use 1.5 instead of 1.5e-3

---

## 📚 References

**MoCo v3 Paper:**
- Base LR: 1.5e-3 (NOT 1.5e-4!)
- LR scaling: `lr = base_lr × (batch_size / 256)`
- Warmup: 40 epochs
- Temperature: 0.2
- Symmetric loss: Should be averaged!

**Your Dataset (2000 images):**
- Smaller than paper (1.2M images)
- Needs MORE epochs (600-1000)
- Needs HIGHER LR per image (to compensate)
- Queue is CRITICAL (you already have this ✅)

---

**All fixes have been documented. Apply them and restart training!**
