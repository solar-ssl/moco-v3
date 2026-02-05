"""
LOW VRAM Configuration for MoCo v3 pretraining (16GB total: 8+8GB dual GPU).

This configuration is optimized for limited VRAM scenarios:
- Uses ResNet-50 instead of ViT-Base (10GB → 4.5GB model VRAM)
- Enables queue for hybrid MoCo v2/v3 (compensates for small batches)
- Uses gradient accumulation to simulate larger batches
- Requires mixed precision (AMP) and gradient checkpointing

VRAM Breakdown (per GPU):
- ResNet-50 model (query + momentum): ~2.5GB
- Batch (64 images): ~3GB
- Gradients + optimizer states: ~1.5GB
- Buffer: ~1GB
Total: ~8GB per GPU ✓ Fits in 8GB!

Effective batch size with accumulation: 64 * 2 GPUs * 4 steps = 512
"""

from dataclasses import dataclass

@dataclass
class Config:
    # Dataset settings
    dataset_path: str = "dataset"
    image_size: int = 224
    
    # Model settings - LOW VRAM OPTIMIZED
    # ResNet-50 uses ~2.5GB vs ViT-Base ~5GB per encoder
    backbone: str = "resnet50"  # Do NOT use ViT-Base with 16GB VRAM
    feature_dim: int = 256
    mlp_dim: int = 4096
    
    # Queue settings - CRITICAL for small batches
    # Queue provides 65K negatives when batch is small
    # MoCo v3 paper disables queue for batch=4096, but we NEED it for batch<1024
    use_queue: bool = True  # Required: compensates for small batch
    queue_size: int = 65536
    
    # Training settings - 16GB VRAM (8+8GB)
    # Per-GPU batch: 64 → Total effective: 128 across 2 GPUs
    batch_size: int = 64  # Per-GPU for DDP
    epochs: int = 300  # Standard for ResNet-50 (could reduce to 200)
    
    # Learning rate scaling: base_lr * (effective_batch / 256)
    # Effective batch with accumulation: 64 * 2 GPUs * 4 steps = 512
    # lr = 1.5e-4 * (512 / 256) = 3.0e-4
    learning_rate: float = 3.0e-4  # Scaled for effective_batch=512
    base_lr: float = 1.5e-4  # Reference LR at batch=256
    warmup_epochs: int = 40
    momentum: float = 0.99
    temperature: float = 0.2
    weight_decay: float = 0.05
    
    # Optimizer settings
    optimizer: str = "adamw"
    
    # Hardware settings - CRITICAL VRAM OPTIMIZATIONS
    num_workers: int = 8
    seed: int = 42
    
    # CRITICAL: These MUST be enabled for 16GB VRAM
    use_amp: bool = True  # Mixed precision saves ~40% VRAM (REQUIRED)
    use_grad_checkpointing: bool = True  # Saves ~30% VRAM (REQUIRED)
    
    # Gradient accumulation: simulate larger batch without VRAM cost
    # accumulation=4 → effective_batch = 64 * 2 * 4 = 512
    gradient_accumulation_steps: int = 4  # Simulates batch=512
    
    # Checkpoint settings
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "moco_pv03_low_vram.pth"
    save_freq: int = 10

    def get_experiment_tag(self) -> str:
        """Returns experiment tag with VRAM profile indicator."""
        bb = "r50" if "resnet50" in self.backbone else self.backbone.replace("vit_", "v")
        eff_batch = self.batch_size * 2 * self.gradient_accumulation_steps  # 2 GPUs assumed
        return f"{bb}_bs{eff_batch}_lr{self.learning_rate}_ep{self.epochs}_lowvram"
    
    def print_vram_estimate(self):
        """Print estimated VRAM usage."""
        print("\n" + "="*70)
        print("VRAM ESTIMATE (per GPU with AMP + grad checkpointing):")
        print("="*70)
        print(f"  Model (ResNet-50 query+momentum): ~2.5 GB")
        print(f"  Batch ({self.batch_size} images):     ~3.0 GB")
        print(f"  Gradients + optimizer states:     ~1.5 GB")
        print(f"  Buffer + overhead:                ~1.0 GB")
        print(f"  " + "-"*66)
        print(f"  TOTAL per GPU:                    ~8.0 GB ✓")
        print(f"  " + "-"*66)
        print(f"  Your hardware: 2x 8GB GPUs        16.0 GB ✓ FITS!")
        print(f"\n  Effective batch size: {self.batch_size} * 2 GPUs * {self.gradient_accumulation_steps} accum = {self.batch_size * 2 * self.gradient_accumulation_steps}")
        print("="*70 + "\n")
