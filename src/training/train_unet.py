"""
Training script for U-Net fine-tuning on PV03 binary segmentation.

Loss function
─────────────
  Total loss = BCE (binary cross-entropy) + Dice loss
  BCE  catches per-pixel imbalance well with logit scaling.
  Dice directly optimises the overlap metric and compensates for class
  imbalance (solar panels are a small fraction of each image).

Metrics (computed on thresholded binary predictions at 0.5)
────────────────────────────────────────────────────────────
  IoU  = TP / (TP + FP + FN)   — Intersection over Union
  F1   = 2·TP / (2·TP + FP + FN)  — Dice / F1 score

LR schedule
───────────
  Cosine decay with linear warmup.
  Warmup: first ``config.warmup_epochs`` epochs ramp from 0 → base_lr.
  After warmup: cosine decay from base_lr → base_lr × min_lr_factor.

Usage
─────
  python -m src.training.train_unet \\
      --backbone  resnet50 \\
      --checkpoint checkpoints/resnet50/best.pth \\
      --exp-name  exp_a_moco_r50

  python -m src.training.train_unet \\
      --backbone  vit_small \\
      --checkpoint checkpoints/vit_small/best.pth \\
      --exp-name  exp_e_moco_vit_small

  # Random initialisation (no MoCo pretraining):
  python -m src.training.train_unet \\
      --backbone  vit_small \\
      --exp-name  exp_f_moco_vit_small_random_initialization
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from src.config import FinetuneConfig
from src.datasets.pv03_finetune import PV03FinetuneDataset, get_finetune_transforms
from src.models.unet import UNet


# ─── Loss functions ───────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Expects raw logits (no sigmoid applied yet).  Smooth=1 prevents division
    by zero when the target or prediction is all zeros.
    """
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)                    # [B, 1, H, W]
        probs   = probs.view(probs.shape[0], -1)         # [B, N]
        targets = targets.view(targets.shape[0], -1)     # [B, N]

        intersection = (probs * targets).sum(dim=1)
        union        = probs.sum(dim=1) + targets.sum(dim=1)
        dice         = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Weighted BCE + Dice loss.

    pos_weight:  Multiplier on the positive (foreground) class in BCE.
                 Solar panels occupy a small fraction of each patch, so the
                 background gradient overwhelms the foreground without this.
                 Rule of thumb: pos_weight ≈ (background pixels) / (foreground
                 pixels).  Default 5.0 is conservative; increase if IoU stays
                 low.  Registered as a buffer so .to(device) moves it.

    dice_weight: Multiplier on the Dice term.  Dice directly optimises the
                 overlap metric and is naturally robust to imbalance.  A value
                 of 2 gives Dice twice the influence of BCE.
    """
    def __init__(self, pos_weight: float = 5.0, dice_weight: float = 2.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.register_buffer('pos_weight_buf', torch.tensor([pos_weight]))
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight_buf
        )
        return bce + self.dice_weight * self.dice(logits, targets)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Compute IoU and F1 on a batch of binary predictions.

    Args:
        logits:    [B, 1, H, W]  raw model output (not sigmoided)
        targets:   [B, 1, H, W]  binary ground truth (0.0 or 1.0)
        threshold: decision boundary applied to sigmoid output

    Returns:
        dict with keys 'iou' and 'f1' (Python floats, averaged over batch)
    """
    preds   = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()

    # Flatten to [B, N]
    preds   = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    iou = (tp / (tp + fp + fn + 1e-7)).mean().item()
    f1  = (2 * tp / (2 * tp + fp + fn + 1e-7)).mean().item()

    return {'iou': iou, 'f1': f1}


# ─── LR schedule ──────────────────────────────────────────────────────────────

def adjust_lr(
    optimizer: optim.Optimizer,
    epoch: int,
    config: "FinetuneConfig",
) -> float:
    """
    Linear warmup for ``config.warmup_epochs`` epochs, then cosine decay to
    ``config.learning_rate * config.min_lr_factor``.

    Returns the learning rate set for this epoch.
    """
    warmup = config.warmup_epochs
    total  = config.epochs
    base   = config.learning_rate
    min_lr = base * config.min_lr_factor

    if epoch < warmup:
        lr = base * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(total - warmup, 1)
        lr = min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ─── One-epoch helpers ────────────────────────────────────────────────────────

def run_epoch(
    loader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    training: bool,
    desc: str,
) -> dict:
    """
    Run one full pass over the dataloader.

    Returns:
        dict with keys 'loss', 'iou', 'f1'
    """
    model.train(training)

    total_loss = 0.0
    total_iou  = 0.0
    total_f1   = 0.0
    n_batches  = 0

    with torch.set_grad_enabled(training):
        pbar = tqdm(loader, desc=desc, leave=False)
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, masks)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics = compute_metrics(logits.detach(), masks)
            total_loss += loss.item()
            total_iou  += metrics['iou']
            total_f1   += metrics['f1']
            n_batches  += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                iou=f"{metrics['iou']:.4f}",
                f1=f"{metrics['f1']:.4f}",
            )

    n = max(n_batches, 1)
    return {
        'loss': total_loss / n,
        'iou':  total_iou  / n,
        'f1':   total_f1   / n,
    }


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_unet_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_unet_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    path: str,
) -> tuple:
    """
    Returns (start_epoch, best_iou).
    """
    if not os.path.isfile(path):
        print(f"=> no U-Net checkpoint at '{path}', starting from scratch")
        return 0, 0.0

    print(f"=> loading U-Net checkpoint '{path}'")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    state = ckpt.get('state_dict', ckpt)
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    start_epoch = ckpt.get('epoch', 0)
    best_iou    = ckpt.get('best_iou', 0.0)
    print(f"=> resumed from epoch {start_epoch}, best_iou={best_iou:.4f}")
    return start_epoch, best_iou


# ─── CLI ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='U-Net fine-tuning on PV03')
    p.add_argument('--backbone',    default='resnet50',
                   choices=['resnet50', 'vit_small', 'vit_base'],
                   help='encoder backbone (default: resnet50)')
    p.add_argument('--checkpoint',  default='',
                   help='path to MoCo v3 pretrained checkpoint (empty = random init)')
    p.add_argument('--exp-name',    default='experiment',
                   help='experiment name (used as subdirectory in checkpoint_dir)')
    p.add_argument('--resume',      default='',
                   help='path to a U-Net checkpoint to resume from')
    p.add_argument('--freeze-encoder', action='store_true',
                   help='freeze encoder weights (linear probe mode)')
    p.add_argument('--epochs',      type=int, default=None,
                   help='override config.epochs')
    p.add_argument('--batch-size',  type=int, default=None,
                   help='override config.batch_size')
    p.add_argument('--lr',          type=float, default=None,
                   help='override config.learning_rate')
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    config = FinetuneConfig()

    # CLI overrides
    config.backbone    = args.backbone
    config.experiment_name = args.exp_name
    if args.epochs     is not None: config.epochs          = args.epochs
    if args.batch_size is not None: config.batch_size      = args.batch_size
    if args.lr         is not None: config.learning_rate   = args.lr

    # ── Reproducibility ──────────────────────────────────────────────────────
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"=> device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_tf, val_tf = get_finetune_transforms(image_size=config.image_size)

    full_ds = PV03FinetuneDataset(
        image_dir = config.train_image_dir,
        label_dir = config.train_label_dir,
        transform = train_tf,
    )
    train_ds, val_ds = full_ds.split(
        val_fraction = config.val_split,
        seed         = config.seed if config.seed is not None else 42,
    )
    val_ds.transform = val_tf
    print(f"=> dataset: {len(train_ds)} train / {len(val_ds)} val images")

    def _worker_init(worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed)
        random.seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        pin_memory  = True,
        drop_last   = True,
        worker_init_fn = _worker_init if config.seed is not None else None,
        persistent_workers = config.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size  = config.batch_size,
        shuffle     = False,
        num_workers = config.num_workers,
        pin_memory  = True,
        worker_init_fn = _worker_init if config.seed is not None else None,
        persistent_workers = config.num_workers > 0,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    moco_ckpt = args.checkpoint or ""
    print(f"=> backbone: {config.backbone}  |  MoCo checkpoint: '{moco_ckpt or 'none (random init)'}'")

    model = UNet(
        backbone_name   = config.backbone,
        num_classes     = config.num_classes,
        checkpoint_path = moco_ckpt if moco_ckpt else None,
        freeze_encoder  = args.freeze_encoder,
        image_size      = config.image_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"=> model: {n_params:.1f}M total params, {n_train:.1f}M trainable")

    # ── Optimiser & loss ─────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config.learning_rate,
        weight_decay = config.weight_decay,
    )
    criterion = CombinedLoss(
        pos_weight  = config.bce_pos_weight,
        dice_weight = config.dice_weight,
    ).to(device)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_iou    = 0.0
    ckpt_dir    = os.path.join(config.checkpoint_dir, config.experiment_name)

    if args.resume:
        start_epoch, best_iou = load_unet_checkpoint(model, optimizer, args.resume)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Experiment : {config.experiment_name}")
    print(f"  Epochs     : {start_epoch} → {config.epochs}")
    print(f"  Warmup     : {config.warmup_epochs} epochs")
    print(f"  Checkpoint : {ckpt_dir}/")
    print(f"{'─'*60}\n")

    for epoch in range(start_epoch, config.epochs):
        lr = adjust_lr(optimizer, epoch, config)

        # ── Train ────────────────────────────────────────────────────────────
        train_metrics = run_epoch(
            train_loader, model, criterion, optimizer, device,
            training = True,
            desc     = f"[Train] Epoch {epoch + 1}/{config.epochs}",
        )

        # ── Validate ─────────────────────────────────────────────────────────
        val_metrics = run_epoch(
            val_loader, model, criterion, optimizer, device,
            training = False,
            desc     = f"[Val]   Epoch {epoch + 1}/{config.epochs}",
        )

        is_best  = val_metrics['iou'] > best_iou
        best_iou = max(val_metrics['iou'], best_iou)

        # ── Per-epoch summary ─────────────────────────────────────────────────
        star = " ★" if is_best else ""
        print(
            f"[Epoch {epoch + 1:>3}/{config.epochs}]  "
            f"lr={lr:.2e}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"train_iou={train_metrics['iou']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_iou={val_metrics['iou']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}  "
            f"best_iou={best_iou:.4f}{star}"
        )

        # ── Checkpointing ─────────────────────────────────────────────────────
        state = {
            'epoch':      epoch + 1,
            'backbone':   config.backbone,
            'state_dict': model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'best_iou':   best_iou,
            'val_metrics': val_metrics,
        }

        if (epoch + 1) % config.save_freq == 0:
            save_unet_checkpoint(
                state,
                os.path.join(ckpt_dir, 'last_unet.pth'),
            )

        if is_best:
            save_unet_checkpoint(
                state,
                os.path.join(ckpt_dir, 'best_unet.pth'),
            )
            print(f"   => saved best_unet.pth  (val_iou={best_iou:.4f})")

    print(f"\n=> Training complete. Best val IoU: {best_iou:.4f}")
    print(f"=> Best model saved to: {os.path.join(ckpt_dir, 'best_unet.pth')}")


if __name__ == '__main__':
    main()
