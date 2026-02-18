"""
Main training script for MoCo v3 pretraining.
Supports multi-GPU via DistributedDataParallel (DDP).
"""

import argparse
import os
import random
import math
import numpy as np

import torch
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from src.config import Config
from src.datasets.pv03_ssl import PV03SSLDataset
from src.models.moco_v3 import MoCoV3
from src.models.backbones import get_backbone
from src.utils.augmentations import get_moco_v3_augmentations
from tqdm.auto import tqdm
from src.utils.logging import AverageMeter
from src.utils.checkpoints import load_checkpoint, save_checkpoint

def _worker_init_fn(worker_id: int) -> None:
    """
    Seed each DataLoader worker's Python/NumPy state deterministically.

    PyTorch automatically sets torch.initial_seed() per-worker based on the
    main process seed + worker_id, so we only need to propagate that value
    to the other RNG libraries used inside augmentations.
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def _restore_rng_state(rng: "dict | None") -> None:
    """
    Restores Python / NumPy / PyTorch (CPU + CUDA) RNG states from the
    rng_state dict returned by load_checkpoint(). Avoids re-reading the
    checkpoint file from disk a second time.

    Args:
        rng: The 'rng_state' dict from load_checkpoint(), or None for
             older checkpoints that did not save RNG state.
    """
    if rng is None:
        return   # older checkpoint without RNG state — skip silently
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if rng.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])
    print("=> restored RNG state from checkpoint")


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MoCo v3 Training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint to resume training from (default: none)')
    parser.add_argument('--warmup-epochs', default=40, type=int,
                        help='number of LR warmup epochs (default: 40)')
    return parser.parse_args()

def main():
    args = get_args()
    config = Config()

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
    else:
        # Simply call main_worker function
        main_worker(0, ngpus_per_node, args, config)

def main_worker(gpu, ngpus_per_node, args, config):
    args.gpu = gpu

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # ------------------------------------------------------------------ #
    # Per-process seeding                                                  #
    # Must happen AFTER args.rank is finalised so each GPU process gets a #
    # unique but deterministic seed, producing different augmentations.   #
    # ------------------------------------------------------------------ #
    if config.seed is not None:
        seed = config.seed + args.rank  # unique per GPU: rank 0→seed, 1→seed+1 …
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print(f"[rank {args.rank}] seeded with {seed}")

    # Validate that warmup is strictly shorter than total training to avoid
    # a ZeroDivisionError in the cosine LR decay phase.
    if args.warmup_epochs >= config.epochs:
        raise ValueError(
            f"--warmup-epochs ({args.warmup_epochs}) must be strictly less than "
            f"config.epochs ({config.epochs}). The cosine decay phase would have "
            f"zero length, causing a ZeroDivisionError in adjust_learning_rate()."
        )

    # Create model
    print(f"=> creating model with {config.backbone} backbone")
    def backbone_fn():
        return get_backbone(config.backbone)

    model = MoCoV3(
        backbone_fn,
        dim=config.feature_dim,
        mlp_dim=config.mlp_dim,
        T=config.temperature,
        m=config.momentum
    )

    start_epoch = 0
    best_loss   = float('inf')

    if args.multiprocessing_distributed:
        # Each spawned process has args.gpu set to its process index by mp.spawn,
        # so device_ids is always a single concrete GPU — never None.
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        config.batch_size = int(config.batch_size / ngpus_per_node)
        config.num_workers = int((config.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    else:
        # Single-GPU: args.gpu is always 0 (main_worker is called as main_worker(0, ...))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # Build the optimizer before DDP wrapping. DDP does not replace parameter
    # tensors — it only adds gradient-sync hooks — so the optimizer's parameter
    # references remain valid after wrapping.
    optimizer = build_optimizer(model, config)

    # Single checkpoint load: reads the file once and restores model weights,
    # optimizer state, and RNG state together. Must happen before DDP wrapping
    # so load_checkpoint can strip the 'module.' prefix if needed.
    if args.resume:
        start_epoch, best_loss, rng_state = load_checkpoint(model, optimizer, args.resume)
        _restore_rng_state(rng_state)

    # Wrap with DDP after weights and optimizer state are loaded
    if args.multiprocessing_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # cuDNN: deterministic and benchmark are mutually exclusive.
    #   deterministic=True  → fixed algorithm selection, reproducible ops (slower).
    #   benchmark=True      → fastest algorithm search, non-deterministic (faster).
    # config.deterministic drives both settings so they can never conflict.
    cudnn.deterministic = config.deterministic
    cudnn.benchmark     = not config.deterministic

    # Data loading code
    traindir = config.dataset_path
    
    train_dataset = PV03SSLDataset(
        traindir,
        get_moco_v3_augmentations(config.image_size),
    )

    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # Seeded generator drives the DataLoader's internal shuffle so data
    # ordering is reproducible across runs on the same machine.
    loader_generator = None
    if config.seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(config.seed + args.rank)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        # Seed each worker's Python/NumPy RNG deterministically
        worker_init_fn=_worker_init_fn if config.seed is not None else None,
        # Reproducible shuffle order
        generator=loader_generator,
        # Keep workers alive between epochs so their seeded state is preserved
        persistent_workers=config.num_workers > 0,
    )

    for epoch in range(start_epoch, config.epochs):
        if args.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)

        # Train for one epoch; returns (avg_loss, final_step_lr)
        avg_loss, cur_lr = train(train_loader, model, optimizer, epoch, args, config)

        # Synchronise loss across all GPUs so every rank agrees on is_best.
        # Without this each rank uses its own shard's loss, which can differ,
        # causing rank-0 to write best.pth at the wrong epochs.
        if args.multiprocessing_distributed:
            avg_loss_t = torch.tensor(avg_loss, device=f'cuda:{args.gpu}')
            dist.all_reduce(avg_loss_t, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_t.item() / args.world_size

        is_best   = avg_loss < best_loss
        best_loss = min(avg_loss, best_loss)

        # Only the rank-0 process writes checkpoints to avoid file conflicts
        is_rank0 = (not args.multiprocessing_distributed
                    or args.rank % ngpus_per_node == 0)

        if is_rank0:
            # Strip 'module.' prefix added by DDP so the checkpoint can be
            # loaded back into a plain (non-DDP) model without key errors.
            raw_state = (model.module.state_dict()
                         if args.multiprocessing_distributed
                         else model.state_dict())
            checkpoint_state = {
                'epoch':      epoch + 1,
                'arch':       config.backbone,
                'state_dict': raw_state,
                'optimizer':  optimizer.state_dict(),
                'best_loss':  best_loss,
                # RNG states — restored by _restore_rng_state() on resume
                'rng_state': {
                    'python': random.getstate(),
                    'numpy':  np.random.get_state(),
                    'torch':  torch.get_rng_state(),
                    'cuda':   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            }
            # Periodic snapshot every save_freq epochs for recovery
            if (epoch + 1) % config.save_freq == 0:
                save_checkpoint(checkpoint_state,
                                filename='last.pth',
                                checkpoint_dir=config.checkpoint_dir)
            # Best checkpoint: written immediately whenever loss improves,
            # independent of save_freq so it is never silently missed.
            if is_best:
                save_checkpoint(checkpoint_state,
                                filename='best.pth',
                                checkpoint_dir=config.checkpoint_dir)
                print(f"=> saved new best checkpoint "
                      f"(epoch {epoch + 1}, best_loss {best_loss:.6f})")
            # Per-epoch summary printed after the tqdm bar closes
            print(
                f"\n[Epoch {epoch + 1}/{config.epochs}]  "
                f"avg_loss={avg_loss:.6f}  best_loss={best_loss:.6f}  "
                f"lr={cur_lr:.2e}"
            )

def build_optimizer(model, config):
    """
    Builds and returns the optimizer specified by config.optimizer.

    Supported values:
        "adamw" — AdamW with configurable betas and weight_decay.
                  Recommended for all backbones; required for ViT.
        "sgd"   — SGD with momentum. Classic choice for ResNet.
    """
    opt = config.optimizer.lower()
    if opt == "adamw":
        print(f"=> using AdamW optimizer  "
              f"(lr={config.learning_rate}, betas={config.adamw_betas}, "
              f"wd={config.weight_decay})")
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.adamw_betas,
            weight_decay=config.weight_decay,
        )
    elif opt == "sgd":
        print(f"=> using SGD optimizer  "
              f"(lr={config.learning_rate}, momentum={config.sgd_momentum}, "
              f"wd={config.weight_decay})")
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer '{config.optimizer}'. "
            f"Supported options: 'adamw', 'sgd'."
        )

def train(train_loader, model, optimizer, epoch, args, config):
    losses     = AverageMeter('Loss', ':.4e')
    grad_norms = AverageMeter('GradNorm', ':.4e')
    cur_lr     = 0.0

    # Switch to train mode (MoCoV3.train() keeps the key encoder in eval)
    model.train()

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch [{epoch + 1}/{config.epochs}]",
        leave=True,
    )

    for i, images in pbar:
        # Adjust learning rate and momentum for this step
        cur_lr = adjust_learning_rate(
            optimizer, epoch, i, len(train_loader), config, args.warmup_epochs
        )
        cur_m = adjust_moco_momentum(epoch, i, len(train_loader), config)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # Compute output and loss
        loss = model(images[0], images[1], cur_m)
        losses.update(loss.item(), images[0].size(0))

        # Backward + optional gradient clipping + optimizer step
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients before the optimizer step.
        # This is required for ViT backbones (MoCo v3 paper uses max_norm=1.0).
        # config.clip_grad_norm == 0.0 disables clipping entirely.
        if config.clip_grad_norm > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.clip_grad_norm
            )
        else:
            # Compute the norm for logging even when clipping is disabled
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float('inf')
            )
        grad_norms.update(grad_norm.item(), images[0].size(0))

        optimizer.step()

        # Update the tqdm bar with live per-step stats.
        # 'lr' shows the exact current learning rate (not an average).
        pbar.set_postfix(
            loss=f"{losses.val:.4e}",
            avg=f"{losses.avg:.4e}",
            gnorm=f"{grad_norms.val:.4e}",
            lr=f"{cur_lr:.2e}",
            m=f"{cur_m:.4f}",
        )

    return losses.avg, cur_lr

def adjust_learning_rate(optimizer, epoch, i, iter_per_epoch, config, warmup_epochs=40):
    """Decay the learning rate with half-cycle cosine after warmup"""
    T = epoch * iter_per_epoch + i
    warmup_iters = warmup_epochs * iter_per_epoch
    T_max = config.epochs * iter_per_epoch
    
    if T < warmup_iters:
        lr = config.learning_rate * T / warmup_iters
    else:
        lr = config.learning_rate * 0.5 * (1. + math.cos(math.pi * (T - warmup_iters) / (T_max - warmup_iters)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_moco_momentum(epoch, i, iter_per_epoch, config):
    """Cosine schedule for MoCo momentum"""
    T = epoch * iter_per_epoch + i
    T_max = config.epochs * iter_per_epoch
    m = 1. - 0.5 * (1. - config.momentum) * (1. + math.cos(math.pi * T / T_max))
    return m

if __name__ == '__main__':
    main()
