"""
Main training script for MoCo v3 pretraining.
Supports multi-GPU via DistributedDataParallel (DDP).
"""

import argparse
import os
import random
import time
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from src.config_low_vram import Config
from src.datasets.pv03_ssl import PV03SSLDataset
from src.models.moco_v3 import MoCoV3
from src.models.backbones import get_backbone
from src.utils.augmentations import get_moco_v3_augmentations, TwoCropsTransform
from src.utils.logging import AverageMeter, ProgressMeter
from src.utils.checkpoints import save_checkpoint

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
    return parser.parse_args()

def main():
    args = get_args()
    config = Config()
    
    if hasattr(config, 'print_vram_estimate'):
        config.print_vram_estimate()

    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        np.random.seed(config.seed)

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

    # Create model
    print(f"=> creating model with {config.backbone} backbone")
    def backbone_fn():
        # MoCo v3 trick: freeze patch projection
        return get_backbone(config.backbone, stop_grad_conv1=True)

    model = MoCoV3(
        backbone_fn,
        dim=config.feature_dim,
        mlp_dim=config.mlp_dim,
        T=config.temperature,
        m=config.momentum,
        K=config.queue_size if hasattr(config, 'queue_size') else 65536
    )

    if args.multiprocessing_distributed:
        # Enable SyncBN for multi-GPU stability
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.num_workers = int((config.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # All-reduce survey (comment out if not needed)
        model = model.cuda()

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), config.learning_rate,
                                     weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), config.learning_rate,
                                    momentum=0.9,
                                    weight_decay=config.weight_decay)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    cudnn.benchmark = True

    # Data loading code
    traindir = config.dataset_path
    
    aug1, aug2 = get_moco_v3_augmentations(config.image_size)
    train_dataset = PV03SSLDataset(
        traindir,
        TwoCropsTransform(aug1, aug2)
    )

    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None),
        num_workers=config.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    min_loss = float('inf')

    for epoch in range(config.epochs):
        if args.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        avg_loss = train(train_loader, model, optimizer, scaler, epoch, args, config)

        # Check for best model (only on main process)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            
            is_best = avg_loss < min_loss
            min_loss = min(avg_loss, min_loss)

            # Define experiment-specific directory
            exp_tag = config.get_experiment_tag()
            exp_dir = os.path.join(config.checkpoint_dir, exp_tag)

            # Save 'last.pth' every epoch, and 'best.pth' if improved
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': config.backbone,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, is_best=is_best, filename='last.pth', best_filename='best.pth', checkpoint_dir=exp_dir)

def train(train_loader, model, optimizer, scaler, epoch, args, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    
    # Switch to train mode
    model.train()

    # Use tqdm only on the main process
    is_main_process = not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % torch.cuda.device_count() == 0)
    
    pbar = None
    if is_main_process:
        pbar = tqdm(total=len(train_loader), desc=f"Epoch [{epoch}]")

    end = time.time()
    optimizer.zero_grad()
    
    for i, (images) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Adjust learning rate and momentum
        cur_lr = adjust_learning_rate(optimizer, epoch, i, len(train_loader), config)
        cur_m = adjust_moco_momentum(epoch, i, len(train_loader), config)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # Compute output and loss with mixed precision
        with torch.cuda.amp.autocast(enabled=config.use_amp):
            loss = model(images[0], images[1], cur_m, use_queue=config.use_queue)
            # Normalize loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps

        # Record loss
        losses.update(loss.item() * config.gradient_accumulation_steps, images[0].size(0))

        # Compute gradient
        scaler.scale(loss).backward()

        # Update weights every 'accumulation_steps'
        if (i + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if pbar:
            pbar.update(1)
            pbar.set_postfix({'loss': f"{losses.avg:.4f}", 'lr': f"{cur_lr:.5f}"})

    if pbar:
        pbar.close()
            
    return losses.avg

def adjust_learning_rate(optimizer, epoch, i, iter_per_epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    T = epoch * iter_per_epoch + i
    warmup_epochs = config.warmup_epochs if hasattr(config, 'warmup_epochs') else 10
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
