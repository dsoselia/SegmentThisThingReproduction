#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Segmentation fine-tuning for Segment This Thing — paper-faithful implementation.
#
# Paper hyperparameters (supplementary §6.2):
#   - 250K iterations
#   - Batch size: starts at 2048, doubles every 50K steps
#   - AdamW: lr=2^-16 ≈ 1.53e-5, weight_decay=0.001
#   - 5K-step linear warmup, then cosine decay to 0
#   - Up to 16 foveated views per image, 256 px boundary margin
#   - Losses: focal(20) + dice(1) + IoU_pred(0.01)
#   - IoU target: expected IoU of real-valued masks (Eq. 1, no thresholding)
#   - Foveation center: uniform random within segment (NOT furthest-from-boundary)
#
# Throughput optimizations (beyond paper):
#   - bf16 AMP for 2× inference speedup with half the VRAM
#   - GPU-side batched foveation (moves tokenization bottleneck off CPU workers)
#   - Gradient accumulation to match paper's large effective batch sizes
#   - persistent_workers + prefetch_factor for efficient data loading
#
# Usage (single node, 2× A6000):
#   torchrun --nproc_per_node=2 training/train_seg.py \
#       --data-root data/sa1b/extracted \
#       --output-dir checkpoints/seg

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent / "repo"))

from segment_this_thing import (
    Foveator,
    build_segment_this_thing_b,
    build_segment_this_thing_l,
    build_segment_this_thing_h,
)
from segment_this_thing.utils import get_imagenet_mean, get_imagenet_std

from datasets import (
    SA1BMultiFovDataset,
    SA1BTarMultiFovDataset,
    COCOMultiFovDataset,
    multi_fov_collate,
    batch_foveate_images,
    batch_foveate_masks,
)
from losses import segmentation_loss
from lr_scheduler import WarmupCosineScheduler

MODEL_BUILDERS = {
    "b": build_segment_this_thing_b,
    "l": build_segment_this_thing_l,
    "h": build_segment_this_thing_h,
}

# Batch size schedule (paper: doubles every 50K steps, starting at 2048)
BASE_BATCH = 2048
DOUBLE_INTERVAL = 50_000


def effective_batch(step: int) -> int:
    doublings = min(step // DOUBLE_INTERVAL, 4)  # cap at 32768
    return BASE_BATCH * (2 ** doublings)


def save_checkpoint(state: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser("STT Segmentation Training")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-source", default="sa1b", choices=["sa1b", "coco"],
                        help="Dataset source: sa1b or coco")
    parser.add_argument("--ann-file", default=None,
                        help="COCO instances annotation JSON (required if --data-source coco)")
    parser.add_argument("--model-size", default="b", choices=["b", "l", "h"])
    parser.add_argument("--pretrain-ckpt", default=None,
                        help="MAE pre-trained encoder checkpoint (.pth)")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--total-steps",   type=int, default=250_000)
    parser.add_argument("--warmup-steps",  type=int, default=5_000)
    parser.add_argument("--lr",            type=float, default=2 ** -16)
    parser.add_argument("--weight-decay",  type=float, default=0.001)
    parser.add_argument("--max-fov",       type=int, default=16,
                        help="Foveated views per image (paper: 16)")
    parser.add_argument("--images-per-gpu", type=int, default=2,
                        help="Images per GPU per step (each produces --max-fov samples)")
    parser.add_argument("--num-workers",   type=int, default=8)
    parser.add_argument("--log-interval",  type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=2_500)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for extra speed")
    parser.add_argument("--wandb-project", default="segment-this-thing",
                        help="W&B project name (set to empty string to disable)")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    # ── Distributed setup ─────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank      = dist.get_rank()
    world     = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device    = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main   = rank == 0

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Preemption signal handling ────────────────────────────────────────────
    _exit_requested = threading.Event()

    def _request_exit(sig, frame):
        _exit_requested.set()

    signal.signal(signal.SIGUSR1, _request_exit)
    signal.signal(signal.SIGTERM, _request_exit)

    # ── W&B (rank 0 only) ─────────────────────────────────────────────────────
    use_wandb = bool(args.wandb_project) and is_main
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"seg-{args.model_size}",
                config=vars(args),
                resume="allow",
            )
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without it.")
            use_wandb = False

    # ── Foveator (CPU for dataset, GPU for tokenization) ─────────────────
    token_size = 16
    foveator_cpu = Foveator(
        token_size=token_size, strides=[1, 2, 4, 6, 8], grid_sizes=[4, 4, 6, 8, 10]
    )
    foveator_gpu = Foveator(
        token_size=token_size, strides=[1, 2, 4, 6, 8], grid_sizes=[4, 4, 6, 8, 10]
    ).to(device)

    imagenet_mean = get_imagenet_mean(device).view(1, 1, 3, 1, 1)  # (1,1,3,1,1)
    imagenet_std  = get_imagenet_std(device).view(1, 1, 3, 1, 1)

    # ── Dataset ───────────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    if args.data_source == "coco":
        assert args.ann_file, "--ann-file required with --data-source coco"
        dataset = COCOMultiFovDataset(
            str(data_root), foveator_cpu,
            ann_file=args.ann_file,
            max_fov=args.max_fov,
            is_mae=False,
        )
    elif data_root.is_dir() and any(data_root.glob("*.json")):
        dataset = SA1BMultiFovDataset(str(data_root), foveator_cpu, max_fov=args.max_fov)
    else:
        tar_files = sorted(str(p) for p in data_root.glob("*.tar"))
        dataset = SA1BTarMultiFovDataset(tar_files, foveator_cpu, max_fov=args.max_fov)

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader  = DataLoader(
        dataset,
        batch_size=args.images_per_gpu,       # images per GPU; each → max_fov samples
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=multi_fov_collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    num_tokens = foveator_cpu.get_num_tokens()
    model = MODEL_BUILDERS[args.model_size](
        token_size=token_size, num_tokens=num_tokens
    ).to(device)

    if args.pretrain_ckpt and os.path.exists(args.pretrain_ckpt):
        if is_main:
            print(f"Loading MAE encoder from {args.pretrain_ckpt}")
        ckpt = torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=False)
        # Support both old format (image_encoder key) and new format (encoder key)
        if "image_encoder" in ckpt:
            enc_state = ckpt["image_encoder"]
        elif "encoder" in ckpt:
            enc_state = ckpt["encoder"]
        else:
            enc_state = {k.removeprefix("image_encoder."): v
                         for k, v in ckpt.items() if k.startswith("image_encoder.")}
        missing, unexpected = model.image_encoder.load_state_dict(enc_state, strict=False)
        if is_main:
            print(f"  Missing: {missing}  Unexpected: {unexpected}")

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    model = DDP(model, device_ids=[local_rank])

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=args.total_steps
    )

    scaler = torch.amp.GradScaler("cuda")

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        if is_main:
            print(f"Resumed from step {start_step}")

    # ── Training ──────────────────────────────────────────────────────────
    model.train()
    torch.backends.cudnn.benchmark = True

    step       = start_step
    loader_it  = iter(loader)
    loss_accum = 0.0
    n_accum    = 0
    t0         = time.time()

    # samples per step per GPU = images_per_gpu * max_fov
    samples_per_step_per_gpu = args.images_per_gpu * args.max_fov

    def _full_state():
        return {
            "step": step,
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }

    while step < args.total_steps:
        # ── Preemption check (all ranks reach this together after each step) ──
        if _exit_requested.is_set():
            if is_main:
                print(f"[preempt] Signal received at step {step}, saving resume checkpoint …")
                save_checkpoint(_full_state(), os.path.join(args.output_dir, "latest.pth"))
                print("[preempt] Saved. Exiting cleanly for requeue.")
            dist.barrier()
            dist.destroy_process_group()
            sys.exit(0)

        # Number of gradient-accumulation micro-steps to match paper's batch schedule
        global_bs  = effective_batch(step)
        local_target = global_bs // world
        accum_steps  = max(1, local_target // samples_per_step_per_gpu)

        optimizer.zero_grad(set_to_none=True)

        for _ in range(accum_steps):
            try:
                img_crops, mask_crops, valid_masks = next(loader_it)
            except StopIteration:
                sampler.set_epoch(step)
                loader_it = iter(loader)
                img_crops, mask_crops, valid_masks = next(loader_it)

            img_crops   = img_crops.to(device, non_blocking=True)    # (B,1280,1280,3) uint8
            mask_crops  = mask_crops.to(device, non_blocking=True)   # (B,1280,1280) float
            valid_masks = valid_masks.to(device, non_blocking=True)  # (B,N) bool

            with torch.autocast("cuda", dtype=torch.bfloat16):
                # GPU foveation (fast, integral images on CUDA)
                tokens_u8 = batch_foveate_images(foveator_gpu, img_crops)  # (B,N,3,P,P) [0,255]
                tokens    = (tokens_u8 / 255.0 - imagenet_mean) / imagenet_std  # normalize
                gt_masks  = batch_foveate_masks(foveator_gpu, mask_crops)       # (B,N,P,P)

                pred_masks, pred_ious = model(tokens, valid_masks)
                loss = segmentation_loss(pred_masks, pred_ious, gt_masks, valid_masks)

            scaler.scale(loss / accum_steps).backward()
            loss_accum += loss.item()
            n_accum += 1

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1

        if is_main and step % args.log_interval == 0:
            avg  = loss_accum / max(n_accum, 1)
            lr_  = scheduler.get_last_lr()[0]
            # sps = actual samples through GPU (per micro-batch * accum * GPUs * steps)
            sps  = samples_per_step_per_gpu * accum_steps * world * args.log_interval / (time.time() - t0)
            print(
                f"step={step:6d}/{args.total_steps}  loss={avg:.4f}  "
                f"lr={lr_:.2e}  global_bs={global_bs}  "
                f"sps={sps:.0f}  accum={accum_steps}"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss": avg,
                    "train/lr": lr_,
                    "train/sps": sps,
                    "train/global_bs": global_bs,
                    "train/accum_steps": accum_steps,
                }, step=step)
            loss_accum, n_accum = 0.0, 0
            t0 = time.time()

        if is_main and step % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"step_{step:07d}.pth")
            state = _full_state()
            save_checkpoint(state, ckpt_path)
            save_checkpoint(state, os.path.join(args.output_dir, "latest.pth"))
            print(f"Saved {ckpt_path}")

    if is_main:
        save_checkpoint(_full_state(), os.path.join(args.output_dir, "final.pth"))
        print("Training complete.")
        if use_wandb:
            import wandb
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
