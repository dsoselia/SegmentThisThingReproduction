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
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent / "repo"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # for evaluation/

from segment_this_thing import (
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
from foveation_lr import LogRectilinearFoveator
from losses import segmentation_loss, expected_iou
from lr_scheduler import WarmupCosineScheduler

@torch.no_grad()
def run_val_seg(model_mod, foveator, val_loader, imagenet_mean, imagenet_std,
                device, step, use_wandb):
    """Validation pass on rank 0 only.

    Logs:
      val/loss, val/exp_iou, val/pred_iou  — computed in foveated token space
      val/crop_iou                          — binary IoU in 1280×1280 crop space
                                             (method-agnostic; comparable across foveators)
    """
    model_mod.eval()
    total_loss = 0.0
    total_exp_iou = 0.0
    total_pred_iou = 0.0
    total_crop_iou = 0.0
    n_batches = 0

    has_unwarp = hasattr(foveator, 'unwarp_to_crop')

    for img_crops, mask_crops, valid_masks in val_loader:
        img_crops   = img_crops.to(device)
        mask_crops  = mask_crops.to(device)   # (B, 1280, 1280) float32 GT in crop space
        valid_masks = valid_masks.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens_u8 = batch_foveate_images(foveator, img_crops)
            tokens    = (tokens_u8 / 255.0 - imagenet_mean) / imagenet_std
            gt_masks  = batch_foveate_masks(foveator, mask_crops)
            pred_masks, pred_ious = model_mod(tokens, valid_masks)
            loss = segmentation_loss(pred_masks, pred_ious, gt_masks, valid_masks)

        B = pred_masks.shape[0]
        batch_exp_iou = 0.0
        batch_pred_iou = 0.0
        best_k_per_sample = []
        count = 0
        for b in range(B):
            valid = valid_masks[b]
            if valid.sum() == 0:
                best_k_per_sample.append(0)
                continue
            pred_b  = pred_masks[b, :, valid].flatten(1)   # (K, M)
            gt_b    = gt_masks[b, valid].flatten()           # (M,)
            e_ious  = expected_iou(pred_b.float(), gt_b.float())  # (K,)
            best_k  = e_ious.argmax().item()
            best_k_per_sample.append(best_k)
            batch_exp_iou  += e_ious[best_k].item()
            batch_pred_iou += pred_ious[b, best_k].sigmoid().item()
            count += 1

        total_loss     += loss.item()
        total_exp_iou  += batch_exp_iou  / max(count, 1)
        total_pred_iou += batch_pred_iou / max(count, 1)

        # ── Crop-space IoU (method-agnostic) ──────────────────────────────────
        if has_unwarp:
            # Select best mask per sample: (B, N, P, P)
            best_pred = torch.stack(
                [pred_masks[b, best_k_per_sample[b]] for b in range(B)]
            ).float()  # (B, N, P, P)
            # Downsample to token resolution if decoder upsampled spatially
            P = best_pred.shape[-1]
            if P != foveator.token_size:
                best_pred = F.adaptive_avg_pool2d(
                    best_pred.flatten(0, 1).unsqueeze(1), foveator.token_size
                ).squeeze(1).unflatten(0, (B, -1))
            soft_pred = best_pred.sigmoid()                     # (B, N, 16, 16) in [0,1]
            pred_crop = foveator.unwarp_to_crop(soft_pred)      # (B, 1280, 1280)
            pred_bin  = pred_crop > 0.5
            gt_bin    = mask_crops.float() > 0.5
            inter = (pred_bin & gt_bin).float().flatten(1).sum(1)   # (B,)
            union = (pred_bin | gt_bin).float().flatten(1).sum(1)   # (B,)
            total_crop_iou += (inter / (union + 1e-6)).mean().item()

        n_batches += 1

    avg_loss     = total_loss     / max(n_batches, 1)
    avg_exp_iou  = total_exp_iou  / max(n_batches, 1)
    avg_pred_iou = total_pred_iou / max(n_batches, 1)
    avg_crop_iou = total_crop_iou / max(n_batches, 1) if has_unwarp else float('nan')

    print(f"  val/loss={avg_loss:.4f}  val/exp_iou={avg_exp_iou:.4f}"
          f"  val/pred_iou={avg_pred_iou:.4f}  val/crop_iou={avg_crop_iou:.4f}")
    if use_wandb:
        import wandb
        log_dict = {
            "val/loss":     avg_loss,
            "val/exp_iou":  avg_exp_iou,
            "val/pred_iou": avg_pred_iou,
        }
        if has_unwarp:
            log_dict["val/crop_iou"] = avg_crop_iou
        wandb.log(log_dict, step=step)

    model_mod.train()


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
    parser.add_argument("--val-data-root",  default=None,
                        help="COCO val2017 images directory")
    parser.add_argument("--val-ann-file",   default=None,
                        help="COCO val2017 instances annotation JSON")
    parser.add_argument("--val-images",     type=int, default=64,
                        help="Number of val images (default 64)")
    parser.add_argument("--val-interval",   type=int, default=500,
                        help="Run val every N steps (0 = disable)")
    parser.add_argument("--eval-images",    type=int, default=0,
                        help="COCO images for SAM-protocol full-res eval (0 = disabled)")
    parser.add_argument("--eval-interval",  type=int, default=0,
                        help="Steps between SAM-protocol eval runs (0 = same as val-interval)")
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
            # Persist run ID so restarts continue the same run (train+val on one timeline)
            _run_id_file = os.path.join(args.output_dir, "wandb_run_id.txt")
            _wandb_id = None
            if os.path.exists(_run_id_file):
                with open(_run_id_file) as _f:
                    _wandb_id = _f.read().strip() or None
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"seg-{args.model_size}",
                id=_wandb_id,
                config=vars(args),
                resume="allow",
            )
            # Save run ID for future restarts
            with open(_run_id_file, "w") as _f:
                _f.write(wandb.run.id)
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without it.")
            use_wandb = False

    # ── Foveator (CPU for dataset, GPU for tokenization) ─────────────────
    token_size = 16
    foveator_cpu = LogRectilinearFoveator()
    foveator_gpu = LogRectilinearFoveator().to(device)

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

    # ── Val loader (rank 0 only, no DDP) ─────────────────────────────────
    val_loader = None
    if is_main and args.val_data_root and args.val_ann_file and args.val_interval > 0:
        val_ds = COCOMultiFovDataset(
            args.val_data_root, foveator_cpu,
            ann_file=args.val_ann_file,
            max_fov=args.max_fov,
            is_mae=False,
        )
        n_val = min(args.val_images, len(val_ds))
        val_subset = torch.utils.data.Subset(val_ds, list(range(n_val)))
        val_loader = DataLoader(
            val_subset,
            batch_size=args.images_per_gpu,
            shuffle=False,
            num_workers=2,
            collate_fn=multi_fov_collate,
            pin_memory=True,
            drop_last=False,
        )
        print(f"Val loader: {n_val} images, interval={args.val_interval}")

    eval_interval = args.eval_interval if args.eval_interval > 0 else args.val_interval

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

    # ── Initial val (on resume, so we have a baseline at the resumed step) ──
    if val_loader is not None and start_step > 0:
        run_val_seg(model.module, foveator_gpu, val_loader,
                    imagenet_mean, imagenet_std, device, start_step, use_wandb)

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
        # ceil so actual effective batch ≥ target regardless of gpu count/batch size
        accum_steps  = max(1, math.ceil(local_target / samples_per_step_per_gpu))
        actual_global_bs = samples_per_step_per_gpu * accum_steps * world

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
                f"lr={lr_:.2e}  "
                f"bs={actual_global_bs}(tgt={global_bs})  "
                f"sps={sps:.0f}  accum={accum_steps}"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss": avg,
                    "train/lr": lr_,
                    "train/sps": sps,
                    "train/global_bs": actual_global_bs,
                    "train/global_bs_target": global_bs,
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

        if val_loader is not None and step % args.val_interval == 0:
            run_val_seg(model.module, foveator_gpu, val_loader,
                        imagenet_mean, imagenet_std, device, step, use_wandb)

        if is_main and args.eval_images > 0 and args.val_data_root and args.val_ann_file \
                and step % eval_interval == 0:
            from evaluation.eval import evaluate_on_coco
            print(f"SAM-protocol eval on {args.eval_images} COCO images...", flush=True)
            model.module.eval()
            sam_miou = evaluate_on_coco(
                args.val_data_root, args.val_ann_file,
                model.module, foveator_gpu,
                imagenet_mean, imagenet_std, device,
                max_images=args.eval_images,
            )
            model.module.train()
            print(f"  eval/sam_miou={sam_miou:.4f}", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"eval/sam_miou": sam_miou}, step=step)
            dist.barrier()

    if is_main:
        save_checkpoint(_full_state(), os.path.join(args.output_dir, "final.pth"))
        print("Training complete.")
        if use_wandb:
            import wandb
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
