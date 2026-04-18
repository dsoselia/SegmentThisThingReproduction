#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# MAE pre-training for Segment This Thing image encoder — paper-faithful.
#
# Paper hyperparameters (supplementary §6.1):
#   - 500K iterations
#   - AdamW: lr=2^-13 ≈ 1.22e-4, weight_decay=0.001
#   - 10K-step linear warmup, then CONSTANT LR (no cosine decay)
#   - Batch size: starts at 1024, doubles every 100K steps
#   - Masking ratio: 0.75 (mask 75% of 172 tokens)
#   - 2 foveated views per image (each yields one training example)
#   - Foveation centers: uniform random, 256 px boundary margin
#   - MAE target: foveated/downsampled tokens (NOT full-res source image)
#   - Loss only on MASKED tokens; visible tokens receive no reconstruction loss
#
# Architecture note:
#   The official repo has no MAE decoder.  We add a lightweight pixel-prediction
#   MLP (He et al. 2022 style), discard it after pre-training.

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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent / "repo"))

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
)
from foveation_lr import LogRectilinearFoveator
from lr_scheduler import WarmupThenConstant

# Center 4×4 token indices for MAE visualization (rows 4-7, cols 4-7 in 13×13 grid)
_LR_VIS_IDX = [r * 13 + c for r in range(4, 8) for c in range(4, 8)]

MODEL_BUILDERS = {
    "b": build_segment_this_thing_b,
    "l": build_segment_this_thing_l,
    "h": build_segment_this_thing_h,
}

BASE_BATCH = 1024
DOUBLE_INTERVAL = 100_000
MASK_RATIO = 0.75


def effective_batch(step: int) -> int:
    doublings = min(step // DOUBLE_INTERVAL, 4)
    return BASE_BATCH * (2 ** doublings)


def save_checkpoint(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight MAE decoder
# ──────────────────────────────────────────────────────────────────────────────

class MAEMaskToken(nn.Module):
    """
    Learnable mask token used to replace masked foveated patches in the
    encoder input.  The encoder still sees all N positions (with mask tokens
    at masked locations and valid_token_mask=False there), preserving the
    position encoding alignment.  The decoder then reconstructs all N patches
    from the encoder output.
    """

    def __init__(self, token_size: int, channels: int = 3):
        super().__init__()
        # Mask token in PATCH space (same dtype/shape as one foveated token)
        self.mask_pixel = nn.Parameter(
            torch.zeros(channels, token_size, token_size)
        )

    def forward(
        self,
        tokens: torch.Tensor,           # (B, N, C, P, P)  normalised float
        mask_indices: torch.Tensor,     # (B, n_mask) long
    ) -> torch.Tensor:
        """Replace masked positions with the learned mask pixel value."""
        B, N, C, P, _ = tokens.shape
        out = tokens.clone()
        mask_pix = self.mask_pixel.view(1, 1, C, P, P)
        # Scatter mask token into masked positions
        idx_exp = mask_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, C, P, P
        )
        out.scatter_(1, idx_exp, mask_pix.expand(B, mask_indices.shape[1], C, P, P))
        return out


class MAEDecoder(nn.Module):
    """
    Two-layer MLP that reconstructs foveated token pixel values from
    encoder output features.

    Input:  (B, N, feature_dim)  — full sequence encoder output
    Output: (B, N, C*P*P)        — reconstructed normalised pixels
    """

    def __init__(self, feature_dim: int, patch_size: int, channels: int = 3,
                 hidden_dim: int = 512):
        super().__init__()
        self.out_dim = channels * patch_size * patch_size
        self.norm = nn.LayerNorm(feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, N, feature_dim) → (B, N, C*P*P)"""
        return self.head(self.norm(features))


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_recon_grid(tokens_gt, mask_idx, recon, imagenet_std, imagenet_mean,
                     n_show=4, grid_w=4, vis_indices=None):
    """
    Build a PIL image: rows of [GT patches | masked input | reconstruction].

    vis_indices: list of token indices to display (default: first grid_w² tokens,
                 which are the finest-resolution center tokens for the ring foveator).
                 For LogRectilinearFoveator pass _LR_VIS_IDX (center 4×4 tokens).
    Returns a PIL.Image or None on any error.
    """
    try:
        import numpy as np
        from PIL import Image as PILImage

        B, N, C, P, _ = tokens_gt.shape
        n_show = min(n_show, B)
        grid_n = grid_w * grid_w  # 16 tokens to display

        if vis_indices is None:
            idx = list(range(grid_n))
        else:
            idx = list(vis_indices)[:grid_n]

        idx_t = torch.tensor(idx, dtype=torch.long, device=tokens_gt.device)

        # imagenet_std/mean are (1,1,3,1,1) — squeeze to (1,3,1,1) for broadcast
        std  = imagenet_std.squeeze(0).float()
        mean = imagenet_mean.squeeze(0).float()

        def patches_to_img(patches):
            """(grid_n, 3, P, P) float [0,1] → np (grid_w*P, grid_w*P, 3)"""
            rows = []
            for r in range(grid_w):
                row = patches[r * grid_w:(r + 1) * grid_w]      # (grid_w, 3, P, P)
                row_np = row.permute(0, 2, 3, 1).cpu().numpy()   # (grid_w, P, P, 3)
                rows.append(np.concatenate(row_np, axis=1))      # (P, grid_w*P, 3)
            return np.concatenate(rows, axis=0)                  # (grid_w*P, grid_w*P, 3)

        sample_rows = []
        for i in range(n_show):
            gt_p  = tokens_gt[i].index_select(0, idx_t).float()       # (16, 3, P, P)
            gt_p  = (gt_p * std + mean).clamp(0, 1)

            rec_p = recon[i].index_select(0, idx_t).float().view(grid_n, C, P, P)
            rec_p = (rec_p * std + mean).clamp(0, 1)

            is_masked_i = torch.zeros(N, dtype=torch.bool, device=tokens_gt.device)
            is_masked_i.scatter_(0, mask_idx[i], True)

            msk_p = gt_p.clone()
            for t_local, t_global in enumerate(idx):
                if is_masked_i[t_global]:
                    msk_p[t_local] = 0.5   # gray placeholder

            gt_img  = patches_to_img(gt_p)
            msk_img = patches_to_img(msk_p)
            rec_img = patches_to_img(rec_p)

            H = gt_img.shape[0]
            div_v = np.ones((H, 2, 3), dtype=np.float32)
            row = np.concatenate([gt_img, div_v, msk_img, div_v, rec_img], axis=1)
            sample_rows.append(row)

        W = sample_rows[0].shape[1]
        div_h = np.ones((2, W, 3), dtype=np.float32)
        full = sample_rows[0]
        for r in sample_rows[1:]:
            full = np.concatenate([full, div_h, r], axis=0)

        return PILImage.fromarray((full * 255).clip(0, 255).astype(np.uint8))
    except Exception:
        return None


@torch.no_grad()
def run_val(encoder_mod, mae_mask_token_mod, mae_dec_mod, val_loader,
            foveator_gpu, imagenet_mean, imagenet_std, num_tokens, mask_ratio,
            device, step, use_wandb):
    """
    Run validation on rank 0 only (call with .module to skip DDP).
    Logs val/loss and a reconstruction image grid to W&B.
    """
    encoder_mod.eval()
    mae_dec_mod.eval()

    n_mask = int(num_tokens * mask_ratio)
    total_loss = 0.0
    n_batches  = 0
    first_data = None

    for img_crops, _mask_crops, valid_masks in val_loader:
        img_crops   = img_crops.to(device)
        valid_masks = valid_masks.to(device)
        B = img_crops.shape[0]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens_u8 = batch_foveate_images(foveator_gpu, img_crops)
            tokens    = (tokens_u8 / 255.0 - imagenet_mean) / imagenet_std

            noise = torch.rand(B, num_tokens, device=device)
            noise[~valid_masks] = float("inf")
            sorted_idx = noise.argsort(dim=1)
            mask_idx   = sorted_idx[:, :n_mask]

            tokens_masked = mae_mask_token_mod(tokens, mask_idx)
            enc_valid = valid_masks.clone()
            enc_valid.scatter_(1, mask_idx, False)

            enc_feat, _ = encoder_mod(tokens_masked, enc_valid)
            recon  = mae_dec_mod(enc_feat)        # (B, N, 3*P*P)
            target = tokens.flatten(2)

            is_masked = torch.zeros(B, num_tokens, dtype=torch.bool, device=device)
            is_masked.scatter_(1, mask_idx, True)
            loss_mask = is_masked & valid_masks

            if loss_mask.any():
                total_loss += F.mse_loss(recon[loss_mask], target[loss_mask]).item()
                n_batches  += 1

        if first_data is None:
            # Keep in float32 for visualization (autocast context has exited)
            first_data = (
                tokens.float().detach(),
                mask_idx.detach(),
                recon.float().detach(),
            )

    avg_loss = total_loss / max(n_batches, 1)
    print(f"  val/loss={avg_loss:.4f}  ({n_batches} batches)")

    if use_wandb:
        import wandb
        log_dict = {"val/loss": avg_loss}

        if first_data is not None:
            tokens_gt, mask_idx_0, recon_0 = first_data
            grid = _make_recon_grid(tokens_gt, mask_idx_0, recon_0,
                                    imagenet_std, imagenet_mean, n_show=4,
                                    vis_indices=_LR_VIS_IDX)
            if grid is not None:
                log_dict["val/reconstruction"] = wandb.Image(
                    grid, caption=f"step {step}  |  GT | masked | recon  (center tokens)"
                )

        wandb.log(log_dict, step=step)

    encoder_mod.train()
    mae_dec_mod.train()
    return avg_loss


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("STT MAE Pre-training")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-source", default="sa1b", choices=["sa1b", "coco"],
                        help="Dataset source: sa1b or coco")
    parser.add_argument("--model-size", default="b", choices=["b", "l", "h"])
    parser.add_argument("--total-steps",    type=int, default=500_000)
    parser.add_argument("--warmup-steps",   type=int, default=10_000)
    parser.add_argument("--lr",             type=float, default=2 ** -13)   # ~1.22e-4
    parser.add_argument("--weight-decay",   type=float, default=0.001)
    parser.add_argument("--mask-ratio",     type=float, default=MASK_RATIO)
    parser.add_argument("--images-per-gpu", type=int, default=4,
                        help="Images per GPU per step; each yields 2 foveated views")
    parser.add_argument("--num-workers",    type=int, default=8)
    parser.add_argument("--log-interval",   type=int, default=20)
    parser.add_argument("--save-interval",    type=int, default=10_000,
                        help="Steps between latest.pth updates")
    parser.add_argument("--milestone-interval", type=int, default=25_000,
                        help="Steps between named step_XXXXXXX.pth milestones")
    parser.add_argument("--keep-milestones",   type=int, default=3,
                        help="Number of milestone checkpoints to retain (oldest deleted)")
    parser.add_argument("--val-data-root", default=None,
                        help="Directory of validation images (e.g. COCO val2017). "
                             "Skipped if not provided.")
    parser.add_argument("--val-images", type=int, default=256,
                        help="Number of val images to use (default: 256)")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--wandb-project", default="segment-this-thing",
                        help="W&B project name (set to empty string to disable)")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world      = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main    = rank == 0

    # ── Preemption signal handling ────────────────────────────────────────────
    # SLURM sends SIGUSR1 (via --signal=SIGUSR1@120) 120 s before preemption,
    # and SIGTERM just before SIGKILL.  We catch both and set a flag that the
    # training loop checks at each step boundary (all ranks are synchronised
    # there), so we can save a clean checkpoint before being killed.
    _exit_requested = threading.Event()

    def _request_exit(sig, frame):
        _exit_requested.set()

    signal.signal(signal.SIGUSR1, _request_exit)
    signal.signal(signal.SIGTERM, _request_exit)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── W&B (rank 0 only) ─────────────────────────────────────────────────────
    use_wandb = bool(args.wandb_project) and is_main
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"mae-{args.model_size}",
                config=vars(args),
                resume="allow",
            )
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without it.")
            use_wandb = False

    token_size = 16
    foveator_cpu = LogRectilinearFoveator()
    foveator_gpu = LogRectilinearFoveator().to(device)
    num_tokens = foveator_cpu.get_num_tokens()  # 169

    imagenet_mean = get_imagenet_mean(device).view(1, 1, 3, 1, 1)
    imagenet_std  = get_imagenet_std(device).view(1, 1, 3, 1, 1)

    # Dataset: 2 foveated views per image (paper)
    data_root = Path(args.data_root)
    if args.data_source == "coco":
        dataset = COCOMultiFovDataset(
            str(data_root), foveator_cpu, max_fov=2, is_mae=True
        )
    elif data_root.is_dir() and any(data_root.glob("*.json")):
        dataset = SA1BMultiFovDataset(str(data_root), foveator_cpu, max_fov=2)
    else:
        tar_files = sorted(str(p) for p in data_root.glob("*.tar"))
        dataset = SA1BTarMultiFovDataset(tar_files, foveator_cpu, max_fov=2)

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader  = DataLoader(
        dataset,
        batch_size=args.images_per_gpu,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=multi_fov_collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ── Val loader (rank 0 only, no DDP) ─────────────────────────────────────
    val_loader = None
    if is_main and args.val_data_root:
        val_root = Path(args.val_data_root)
        if val_root.exists():
            val_dataset = COCOMultiFovDataset(
                str(val_root), foveator_cpu, max_fov=2, is_mae=True,
            )
            n_val = min(args.val_images, len(val_dataset))
            val_dataset = torch.utils.data.Subset(val_dataset, list(range(n_val)))
            val_loader  = DataLoader(
                val_dataset,
                batch_size=args.images_per_gpu,
                shuffle=False,
                num_workers=4,
                collate_fn=multi_fov_collate,
                pin_memory=True,
            )
            print(f"Val loader: {n_val} images from {val_root}")
        else:
            print(f"[warn] --val-data-root {args.val_data_root} not found, skipping val.")

    # Encoder only (full model just for weight init)
    full_model = MODEL_BUILDERS[args.model_size](token_size=token_size, num_tokens=num_tokens)
    encoder = full_model.image_encoder.to(device)

    # feature_dim = 256 (neck output of build_segment_this_thing)
    feature_dim = 256
    mae_mask_token = MAEMaskToken(token_size=token_size, channels=3).to(device)
    mae_dec = MAEDecoder(
        feature_dim=feature_dim, patch_size=token_size, channels=3
    ).to(device)

    if args.compile:
        encoder = torch.compile(encoder, mode="reduce-overhead")

    encoder        = DDP(encoder, device_ids=[local_rank])
    mae_dec        = DDP(mae_dec, device_ids=[local_rank])
    mae_mask_token = DDP(mae_mask_token, device_ids=[local_rank])

    params = (list(encoder.parameters())
              + list(mae_dec.parameters())
              + list(mae_mask_token.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Paper: constant LR after warmup (not cosine decay)
    scheduler = WarmupThenConstant(optimizer, warmup_steps=args.warmup_steps)
    scaler    = torch.amp.GradScaler("cuda")

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        encoder.module.load_state_dict(ckpt["encoder"])
        mae_dec.module.load_state_dict(ckpt["mae_decoder"])
        mae_mask_token.module.load_state_dict(ckpt["mae_mask_token"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        if is_main:
            print(f"Resumed from step {start_step}")

    encoder.train()
    mae_dec.train()
    torch.backends.cudnn.benchmark = True

    step       = start_step
    loader_it  = iter(loader)
    loss_accum = 0.0
    n_accum    = 0
    t0         = time.time()

    # samples per step per GPU: images_per_gpu * 2 views
    samples_per_step_per_gpu = args.images_per_gpu * 2

    def _full_state():
        return {
            "step": step,
            "encoder": encoder.module.state_dict(),
            "mae_decoder": mae_dec.module.state_dict(),
            "mae_mask_token": mae_mask_token.module.state_dict(),
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

        global_bs    = effective_batch(step)
        local_target = global_bs // world
        # ceil so actual effective batch ≥ target regardless of gpu count/batch size
        accum_steps  = max(1, math.ceil(local_target / samples_per_step_per_gpu))
        actual_global_bs = samples_per_step_per_gpu * accum_steps * world

        optimizer.zero_grad(set_to_none=True)

        for _ in range(accum_steps):
            try:
                img_crops, _mask_crops, valid_masks = next(loader_it)
            except StopIteration:
                sampler.set_epoch(step)
                loader_it = iter(loader)
                img_crops, _mask_crops, valid_masks = next(loader_it)

            img_crops   = img_crops.to(device, non_blocking=True)
            valid_masks = valid_masks.to(device, non_blocking=True)

            B = img_crops.shape[0]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                # GPU foveation
                tokens_u8 = batch_foveate_images(foveator_gpu, img_crops)  # (B,N,3,P,P) [0,255]
                tokens    = (tokens_u8 / 255.0 - imagenet_mean) / imagenet_std   # (B,N,3,P,P)

                # Random masking: mask 75% of VALID tokens
                # Invalid (out-of-bounds) tokens are never masked — they are
                # excluded from both encoding and loss.
                n_mask = int(num_tokens * args.mask_ratio)

                # Assign infinite noise to invalid tokens so they sort last
                # → they always end up in the "visible" group (not masked)
                noise = torch.rand(B, num_tokens, device=device)
                noise[~valid_masks] = float("inf")
                sorted_idx = noise.argsort(dim=1)          # (B, N)
                mask_idx   = sorted_idx[:, :n_mask]         # (B, n_mask)

                # Build full masked token sequence for the encoder:
                # replace masked positions with the learnable mask pixel value.
                tokens_masked = mae_mask_token(tokens, mask_idx)  # (B, N, 3, P, P)

                # valid_token_mask for encoder: exclude masked AND out-of-bounds tokens
                enc_valid = valid_masks.clone()
                enc_valid.scatter_(1, mask_idx, False)  # mask_idx positions → invisible

                # Encoder processes full N-token sequence (position-encoding compatible)
                enc_feat, _reg = encoder(tokens_masked, enc_valid)  # (B, N, feat_dim)

                # Decoder reconstructs ALL tokens from encoder features
                recon  = mae_dec(enc_feat)   # (B, N, 3*P*P)
                target = tokens.flatten(2)   # (B, N, 3*P*P)  — normalized GT

                # Loss only on MASKED AND VALID tokens
                is_masked = torch.zeros(B, num_tokens, dtype=torch.bool, device=device)
                is_masked.scatter_(1, mask_idx, True)
                loss_mask = is_masked & valid_masks  # (B, N)

                if loss_mask.any():
                    loss = F.mse_loss(recon[loss_mask], target[loss_mask].detach())
                else:
                    loss = tokens.new_zeros(())

            scaler.scale(loss / accum_steps).backward()
            loss_accum += loss.item()
            n_accum += 1

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1

        if is_main and step % args.log_interval == 0:
            avg = loss_accum / max(n_accum, 1)
            lr_ = scheduler.get_last_lr()[0]
            # sps = actual samples through GPU (per micro-batch * accum * GPUs * steps)
            sps = samples_per_step_per_gpu * accum_steps * world * args.log_interval / (time.time() - t0)
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

        if step % args.save_interval == 0:
            # Free cached GPU memory to prevent allocator fragmentation over long runs
            torch.cuda.empty_cache()
            if is_main:
                state = _full_state()
                # Always update latest.pth (used for preemption resume)
                save_checkpoint(state, os.path.join(args.output_dir, "latest.pth"))

                # Save a milestone checkpoint every milestone_interval steps, then
                # rotate: keep only the most recent args.keep_milestones milestones
                # to avoid filling the disk (each checkpoint is ~1 GB).
                if step % args.milestone_interval == 0:
                    ckpt_path = os.path.join(args.output_dir, f"step_{step:07d}.pth")
                    save_checkpoint(state, ckpt_path)
                    print(f"Saved milestone {ckpt_path}")

                    # Rotate: delete oldest milestones beyond keep_milestones
                    import glob as _glob
                    milestones = sorted(
                        _glob.glob(os.path.join(args.output_dir, "step_*.pth"))
                    )
                    while len(milestones) > args.keep_milestones:
                        old = milestones.pop(0)
                        os.remove(old)
                        print(f"  Rotated out {os.path.basename(old)}")
                else:
                    print(f"Checkpoint latest.pth @ step {step}")

                # Validation (rank 0 only — other ranks wait at the barrier below)
                if val_loader is not None:
                    run_val(
                        encoder.module, mae_mask_token.module, mae_dec.module,
                        val_loader, foveator_gpu,
                        imagenet_mean, imagenet_std,
                        num_tokens, args.mask_ratio,
                        device, step, use_wandb,
                    )
            # Barrier so non-zero ranks wait for rank 0's checkpoint+val before
            # proceeding to the next training step (avoids DDP sync deadlock).
            dist.barrier()

    if is_main:
        save_checkpoint(_full_state(), os.path.join(args.output_dir, "final.pth"))
        print("MAE pre-training complete.")
        if use_wandb:
            import wandb
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
