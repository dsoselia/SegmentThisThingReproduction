#!/usr/bin/env python3
"""
Standalone evaluation: runs both seg and MAE evals on a single GPU.

Outputs (all in crop / pixel space — comparable across foveation methods):
  eval/sam_miou         — SAM-protocol seg IoU (pred_iou head picks best mask)
  mae/crop_mse          — MAE reconstruction MSE in 1280×1280 crop space
  mae/crop_psnr         — derived from crop_mse (dB)

Usage:
  python evaluation/run_eval.py \
      --seg-ckpt  checkpoints/seg_coco_lr_actual/latest.pth \
      --mae-ckpts checkpoints/mae_coco_lr_actual/step_0025000.pth \
                  checkpoints/mae_coco_lr_actual/step_0050000.pth \
                  checkpoints/mae_coco_lr_actual/step_0075000.pth \
      --val-images-dir /fs/cml-datasets/coco/images/val2017 \
      --val-ann-file   /fs/cml-datasets/coco/annotations/instances_val2017.json \
      --eval-images 500 \
      --model-size b
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "repo"))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT))

from segment_this_thing import build_segment_this_thing_b, build_segment_this_thing_l, build_segment_this_thing_h
from segment_this_thing.utils import get_imagenet_mean, get_imagenet_std
from foveation_lr import LogRectilinearFoveator
from datasets import batch_foveate_images

MODEL_BUILDERS = {
    "b": build_segment_this_thing_b,
    "l": build_segment_this_thing_l,
    "h": build_segment_this_thing_h,
}


# ── MAE reconstruction eval ───────────────────────────────────────────────────

def eval_mae_crop(mae_ckpt_path: str, foveator, imagenet_mean, imagenet_std,
                  images_dir: str, ann_file: str, device, max_images: int,
                  mask_ratio: float = 0.75, model_size: str = "b") -> dict:
    """
    Evaluate MAE reconstruction quality in 1280×1280 crop space.

    Masks mask_ratio of tokens, reconstructs, unwarps to crop space, computes
    MSE vs original crop. Returns dict with crop_mse and crop_psnr.
    """
    import numpy as np
    import matplotlib.image
    from pycocotools.coco import COCO
    from segment_this_thing.utils import get_crop_bounds, get_centered_crop
    from train_mae import MAEMaskToken, MAEDecoder

    token_size = foveator.token_size
    num_tokens = foveator.get_num_tokens()
    feature_dim = 256

    # Build and load MAE model
    full_model = MODEL_BUILDERS[model_size](token_size=token_size, num_tokens=num_tokens)
    encoder = full_model.image_encoder.to(device)
    mae_mask_token = MAEMaskToken(token_size=token_size, channels=3).to(device)
    mae_dec = MAEDecoder(feature_dim=feature_dim, patch_size=token_size, channels=3).to(device)

    ckpt = torch.load(mae_ckpt_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    mae_dec.load_state_dict(ckpt["mae_decoder"])
    mae_mask_token.load_state_dict(ckpt["mae_mask_token"])
    encoder.eval(); mae_dec.eval(); mae_mask_token.eval()

    step = ckpt.get("step", 0)
    n_mask = int(num_tokens * mask_ratio)

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()[:max_images]

    total_mse = 0.0
    n_samples = 0

    with torch.no_grad():
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = Path(images_dir) / img_info["file_name"]
            if not img_path.exists():
                continue

            img_np = matplotlib.image.imread(str(img_path))
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)
            image = torch.from_numpy(img_np.copy()).to(device)
            H, W = image.shape[:2]

            # Use image center as foveation point
            cx, cy = W // 2, H // 2
            fov_center = torch.tensor([cx, cy], device=device)
            from segment_this_thing.utils import get_crop_bounds, get_centered_crop
            crop_bounds = get_crop_bounds(fov_center.float(), foveator.get_pattern_bounds_size()).to(device)
            crop_hwc = get_centered_crop(image.cpu(), crop_bounds.cpu()).to(device)  # (1280,1280,3)

            tokens_u8 = batch_foveate_images(foveator, crop_hwc.unsqueeze(0))  # (1, N, 3, P, P)
            tokens = (tokens_u8 / 255.0 - imagenet_mean) / imagenet_std         # (1, N, 3, P, P)

            # Mask tokens
            noise = torch.rand(1, num_tokens, device=device)
            mask_idx = noise.argsort(dim=1)[:, :n_mask]  # (1, n_mask)
            tokens_masked = mae_mask_token(tokens, mask_idx)
            enc_valid = torch.ones(1, num_tokens, dtype=torch.bool, device=device)
            enc_valid.scatter_(1, mask_idx, False)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                enc_feat, _ = encoder(tokens_masked, enc_valid)
                recon = mae_dec(enc_feat)  # (1, N, 3*P*P)

            # Step 3: unnormalize both GT and predicted tokens → [0,1] pixel range
            P = token_size
            recon_tokens = recon.float().reshape(1, num_tokens, 3, P, P)
            recon_pixels = (recon_tokens * imagenet_std.view(1, 1, 3, 1, 1)
                            + imagenet_mean.view(1, 1, 3, 1, 1)).clamp(0, 1)
            orig_pixels  = (tokens.float() * imagenet_std.view(1, 1, 3, 1, 1)
                            + imagenet_mean.view(1, 1, 3, 1, 1)).clamp(0, 1)

            # Step 4: project both through the same foveated → crop-space mapping.
            # unwarp_to_crop(B=3, N, P, P) → (3, 1280, 1280), one output per channel.
            # Equivalent to STT's generate_foveated_visualization on both sides.
            recon_vis = foveator.unwarp_to_crop(recon_pixels[0].permute(1, 0, 2, 3))  # (3,1280,1280)
            orig_vis  = foveator.unwarp_to_crop(orig_pixels[0].permute(1, 0, 2, 3))   # (3,1280,1280)

            # Step 5: MSE between the two (3, 1280, 1280) crop-space images
            total_mse += F.mse_loss(recon_vis, orig_vis).item()
            n_samples += 1
            n_samples += 1

            if n_samples % 50 == 0:
                avg = total_mse / n_samples
                psnr = -10 * math.log10(avg) if avg > 0 else float('inf')
                print(f"  [{n_samples}/{max_images}] crop_mse={avg:.5f}  crop_psnr={psnr:.2f}dB",
                      flush=True)

    avg_mse = total_mse / max(n_samples, 1)
    crop_psnr = -10 * math.log10(avg_mse) if avg_mse > 0 else float('inf')
    return {"step": step, "crop_mse": avg_mse, "crop_psnr": crop_psnr, "n_samples": n_samples}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Log-rect eval: seg + MAE")
    parser.add_argument("--seg-ckpt",       default=None, help="Seg checkpoint (latest.pth)")
    parser.add_argument("--mae-ckpts",      nargs="*", default=[], help="MAE checkpoint(s)")
    parser.add_argument("--val-images-dir", required=True)
    parser.add_argument("--val-ann-file",   required=True)
    parser.add_argument("--eval-images",    type=int, default=500)
    parser.add_argument("--model-size",     default="b", choices=["b", "l", "h"])
    parser.add_argument("--mask-ratio",     type=float, default=0.75)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    token_size = 16
    foveator = LogRectilinearFoveator().to(device)
    imagenet_mean = get_imagenet_mean(device).view(1, 1, 3, 1, 1)
    imagenet_std  = get_imagenet_std(device).view(1, 1, 3, 1, 1)

    # ── Seg eval ──────────────────────────────────────────────────────────────
    if args.seg_ckpt:
        from evaluation.eval import evaluate_on_coco

        num_tokens = foveator.get_num_tokens()
        model = MODEL_BUILDERS[args.model_size](
            token_size=token_size, num_tokens=num_tokens
        ).to(device)
        ckpt = torch.load(args.seg_ckpt, map_location="cpu", weights_only=False)
        seg_step = ckpt.get("step", 0)
        model.load_state_dict(ckpt["model"])
        model.eval()

        print(f"\n── Seg eval (step {seg_step}) ───────────────────────────────────")
        print(f"   {args.eval_images} COCO images, SAM protocol (furthest-from-boundary click)")
        sam_miou = evaluate_on_coco(
            args.val_images_dir, args.val_ann_file,
            model, foveator, imagenet_mean, imagenet_std, device,
            max_images=args.eval_images,
        )
        print(f"\n  eval/sam_miou @ step {seg_step} = {sam_miou:.4f}")

    # ── MAE crop-space reconstruction eval ───────────────────────────────────
    for mae_ckpt in args.mae_ckpts:
        print(f"\n── MAE crop-space eval: {Path(mae_ckpt).name} ─────────────────────")
        result = eval_mae_crop(
            mae_ckpt, foveator, imagenet_mean, imagenet_std,
            args.val_images_dir, args.val_ann_file, device,
            max_images=args.eval_images,
            mask_ratio=args.mask_ratio,
            model_size=args.model_size,
        )
        print(f"\n  step={result['step']}  crop_mse={result['crop_mse']:.5f}"
              f"  crop_psnr={result['crop_psnr']:.2f}dB  (n={result['n_samples']})")


if __name__ == "__main__":
    main()
