#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Zero-shot single-point valid mask evaluation for Segment This Thing.
#
# Evaluation protocol (from paper):
#   For each ground-truth mask, the click point is the pixel furthest from the
#   mask boundary (i.e., the "point_coords" field already stored in SA-1B annotations).
#   Run the predictor, take the mask with highest predicted IoU, compute actual IoU
#   against the GT mask.  Report mean IoU across all instances.
#
# Usage:
#   python evaluation/eval.py \
#       --data-root data/sa1b/extracted \
#       --weights weights/stt-b.pth \
#       --model-size b

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "repo"))

from segment_this_thing import Foveator, SegmentThisThingPredictor
from segment_this_thing import (
    build_segment_this_thing_b,
    build_segment_this_thing_l,
    build_segment_this_thing_h,
)

MODEL_BUILDERS = {"b": build_segment_this_thing_b, "l": build_segment_this_thing_l, "h": build_segment_this_thing_h}


def rle_decode_to_tensor(rle: dict) -> torch.Tensor:
    from pycocotools.mask import decode as coco_decode
    import numpy as np
    arr = coco_decode(rle)
    return torch.from_numpy(arr.astype(bool))


def compute_iou_full_res(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Compute IoU on full-resolution binary masks."""
    intersection = (pred_mask & gt_mask).sum().item()
    union = (pred_mask | gt_mask).sum().item()
    if union == 0:
        return 1.0
    return intersection / union


def reconstruct_full_mask(
    foveated_logits: torch.Tensor,  # (N, 1, 16, 16) or (K, N, 16, 16) logits
    foveator: Foveator,
    crop_size: int = 1280,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Reconstruct a binary full-resolution mask from foveated logits.
    Uses generate_foveated_visualization → sigmoid → threshold.
    Input: (N, P, P) logits for one mask.
    Output: (crop_size, crop_size) bool.
    """
    vis = foveator.generate_foveated_visualization(
        foveated_logits.unsqueeze(1)  # (N, 1, P, P)
    )  # (1, crop_size, crop_size)
    return vis.squeeze(0).sigmoid() > threshold


def evaluate_on_dataset(
    data_root: str,
    predictor: SegmentThisThingPredictor,
    foveator: Foveator,
    device: torch.device,
    max_images: int = None,
):
    import matplotlib.image
    import numpy as np

    data_root = Path(data_root)
    json_files = sorted(data_root.glob("*.json"))
    if max_images:
        json_files = json_files[:max_images]

    all_ious = []
    n_images = 0

    for jf in json_files:
        img_path = jf.with_suffix(".jpg")
        if not img_path.exists():
            continue

        img_np = matplotlib.image.imread(str(img_path))
        if img_np.dtype != "uint8":
            img_np = (img_np * 255).astype("uint8")
        image = torch.from_numpy(img_np.copy()).to(device)

        with open(jf) as f:
            meta = json.load(f)

        for ann in meta["annotations"]:
            gt_mask = rle_decode_to_tensor(ann["segmentation"]).to(device)
            point = ann["point_coords"][0]
            fov_center = torch.tensor([int(point[0]), int(point[1])], device=device)

            with torch.no_grad():
                masks, pred_ious = predictor.get_prediction(image, fov_center)
            # masks: (K, N, P, P) logits, pred_ious: (K,)

            # Pick mask with highest predicted IoU
            best_k = pred_ious.argmax().item()
            best_mask_logits = masks[best_k]  # (N, P, P)

            # Reconstruct full-res mask in crop space
            # NOTE: This is an approximation — we compare in foveated visualization space.
            # For a true comparison we'd need to map back to the original image coordinates.
            # The foveated visualization gives a 1280×1280 view; GT mask is full-res.
            # We use the center crop of the GT for a rough eval.
            from segment_this_thing.utils import get_crop_bounds, get_centered_crop
            H, W = image.shape[:2]
            crop_bounds = get_crop_bounds(fov_center.float(), foveator.get_pattern_bounds_size())

            # Reconstruct predicted mask in foveated space
            pred_binary = reconstruct_full_mask(best_mask_logits, foveator)  # (1280, 1280) bool

            # Crop GT mask to the same 1280×1280 region
            gt_3ch = gt_mask.unsqueeze(-1).expand(-1, -1, 3).byte() * 255
            gt_crop_3ch = get_centered_crop(gt_3ch.cpu(), crop_bounds.cpu())
            gt_crop = gt_crop_3ch[:, :, 0].to(device).bool()  # (1280, 1280) bool

            iou = compute_iou_full_res(pred_binary.to(device), gt_crop)
            all_ious.append(iou)

        n_images += 1
        if n_images % 10 == 0:
            mean_iou = sum(all_ious) / len(all_ious)
            print(f"  {n_images} images, {len(all_ious)} masks, mIoU={mean_iou:.4f}")

    return sum(all_ious) / len(all_ious) if all_ious else 0.0


def evaluate_on_coco(
    images_dir: str,
    ann_file: str,
    model,
    foveator,
    imagenet_mean,
    imagenet_std,
    device: torch.device,
    max_images: int = None,
    return_oracle: bool = False,
) -> float:
    """
    SAM-protocol evaluation on COCO val2017 using log-rectilinear foveation.

    For each GT mask: click = pixel furthest from boundary (distance transform),
    run model on the 1280×1280 crop, pick best mask by pred_iou head output,
    unwarp to crop space, compare against GT cropped to same region.

    Returns mean IoU across all instances.
    """
    import numpy as np
    import matplotlib.image
    from scipy.ndimage import distance_transform_edt
    from pycocotools.coco import COCO
    from segment_this_thing.utils import get_crop_bounds, get_centered_crop
    from datasets import batch_foveate_images

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    if max_images is not None:
        img_ids = img_ids[:max_images]

    all_ious = []
    oracle_ious = []
    n_images = 0

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
        image = torch.from_numpy(img_np.copy()).to(device)  # (H, W, 3) uint8

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            gt_mask_np = coco.annToMask(ann).astype(bool)
            if gt_mask_np.sum() == 0:
                continue

            dist = distance_transform_edt(gt_mask_np)
            fy, fx = np.unravel_index(dist.argmax(), dist.shape)
            fov_center = torch.tensor([int(fx), int(fy)], device=device)

            crop_bounds = get_crop_bounds(
                fov_center.float(), foveator.get_pattern_bounds_size()
            ).to(device)
            crop_hwc = get_centered_crop(image.cpu(), crop_bounds.cpu()).to(device)  # (1280,1280,3)

            img_wh = torch.tensor([image.shape[1], image.shape[0]], device=device)
            valid_mask = foveator.get_in_bounds_tokens(img_wh, crop_bounds).unsqueeze(0)  # (1, N)

            tokens_u8 = batch_foveate_images(foveator, crop_hwc.unsqueeze(0))   # (1, N, 3, P, P)
            tokens = (tokens_u8 / 255.0 - imagenet_mean) / imagenet_std

            with torch.no_grad():
                pred_masks, pred_ious = model(tokens, valid_mask)
            # pred_masks: (1, K, N, P, P)  pred_ious: (1, K)

            gt_mask = torch.from_numpy(gt_mask_np).to(device)
            gt_3ch = gt_mask.unsqueeze(-1).expand(-1, -1, 3).byte() * 255
            gt_crop = get_centered_crop(gt_3ch.cpu(), crop_bounds.cpu())[:, :, 0].to(device).bool()

            K = pred_masks.shape[1]

            def _crop_iou(k):
                logits_k = pred_masks[0, k].unsqueeze(0).float()
                bin_k = foveator.unwarp_to_crop(logits_k.sigmoid())[0] > 0.5
                inter = (bin_k & gt_crop).sum().item()
                union = (bin_k | gt_crop).sum().item()
                return inter / (union + 1e-6) if union > 0 else 1.0

            # Model-selected: pick by pred_iou head
            best_k = pred_ious[0].argmax().item()
            all_ious.append(_crop_iou(best_k))

            # Oracle-selected: pick the mask with highest actual crop IoU
            if return_oracle:
                oracle_ious.append(max(_crop_iou(k) for k in range(K)))

        n_images += 1
        if n_images % 10 == 0:
            msg = (f"  eval: {n_images} images, {len(all_ious)} masks,"
                   f" mIoU={sum(all_ious)/len(all_ious):.4f}")
            if return_oracle and oracle_ious:
                msg += f"  oracle={sum(oracle_ious)/len(oracle_ious):.4f}"
            print(msg, flush=True)

    miou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    if return_oracle:
        oracle = sum(oracle_ious) / len(oracle_ious) if oracle_ious else 0.0
        return miou, oracle
    return miou


def main():
    parser = argparse.ArgumentParser("STT Evaluation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model-size", default="b", choices=["b", "l", "h"])
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    token_size = 16
    foveator = Foveator(
        token_size=token_size, strides=[1, 2, 4, 6, 8], grid_sizes=[4, 4, 6, 8, 10]
    ).to(device)

    model = MODEL_BUILDERS[args.model_size](
        token_size=token_size, num_tokens=foveator.get_num_tokens()
    )
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=True)
    # Support both full model state dicts and "model" key
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    predictor = SegmentThisThingPredictor(model, foveator)

    print(f"Evaluating on {args.data_root}...")
    mean_iou = evaluate_on_dataset(
        args.data_root, predictor, foveator, device, args.max_images
    )
    print(f"\nFinal mIoU: {mean_iou:.4f}")


if __name__ == "__main__":
    main()
