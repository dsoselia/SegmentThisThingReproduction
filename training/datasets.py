# Copyright (c) Meta Platforms, Inc. and affiliates.
# SA-1B dataset for Segment This Thing training.
#
# Paper protocol:
#   - Foveation center is sampled UNIFORMLY WITHIN the target segment
#     (not the "furthest from boundary" point used for evaluation).
#   - 256 pixel margin from image boundary for the foveation center.
#   - Up to 16 foveated views per image per training step.
#   - No full-resolution reconstruction — foveated crop only.

import io
import json
import os
import pickle
import random
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# RLE decode
# ──────────────────────────────────────────────────────────────────────────────

def rle_decode(rle: dict) -> np.ndarray:
    """Decode COCO RLE to (H, W) uint8 numpy array."""
    from pycocotools.mask import decode as coco_decode
    return coco_decode(rle)  # (H, W) uint8


# ──────────────────────────────────────────────────────────────────────────────
# Random center selection (paper: uniform within mask, 256 px margin)
# ──────────────────────────────────────────────────────────────────────────────

def sample_center_in_mask(
    mask: np.ndarray,
    image_h: int,
    image_w: int,
    margin: int = 256,
) -> Tuple[int, int]:
    """
    Sample a foveation center uniformly within `mask`, respecting `margin`.

    Returns (cx, cy) in image pixel coordinates (x=col, y=row).
    Falls back to the unconstrained mask centroid if no valid pixel exists.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        # Empty mask — return image center (shouldn't happen in practice)
        return image_w // 2, image_h // 2

    valid = (
        (xs >= margin)
        & (xs < image_w - margin)
        & (ys >= margin)
        & (ys < image_h - margin)
    )

    if valid.any():
        idx = np.random.randint(0, int(valid.sum()))
        return int(xs[valid][idx]), int(ys[valid][idx])

    # Fall back: clamp centroid to margin band
    cx = int(np.clip(xs.mean(), margin, image_w - margin))
    cy = int(np.clip(ys.mean(), margin, image_h - margin))
    return cx, cy


# ──────────────────────────────────────────────────────────────────────────────
# Centered crop with ImageNet-mean padding (matches predictor.py)
# ──────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN_U8 = np.array([round(0.485 * 255), round(0.456 * 255), round(0.406 * 255)],
                              dtype=np.uint8)


def centered_crop_np(
    arr: np.ndarray,
    cx: int,
    cy: int,
    crop_size: int,
    fill_value=None,
) -> np.ndarray:
    """
    Extract a `crop_size` × `crop_size` crop of `arr` centered at (cx, cy).
    Regions outside the array are padded with `fill_value`.

    arr: (H, W) or (H, W, C)
    Returns: (crop_size, crop_size) or (crop_size, crop_size, C)
    """
    H, W = arr.shape[:2]
    half = crop_size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + crop_size, y0 + crop_size

    # Source indices (clamped)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, W), min(y1, H)
    # Destination indices
    dx0, dy0 = sx0 - x0, sy0 - y0
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)

    if fill_value is None:
        fill_value = _IMAGENET_MEAN_U8 if arr.ndim == 3 else 0

    if arr.ndim == 3:
        out = np.full((crop_size, crop_size, arr.shape[2]), fill_value, dtype=arr.dtype)
    else:
        out = np.full((crop_size, crop_size), fill_value, dtype=arr.dtype)

    out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Valid token mask (which tokens overlap the image, not padding)
# ──────────────────────────────────────────────────────────────────────────────

def compute_valid_token_mask(
    foveator,
    image_w: int,
    image_h: int,
    cx: int,
    cy: int,
) -> torch.Tensor:
    """Compute (N,) bool valid-token mask via foveator.get_in_bounds_tokens."""
    from segment_this_thing.utils import get_crop_bounds
    center = torch.tensor([cx, cy], dtype=torch.float32)
    crop_bounds = get_crop_bounds(center, foveator.get_pattern_bounds_size())
    img_size = torch.tensor([image_w, image_h])
    return foveator.get_in_bounds_tokens(img_size, crop_bounds)  # (N,) bool


# ──────────────────────────────────────────────────────────────────────────────
# Multi-foveation dataset (paper: up to 16 foveated views per image)
# ──────────────────────────────────────────────────────────────────────────────

class SA1BMultiFovDataset(torch.utils.data.Dataset):
    """
    SA-1B dataset that returns up to `max_fov` foveated crops per image.

    Each item is a list of (image_crop_uint8, mask_crop_float32, valid_mask_bool) triples,
    one per sampled foveation.  Foveation centers are sampled uniformly within
    randomly selected segments, with a 256-pixel margin from image boundaries,
    as described in the paper.

    The foveated TOKEN extraction is deferred to the training loop (GPU).
    Workers only do: JPEG load + RLE decode + crop.

    Args:
        data_root: directory of extracted SA-1B files (*.jpg + *.json pairs).
        max_fov:   Maximum foveated views per image (paper: 16 for seg training,
                   2 for MAE pre-training).
        margin:    Pixel margin from image boundary (paper: 256).
        foveator:  A Foveator instance (CPU) for computing valid token masks.
    """

    def __init__(
        self,
        data_root: str,
        foveator,
        max_fov: int = 16,
        margin: int = 256,
    ):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "repo"))

        self.max_fov = max_fov
        self.margin = margin
        self.foveator = foveator
        self.crop_size = foveator.get_pattern_bounds_size()  # 1280

        data_root = Path(data_root)

        # Cache the file list to avoid slow NFS directory scans on every init
        cache_path = data_root / ".stt_index.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                self.items = pickle.load(f)
        else:
            self.items: List[Tuple[Path, Path]] = []
            for jf in sorted(data_root.glob("*.json")):
                img_path = jf.with_suffix(".jpg")
                if img_path.exists():
                    self.items.append((str(jf), str(img_path)))
            with open(cache_path, "wb") as f:
                pickle.dump(self.items, f)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        jf, img_path = self.items[idx]

        # Load image (PIL is 2-3× faster than matplotlib for JPEG)
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.asarray(img_pil)  # (H, W, 3) uint8
        H, W = img_np.shape[:2]

        # Load annotations
        with open(jf) as f:
            meta = json.load(f)
        anns = meta["annotations"]
        if not anns:
            # Return a single zero view so the collate doesn't crash
            zero_img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            zero_mask = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
            zero_valid = torch.ones(self.foveator.get_num_tokens(), dtype=torch.bool)
            return [(torch.from_numpy(zero_img), torch.from_numpy(zero_mask), zero_valid)]

        # Sample up to max_fov annotations (without replacement if possible)
        n_pick = min(self.max_fov, len(anns))
        picked = random.sample(anns, n_pick)

        views = []
        for ann in picked:
            mask_np = rle_decode(ann["segmentation"])  # (H, W) uint8

            # Sample random center within mask & margin
            cx, cy = sample_center_in_mask(mask_np, H, W, self.margin)

            # Crop image and mask
            img_crop = centered_crop_np(img_np, cx, cy, self.crop_size)          # (1280,1280,3) uint8
            mask_crop = centered_crop_np(mask_np.astype(np.float32), cx, cy, self.crop_size, fill_value=0.0)  # (1280,1280) float32

            # Valid token mask (which tokens overlap the actual image, not padding)
            valid = compute_valid_token_mask(self.foveator, W, H, cx, cy)  # (N,) bool

            views.append((
                torch.from_numpy(img_crop),      # (1280, 1280, 3) uint8
                torch.from_numpy(mask_crop),     # (1280, 1280) float32 ∈ {0,1}
                valid,                           # (N,) bool
            ))

        return views  # list of up to 16 (img_crop, mask_crop, valid) triples


class SA1BTarMultiFovDataset(torch.utils.data.Dataset):
    """
    Same as SA1BMultiFovDataset but reads directly from .tar shards.

    Uses byte-offset indexing so each __getitem__ does two O(1) seeks instead
    of scanning the full tar file — critical for NFS performance.
    Index is cached to disk (one .pkl per tar) to avoid re-scanning on restart.
    """

    def __init__(self, tar_files: List[str], foveator, max_fov: int = 16, margin: int = 256):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "repo"))

        self.max_fov = max_fov
        self.margin = margin
        self.foveator = foveator
        self.crop_size = foveator.get_pattern_bounds_size()

        # items: list of dicts with tar_path and either byte offsets (uncompressed)
        # or member names (compressed — gzip/bzip2 can't seek by offset).
        self.items = []
        for tf_path in tar_files:
            cache = tf_path + ".idx.pkl"
            if os.path.exists(cache):
                with open(cache, "rb") as f:
                    self.items.extend(pickle.load(f))
                continue
            try:
                shard_items = []
                # r:* handles gzip/bzip2/xz/uncompressed transparently
                with tarfile.open(tf_path, "r:*") as tf:
                    compressed = tf.fileobj.__class__.__name__ != "ExFileObject"
                    # Detect gzip: stream-mode fileobj means we can't seek
                    try:
                        tf.fileobj.tell()
                        seekable = True
                    except Exception:
                        seekable = False

                    members = {m.name: m for m in tf.getmembers() if m.isfile()}
                    for name, m in members.items():
                        if name.endswith(".jpg"):
                            base = name[:-4]
                            json_name = base + ".json"
                            if json_name in members:
                                jm = members[json_name]
                                if seekable and not compressed:
                                    # Uncompressed tar: store byte offsets for O(1) seeks
                                    shard_items.append({
                                        "tar": tf_path, "mode": "offset",
                                        "jpg_off": m.offset_data, "jpg_sz": m.size,
                                        "json_off": jm.offset_data, "json_sz": jm.size,
                                    })
                                else:
                                    # Compressed tar: store names, open with r:* per access
                                    shard_items.append({
                                        "tar": tf_path, "mode": "name",
                                        "jpg": name, "json": json_name,
                                    })
                with open(cache, "wb") as f:
                    pickle.dump(shard_items, f)
                self.items.extend(shard_items)
            except Exception as e:
                print(f"Warning: {tf_path}: {e}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        if item["mode"] == "offset":
            # Uncompressed tar: O(1) byte-offset reads
            with open(item["tar"], "rb") as f:
                f.seek(item["jpg_off"]); jpg_bytes = f.read(item["jpg_sz"])
                f.seek(item["json_off"]); json_bytes = f.read(item["json_sz"])
            meta = json.loads(json_bytes)
        else:
            # Compressed tar: decompress and extract named members
            with tarfile.open(item["tar"], "r:*") as tf:
                jpg_bytes = tf.extractfile(item["jpg"]).read()
                json_bytes = tf.extractfile(item["json"]).read()
            meta = json.loads(json_bytes)

        img_pil = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
        img_np = np.asarray(img_pil)  # (H, W, 3) uint8
        H, W = img_np.shape[:2]

        anns = meta.get("annotations", [])
        if not anns:
            zero_img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            zero_mask = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
            zero_valid = torch.ones(self.foveator.get_num_tokens(), dtype=torch.bool)
            return [(torch.from_numpy(zero_img), torch.from_numpy(zero_mask), zero_valid)]

        n_pick = min(self.max_fov, len(anns))
        picked = random.sample(anns, n_pick)

        views = []
        for ann in picked:
            mask_np = rle_decode(ann["segmentation"])
            cx, cy = sample_center_in_mask(mask_np, H, W, self.margin)
            img_crop = centered_crop_np(img_np, cx, cy, self.crop_size)
            mask_crop = centered_crop_np(mask_np.astype(np.float32), cx, cy, self.crop_size, 0.0)
            valid = compute_valid_token_mask(self.foveator, W, H, cx, cy)
            views.append((torch.from_numpy(img_crop), torch.from_numpy(mask_crop), valid))

        return views


def multi_fov_collate(batch):
    """
    Collate a batch of variable-length view lists into flat tensors.

    Returns:
        img_crops:  (B_total, 1280, 1280, 3) uint8  — to be moved to GPU for foveation
        mask_crops: (B_total, 1280, 1280) float32   — ground-truth mask crops
                    OR a scalar zero tensor for MAE (where masks are dummy (1,1) arrays)
        valid_masks:(B_total, N) bool                — valid token masks
    """
    img_list, mask_list, valid_list = [], [], []
    for views in batch:
        for img_c, mask_c, valid in views:
            img_list.append(img_c)
            mask_list.append(mask_c)
            valid_list.append(valid)
    # MAE datasets return (1,1) dummy masks to avoid the 3 GB/step zero-tensor cost.
    # Detect this and skip stacking — the MAE training loop ignores _mask_crops anyway.
    if mask_list[0].shape == (1, 1):
        mask_out = torch.zeros(1)   # lightweight placeholder
    else:
        mask_out = torch.stack(mask_list)  # (B, 1280, 1280) float32
    return (
        torch.stack(img_list),    # (B, 1280, 1280, 3) uint8
        mask_out,
        torch.stack(valid_list),  # (B, N) bool
    )


# ──────────────────────────────────────────────────────────────────────────────
# GPU-side batched foveation (moves the bottleneck from CPU workers to GPU)
# ──────────────────────────────────────────────────────────────────────────────

def batch_foveate_images(
    foveator,
    images_hwc: torch.Tensor,   # (B, H, W, 3) uint8 on GPU
) -> torch.Tensor:
    """
    Batched GPU foveation of image crops.

    Supports both the concentric-ring Foveator and LogRectilinearFoveator.
    Dispatch is based on whether the foveator has precomputed `lower_coords`.

    Returns: (B, N, 3, ts, ts) float32 — normalized pixel averages per token.
    Values are in [0, 255] (NOT yet ImageNet-normalized).
    """
    B, H, W, C = images_hwc.shape
    ts = foveator.token_size
    N = foveator.get_num_tokens()
    device = images_hwc.device

    # Build integral image (common to both paths)
    img_f = images_hwc.permute(0, 3, 1, 2).float()  # (B, 3, H, W)
    padded = F.pad(img_f, (1, 0, 1, 0), value=0.0)  # (B, 3, H+1, W+1)
    integral = padded.cumsum(dim=3).cumsum(dim=2)     # (B, 3, H+1, W+1)
    W1 = W + 1

    def flat(y, x):
        return (y * W1 + x).view(1, 1, -1).expand(B, C, -1)  # (B, C, N*ts*ts)

    int_flat = integral.flatten(2)  # (B, C, (H+1)*(W+1))

    if hasattr(foveator, 'lower_coords'):
        # ── Log-rectilinear path: per-cell coords precomputed ─────────────────
        lx = foveator.lower_coords[..., 0].long()   # (N, ts, ts)
        ly = foveator.lower_coords[..., 1].long()
        ux = foveator.upper_coords[..., 0].long()
        uy = foveator.upper_coords[..., 1].long()
        areas = (
            int_flat.gather(2, flat(uy, ux))
            - int_flat.gather(2, flat(uy, lx))
            - int_flat.gather(2, flat(ly, ux))
            + int_flat.gather(2, flat(ly, lx))
        )  # (B, C, N*ts*ts)
        areas = areas.view(B, C, N, ts, ts)
        # Clamp: float32 cumsum precision errors can push values slightly outside [0,255]
        tokens = (areas / foveator.pixel_areas.view(1, 1, N, ts, ts)).clamp(0.0, 255.0)
    else:
        # ── Original concentric-ring path: per-token uniform stride ───────────
        from segment_this_thing.foveation import generate_grid_coords_2d
        grid = generate_grid_coords_2d(ts).to(device)           # (ts, ts, 2)
        lower = (
            foveator.token_corner_indices.view(N, 1, 1, 2)
            + foveator.token_strides.view(N, 1, 1, 1) * grid.unsqueeze(0)
        )  # (N, ts, ts, 2)
        upper = lower + foveator.token_strides.view(N, 1, 1, 1)
        lx, ly = lower[..., 0].long(), lower[..., 1].long()
        ux, uy = upper[..., 0].long(), upper[..., 1].long()
        areas = (
            int_flat.gather(2, flat(uy, ux))
            - int_flat.gather(2, flat(uy, lx))
            - int_flat.gather(2, flat(ly, ux))
            + int_flat.gather(2, flat(ly, lx))
        )  # (B, C, N*ts*ts)
        areas = areas.view(B, C, N, ts, ts)
        tokens = areas / foveator.token_strides.square().view(1, 1, N, 1, 1).float()

    return tokens.permute(0, 2, 1, 3, 4)  # (B, N, C, ts, ts)  values in [0,255]


def batch_foveate_masks(
    foveator,
    masks_hw: torch.Tensor,     # (B, H, W) float32 ∈ [0,1] on GPU
) -> torch.Tensor:
    """
    Batched GPU foveation of binary mask crops.

    Supports both the concentric-ring Foveator and LogRectilinearFoveator.

    Returns: (B, N, ts, ts) float32 — proportion of masked pixels per token cell.
    """
    B, H, W = masks_hw.shape
    ts = foveator.token_size
    N = foveator.get_num_tokens()
    device = masks_hw.device

    padded = F.pad(masks_hw.unsqueeze(1), (1, 0, 1, 0), value=0.0)  # (B, 1, H+1, W+1)
    integral = padded.cumsum(dim=3).cumsum(dim=2)                     # (B, 1, H+1, W+1)
    W1 = W + 1

    def flat(y, x):
        return (y * W1 + x).view(1, 1, -1).expand(B, 1, -1)

    int_flat = integral.flatten(2)

    if hasattr(foveator, 'lower_coords'):
        # ── Log-rectilinear path ──────────────────────────────────────────────
        lx = foveator.lower_coords[..., 0].long()
        ly = foveator.lower_coords[..., 1].long()
        ux = foveator.upper_coords[..., 0].long()
        uy = foveator.upper_coords[..., 1].long()
        areas = (
            int_flat.gather(2, flat(uy, ux))
            - int_flat.gather(2, flat(uy, lx))
            - int_flat.gather(2, flat(ly, ux))
            + int_flat.gather(2, flat(ly, lx))
        ).view(B, N, ts, ts)
        proportions = (areas / foveator.pixel_areas.view(1, N, ts, ts)).clamp(0.0, 1.0)
    else:
        # ── Original concentric-ring path ─────────────────────────────────────
        from segment_this_thing.foveation import generate_grid_coords_2d
        grid = generate_grid_coords_2d(ts).to(device)
        lower = (
            foveator.token_corner_indices.view(N, 1, 1, 2)
            + foveator.token_strides.view(N, 1, 1, 1) * grid.unsqueeze(0)
        )
        upper = lower + foveator.token_strides.view(N, 1, 1, 1)
        lx, ly = lower[..., 0].long(), lower[..., 1].long()
        ux, uy = upper[..., 0].long(), upper[..., 1].long()
        areas = (
            int_flat.gather(2, flat(uy, ux))
            - int_flat.gather(2, flat(uy, lx))
            - int_flat.gather(2, flat(ly, ux))
            + int_flat.gather(2, flat(ly, lx))
        ).view(B, N, ts, ts)
        proportions = areas / foveator.token_strides.square().view(1, N, 1, 1).float()

    return proportions  # (B, N, ts, ts)


# ──────────────────────────────────────────────────────────────────────────────
# COCO dataset (MAE pre-training + segmentation fine-tuning)
# ──────────────────────────────────────────────────────────────────────────────

def _polygon_to_mask(segmentation, h: int, w: int) -> np.ndarray:
    """Convert COCO polygon or RLE segmentation to (H, W) uint8 mask."""
    from pycocotools.mask import frPyObjects, merge, decode
    if isinstance(segmentation, dict):
        # Already RLE (crowd annotation)
        rle = segmentation
        if isinstance(rle["counts"], list):
            rle = frPyObjects([rle], h, w)[0]
    else:
        # Polygon list — convert to RLE then decode
        rles = frPyObjects(segmentation, h, w)
        rle  = merge(rles)
    return decode(rle)  # (H, W) uint8


class COCOMultiFovDataset(torch.utils.data.Dataset):
    """
    COCO dataset for STT training, returning up to `max_fov` foveated views per image.

    For MAE pre-training (`is_mae=True`):
      - No annotation file needed; samples random foveation centers within the image.
      - Returns zero mask crops (loss is image-reconstruction, not segmentation).

    For segmentation fine-tuning (`is_mae=False`):
      - Requires `ann_file` (instances_train2017.json).
      - Each view is centered on a randomly sampled instance mask pixel.
      - Returns binary mask crop (float32 ∈ {0,1}).

    Args:
        image_dir:  Path to COCO image directory (e.g. .../train2017/).
        foveator:   CPU Foveator instance.
        ann_file:   Path to instances_*.json. Required if is_mae=False.
        max_fov:    Max foveated views per image (2 for MAE, 16 for seg).
        is_mae:     If True, use random centers (no mask required).
        margin:     Pixel margin from image boundary (paper: 256).
    """

    def __init__(
        self,
        image_dir: str,
        foveator,
        ann_file: str = None,
        max_fov: int = 16,
        is_mae: bool = False,
        margin: int = 256,
    ):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "repo"))

        self.image_dir = Path(image_dir)
        self.foveator  = foveator
        self.max_fov   = max_fov
        self.is_mae    = is_mae
        self.margin    = margin
        self.crop_size = foveator.get_pattern_bounds_size()  # 1280

        # Cache directory: prefer local project dir over potentially read-only NFS
        _cache_dir = Path(__file__).parent.parent / "cache"
        _cache_dir.mkdir(exist_ok=True)

        if is_mae:
            # For MAE just need image paths — no annotations
            import hashlib
            _dir_hash = hashlib.md5(str(self.image_dir).encode()).hexdigest()[:8]
            cache = _cache_dir / f"coco_mae_{_dir_hash}.pkl"
            if cache.exists():
                with open(cache, "rb") as f:
                    self.items = pickle.load(f)
            else:
                self.items = sorted(str(p) for p in self.image_dir.glob("*.jpg"))
                with open(cache, "wb") as f:
                    pickle.dump(self.items, f)
        else:
            assert ann_file, "ann_file required for segmentation training"
            import hashlib
            _ann_hash = hashlib.md5(ann_file.encode()).hexdigest()[:8]
            cache = _cache_dir / f"coco_seg_{_ann_hash}.pkl"
            if cache.exists():
                with open(cache, "rb") as f:
                    self.items = pickle.load(f)
            else:
                print(f"Building COCO index from {ann_file} …")
                with open(ann_file) as f:
                    data = json.load(f)
                # Map image_id → file_name, height, width
                id2img = {img["id"]: img for img in data["images"]}
                # Group annotations by image_id, skip crowd
                from collections import defaultdict
                img2anns = defaultdict(list)
                for ann in data["annotations"]:
                    if ann.get("iscrowd", 0):
                        continue
                    if ann.get("area", 0) < 100:   # skip tiny instances
                        continue
                    img2anns[ann["image_id"]].append(ann)
                # Build item list: (img_path, img_h, img_w, [ann, ...])
                self.items = []
                for img_id, anns in img2anns.items():
                    meta = id2img[img_id]
                    img_path = str(self.image_dir / meta["file_name"])
                    if os.path.exists(img_path):
                        self.items.append((
                            img_path,
                            meta["height"],
                            meta["width"],
                            anns,
                        ))
                with open(cache, "wb") as f:
                    pickle.dump(self.items, f, protocol=4)
                print(f"  → {len(self.items)} images with annotations")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        if self.is_mae:
            return self._getitem_mae(self.items[idx])
        else:
            img_path, H, W, anns = self.items[idx]
            return self._getitem_seg(img_path, H, W, anns)

    def _getitem_mae(self, img_path: str):
        img_pil = Image.open(img_path).convert("RGB")
        img_np  = np.asarray(img_pil)  # (H, W, 3) uint8
        H, W    = img_np.shape[:2]

        views = []
        for _ in range(self.max_fov):
            # Random center within margin
            lo_x = max(self.margin, W // 2)
            hi_x = max(lo_x + 1, W - self.margin)
            lo_y = max(self.margin, H // 2)
            hi_y = max(lo_y + 1, H - self.margin)
            cx = random.randint(lo_x, hi_x - 1)
            cy = random.randint(lo_y, hi_y - 1)

            img_crop = centered_crop_np(img_np, cx, cy, self.crop_size)
            # Return a 1×1 dummy mask — MAE training ignores masks entirely.
            # Avoids allocating a (crop_size × crop_size) float32 zero array per view,
            # which wastes ~3 GB/step in CPU+GPU RAM at large batch sizes.
            mask_crop = np.zeros((1, 1), dtype=np.float32)
            valid     = compute_valid_token_mask(self.foveator, W, H, cx, cy)
            views.append((
                torch.from_numpy(img_crop),
                torch.from_numpy(mask_crop),
                valid,
            ))
        return views

    def _getitem_seg(self, img_path: str, H: int, W: int, anns: list):
        img_pil = Image.open(img_path).convert("RGB")
        img_np  = np.asarray(img_pil)

        n_pick = min(self.max_fov, len(anns))
        picked = random.sample(anns, n_pick)

        views = []
        for ann in picked:
            mask_np = _polygon_to_mask(ann["segmentation"], H, W)  # (H, W) uint8
            cx, cy  = sample_center_in_mask(mask_np, H, W, self.margin)
            img_crop  = centered_crop_np(img_np, cx, cy, self.crop_size)
            mask_crop = centered_crop_np(
                mask_np.astype(np.float32), cx, cy, self.crop_size, fill_value=0.0
            )
            valid = compute_valid_token_mask(self.foveator, W, H, cx, cy)
            views.append((
                torch.from_numpy(img_crop),
                torch.from_numpy(mask_crop),
                valid,
            ))

        if not views:
            z_img  = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            z_mask = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
            valid  = torch.ones(self.foveator.get_num_tokens(), dtype=torch.bool)
            views  = [(torch.from_numpy(z_img), torch.from_numpy(z_mask), valid)]

        return views
