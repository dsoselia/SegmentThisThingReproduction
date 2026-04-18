"""
Log-Rectilinear Foveated Tokenization for Segment This Thing.

Replaces the concentric-ring Foveator with a continuous log-rectilinear warp
(Li et al. 2021: "A Log-Rectilinear Transformation for Foveated 360-degree
Video Streaming").

Forward mapping (separable, applied independently to x and y):
    λ = crop_half / (e − 1)
    Δx = max(|Δu|, λ * (exp((|Δu| / buf_half)^4) − 1)) * sign(Δu)

Properties:
  - At Δu = 0:          Δx = 0          (center maps to center)
  - |Δu| < ~0.63*buf:   Δx ≈ Δu        (identity / 1-to-1 region)
  - |Δu| = buf_half:    Δx = crop_half  (buffer edge → crop edge)

Design choices (fixed for this experiment):
  - 13×13 = 169 tokens  (vs 172 in concentric-ring baseline)
  - Token size: 16×16 pixels
  - Buffer size: 208×208 pixels (13 tokens × 16 px)
  - Crop size:  1280×1280 pixels (unchanged from baseline)
"""

import math
import numpy as np
import torch
import torch.nn as nn

# ── Constants ──────────────────────────────────────────────────────────────────
_GRID_SIZE  = 13        # G: tokens per side
_TOKEN_SIZE = 16        # ts: pixels per token side
_BUF_SIZE   = _GRID_SIZE * _TOKEN_SIZE   # 208 pixels
_BUF_HALF   = _BUF_SIZE / 2.0           # 104.0
_CROP_SIZE  = 1280
_CROP_HALF  = _CROP_SIZE / 2.0          # 640.0
_LAMBDA     = _CROP_HALF / (math.e - 1) # ≈ 372.47


def _lr_map(du: np.ndarray) -> np.ndarray:
    """
    Log-rectilinear forward mapping: buffer offset → crop coordinate.

    du: signed offset from buffer center (scalar or ndarray)
    Returns absolute crop coordinate in [0, _CROP_SIZE].
    """
    du = np.asarray(du, dtype=np.float64)
    ad = np.abs(du)
    exp_term = _LAMBDA * (np.exp((ad / _BUF_HALF) ** 4) - 1.0)
    dx = np.where(ad < 1e-9, 0.0, np.maximum(ad, exp_term) * np.sign(du))
    return dx + _CROP_HALF


def _precompute_coords():
    """
    Precompute per-cell source pixel coordinates for all 169 × 16 × 16 cells.

    The mapping is separable:
      - x-coordinate of cell (col*ts + px) depends only on col and px
      - y-coordinate of cell (row*ts + py) depends only on row and py

    Returns:
        lower_coords (169, 16, 16, 2) int32  — (lx, ly) crop coords of cell lower-left
        upper_coords (169, 16, 16, 2) int32  — (ux, uy) crop coords of cell upper-right
        pixel_areas  (169, 16, 16)    float32 — (ux-lx)*(uy-ly), clipped to ≥ 1
        token_lower  (169, 2)         int32  — per-token bounding box lower
        token_upper  (169, 2)         int32  — per-token bounding box upper
    """
    G  = _GRID_SIZE
    ts = _TOKEN_SIZE
    N  = G * G  # 169

    # Buffer pixel boundaries → crop x/y coordinates (209 values: [0 .. _BUF_SIZE])
    # Each buffer pixel u covers crop interval [x_src[u], x_src[u+1])
    buf_boundaries = np.arange(_BUF_SIZE + 1, dtype=np.float64)  # (209,)
    x_src = _lr_map(buf_boundaries - _BUF_HALF)   # (209,) in [0, 1280], monotone
    y_src = x_src.copy()                           # symmetric (square crop + buffer)

    # Integer floor to get integral-image indices
    x_src_i = np.floor(x_src).astype(np.int32)   # (209,)
    y_src_i  = np.floor(y_src).astype(np.int32)

    # For column c, pixel px within token: absolute buffer pixel = c*ts + px
    # lx = x_src_i[c*ts + px],  ux = x_src_i[c*ts + px + 1]
    # Shape: (G, ts) — col × px
    col_px = np.arange(G)[:, None] * ts + np.arange(ts)[None, :]  # (G, ts)
    lx_2d  = x_src_i[col_px]       # (G, ts)
    ux_2d  = x_src_i[col_px + 1]   # (G, ts)

    # For row r, pixel py: absolute buffer pixel = r*ts + py
    row_py = np.arange(G)[:, None] * ts + np.arange(ts)[None, :]  # (G, ts)
    ly_2d  = y_src_i[row_py]       # (G, ts)
    uy_2d  = y_src_i[row_py + 1]   # (G, ts)

    # Broadcast into (G_row, G_col, ts_py, ts_px) using separability
    # Token n = row*G + col → row = n//G, col = n%G
    # lower_coords[n, py, px, 0] = lx_2d[col, px]  (x depends on col, px)
    # lower_coords[n, py, px, 1] = ly_2d[row, py]  (y depends on row, py)

    # (G, G, ts, ts) arrays via broadcasting
    lx = np.broadcast_to(lx_2d[None, :, None, :], (G, G, ts, ts)).copy()  # [row,col,py,px]
    ux = np.broadcast_to(ux_2d[None, :, None, :], (G, G, ts, ts)).copy()
    ly = np.broadcast_to(ly_2d[:, None, :, None], (G, G, ts, ts)).copy()
    uy = np.broadcast_to(uy_2d[:, None, :, None], (G, G, ts, ts)).copy()

    # Reshape to (N, ts, ts) — token-major, then py, px
    lx = lx.reshape(N, ts, ts)
    ux = ux.reshape(N, ts, ts)
    ly = ly.reshape(N, ts, ts)
    uy = uy.reshape(N, ts, ts)

    # Stack into (N, ts, ts, 2)
    lower_coords = np.stack([lx, ly], axis=-1).astype(np.int32)  # (N, ts, ts, 2)
    upper_coords = np.stack([ux, uy], axis=-1).astype(np.int32)

    # Pixel areas: (ux - lx) * (uy - ly), clipped to ≥ 1
    areas = ((ux - lx) * (uy - ly)).astype(np.float32)
    pixel_areas = np.clip(areas, 1.0, None)  # (N, ts, ts)

    # Per-token bounding box: min/max over all cells
    token_lower = lower_coords.reshape(N, ts * ts, 2).min(axis=1)  # (N, 2)
    token_upper = upper_coords.reshape(N, ts * ts, 2).max(axis=1)  # (N, 2)

    return lower_coords, upper_coords, pixel_areas, token_lower, token_upper


# ──────────────────────────────────────────────────────────────────────────────

class LogRectilinearFoveator(nn.Module):
    """
    Log-Rectilinear foveated tokenizer for Segment This Thing.

    Produces a 13×13 = 169 regular token grid where each token corresponds to
    a log-rectilinearly warped region of a 1280×1280 crop.  The interface
    mirrors the upstream `Foveator` class so it can be used as a drop-in
    replacement in training and dataset code.

    Key attributes (registered buffers, moved to GPU with .to(device)):
        lower_coords  (169, 16, 16, 2) int32   — per-cell (lx, ly)
        upper_coords  (169, 16, 16, 2) int32   — per-cell (ux, uy)
        pixel_areas   (169, 16, 16)    float32 — per-cell (ux-lx)*(uy-ly)
        _token_lower  (169, 2)         int32   — per-token bbox lower
        _token_upper  (169, 2)         int32   — per-token bbox upper
    """

    token_size: int = _TOKEN_SIZE

    def __init__(self):
        super().__init__()
        lc, uc, pa, tl, tu = _precompute_coords()
        self.register_buffer("lower_coords",  torch.from_numpy(lc))
        self.register_buffer("upper_coords",  torch.from_numpy(uc))
        self.register_buffer("pixel_areas",   torch.from_numpy(pa))
        self.register_buffer("_token_lower",  torch.from_numpy(tl))
        self.register_buffer("_token_upper",  torch.from_numpy(tu))

    # ── Foveator-compatible interface ─────────────────────────────────────────

    def get_pattern_bounds_size(self) -> int:
        """Size of the square crop required (pixels). Same as baseline: 1280."""
        return _CROP_SIZE

    def get_num_tokens(self) -> int:
        """Total number of tokens: 13 × 13 = 169."""
        return _GRID_SIZE * _GRID_SIZE

    def get_in_bounds_tokens(
        self,
        image_size: torch.Tensor,      # (2,) [img_w, img_h]
        crop_bounds: torch.Tensor,     # (2, 2) [[x1, y1], [x2, y2]]
        in_bounds_threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        Returns (169,) bool mask: True where a token's source region overlaps
        the actual image (not padding).

        Mirrors Foveator.get_in_bounds_tokens() signature.
        """
        fov_origin = crop_bounds[0].float()   # (2,) top-left of crop in image coords

        # Image region in crop-local coordinates
        img_lo = (-fov_origin).clamp(min=0.0)                            # (2,)
        img_hi = (image_size.float() - fov_origin).clamp(max=float(_CROP_SIZE))  # (2,)

        tok_lo = self._token_lower.float()   # (N, 2) in crop space
        tok_hi = self._token_upper.float()   # (N, 2)

        overlap = (
            torch.minimum(tok_hi, img_hi) - torch.maximum(tok_lo, img_lo)
        ).clamp(min=0.0).prod(dim=-1)   # (N,)

        bbox_area = (tok_hi - tok_lo).prod(dim=-1).clamp(min=1.0)   # (N,)

        return (overlap / bbox_area) > in_bounds_threshold   # (N,) bool

    def generate_foveated_visualization(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a 208×208 image from 169 tokens by tiling them in a
        regular 13×13 grid (the log-rectilinear buffer).

        Input:  (169, C, 16, 16)
        Output: (C, 208, 208)
        """
        N, C, P, _ = tokens.shape
        assert N == _GRID_SIZE * _GRID_SIZE, f"Expected {_GRID_SIZE**2} tokens, got {N}"
        # (G, G, C, ts, ts) → (C, G, ts, G, ts) → (C, G*ts, G*ts)
        return (
            tokens
            .unflatten(0, (_GRID_SIZE, _GRID_SIZE))   # (G, G, C, ts, ts)
            .permute(2, 0, 3, 1, 4)                   # (C, G, ts, G, ts)
            .flatten(3, 4)                             # (C, G, ts, G*ts)
            .flatten(1, 2)                             # (C, G*ts, G*ts)
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "LogRectilinearFoveator has no forward() — use batch_foveate_images() "
            "from datasets.py for GPU-side tokenization."
        )


# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test():
    """Verify correctness of the log-rectilinear coordinate precomputation."""
    print("Running LogRectilinearFoveator self-test …")
    fov = LogRectilinearFoveator()
    lc = fov.lower_coords   # (169, 16, 16, 2) int32
    uc = fov.upper_coords   # (169, 16, 16, 2) int32

    # ── Test 1: first token (0,0) lower-left cell maps to crop origin (0, 0) ──
    lx0 = lc[0, 0, 0, 0].item()
    ly0 = lc[0, 0, 0, 1].item()
    assert lx0 == 0 and ly0 == 0, f"Test 1 FAIL: token(0,0) cell(0,0) lower = ({lx0},{ly0}), expected (0,0)"
    print("  Test 1 PASS: first token lower-left maps to crop origin (0, 0)")

    # ── Test 2: last token (12,12) last cell maps to crop edge (1280, 1280) ──
    last_n = _GRID_SIZE * _GRID_SIZE - 1   # 168
    ux_last = uc[last_n, _TOKEN_SIZE - 1, _TOKEN_SIZE - 1, 0].item()
    uy_last = uc[last_n, _TOKEN_SIZE - 1, _TOKEN_SIZE - 1, 1].item()
    assert ux_last == _CROP_SIZE and uy_last == _CROP_SIZE, \
        f"Test 2 FAIL: last token upper = ({ux_last},{uy_last}), expected ({_CROP_SIZE},{_CROP_SIZE})"
    print(f"  Test 2 PASS: last token upper maps to crop edge ({_CROP_SIZE}, {_CROP_SIZE})")

    # ── Test 3: center token (6,6) center cell maps near crop center (640, 640) ──
    center_n = 6 * _GRID_SIZE + 6   # 84
    # Cell (8, 8) → buffer pixel (6*16+8, 6*16+8) = (104, 104) → du = (0, 0) → Δx = 0
    lx_c = lc[center_n, 8, 8, 0].item()
    ly_c = lc[center_n, 8, 8, 1].item()
    assert lx_c == int(_CROP_HALF), \
        f"Test 3 FAIL: center cell lx = {lx_c}, expected {int(_CROP_HALF)}"
    assert ly_c == int(_CROP_HALF), \
        f"Test 3 FAIL: center cell ly = {ly_c}, expected {int(_CROP_HALF)}"
    print(f"  Test 3 PASS: center cell lower maps to crop center ({lx_c}, {ly_c})")

    # ── Test 4: upper > lower everywhere ──────────────────────────────────────
    diff_x = (uc[..., 0] - lc[..., 0])
    diff_y = (uc[..., 1] - lc[..., 1])
    assert (diff_x > 0).all(), f"Test 4 FAIL: some ux == lx (min diff = {diff_x.min()})"
    assert (diff_y > 0).all(), f"Test 4 FAIL: some uy == ly (min diff = {diff_y.min()})"
    print("  Test 4 PASS: upper_coords > lower_coords everywhere")

    # ── Test 5: all pixel_areas > 0 ───────────────────────────────────────────
    assert (fov.pixel_areas > 0).all(), "Test 5 FAIL: some pixel_areas <= 0"
    print("  Test 5 PASS: all pixel_areas > 0")

    # ── Test 6: identity region (center-side tokens map 1-to-1) ───────────────
    # Token (6, 4): col=4 → buf x in [64, 80), du ∈ [-40, -24]
    # λ * (40/104)^4 = 372.47 * 0.0219 ≈ 8.1 << 40 → identity (Δx ≈ Δu)
    # So lx for col=4, px=0 should equal floor(lr_map(64 - 104)) = floor(640 - 40) = 600
    n_6_4 = 6 * _GRID_SIZE + 4   # row=6, col=4 → token 82
    lx_id = lc[n_6_4, 0, 0, 0].item()
    expected_id = int(math.floor(_lr_map(4 * _TOKEN_SIZE - _BUF_HALF)))  # floor(lr_map(-40))
    assert lx_id == expected_id, \
        f"Test 6 FAIL: identity region lx = {lx_id}, expected {expected_id}"
    print(f"  Test 6 PASS: identity-region token lx = {lx_id} (expected {expected_id})")

    # ── Test 7: monotonicity (lower_x is non-decreasing across cols) ──────────
    # Check that lc[col+1, 0, 0, 0] >= lc[col, 0, 0, 0] for all col
    # Token n for row=0: n = col
    lx_row0 = lc[:_GRID_SIZE, 0, 0, 0]   # (13,) — first cell of each token in row 0
    assert (lx_row0.diff() >= 0).all(), \
        f"Test 7 FAIL: lower_x not monotone in row 0: {lx_row0.tolist()}"
    print("  Test 7 PASS: lower_x is monotone non-decreasing across columns")

    # ── Test 8: get_num_tokens and get_pattern_bounds_size ────────────────────
    assert fov.get_num_tokens() == 169
    assert fov.get_pattern_bounds_size() == 1280
    print("  Test 8 PASS: get_num_tokens()=169, get_pattern_bounds_size()=1280")

    print("All tests passed.")


if __name__ == "__main__":
    _self_test()
