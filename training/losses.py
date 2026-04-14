# Copyright (c) Meta Platforms, Inc. and affiliates.
# Loss functions for Segment This Thing.
#
# Paper loss weights: focal=20, dice=1, iou_pred=0.01
#
# IoU target: uses the "expected IoU" formula from the paper (Eq. 1):
#   E[IoU] = Σ(p_i * q_i) / Σ(1 - (1-p_i)(1-q_i))
# where p_i = sigmoid(logit_i), q_i = gt_proportion_i.
# This avoids instability from thresholding near-0.5 values.

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Sigmoid focal loss (supports soft targets in [0, 1]).

    Args:
        logits:  Predicted logits, arbitrary shape.
        targets: Soft binary targets in [0, 1], same shape.
        alpha, gamma: Focal loss hyper-parameters.
        reduction: 'mean', 'sum', or 'none'.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice loss over the last dimension(s).

    Args:
        logits:  (..., L) predicted logits.
        targets: (..., L) soft binary targets in [0, 1].
    """
    p = torch.sigmoid(logits)
    flat_p = p.flatten(-2) if p.dim() > 1 else p.unsqueeze(0)
    flat_t = targets.flatten(-2) if targets.dim() > 1 else targets.unsqueeze(0)
    inter = (flat_p * flat_t).sum(-1)
    denom = flat_p.sum(-1) + flat_t.sum(-1)
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def expected_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Expected IoU from the paper (Eq. 1):
        E[IoU] = Σ(p*q) / Σ(1-(1-p)(1-q))

    Uses real-valued p=sigmoid(logits) and q=gt_proportions to avoid
    thresholding instability near 0.5.

    Args:
        logits:  (K, L) — K masks, L valid pixels (flattened spatial).
        targets: (L,)   — GT proportions for those pixels.

    Returns:
        (K,) expected IoU for each mask.
    """
    p = torch.sigmoid(logits)          # (K, L)
    q = targets.unsqueeze(0)           # (1, L) → broadcasts to (K, L)
    intersection = (p * q).sum(-1)     # (K,)
    union = (1 - (1 - p) * (1 - q)).sum(-1)  # (K,)
    return intersection / (union + eps)  # (K,)


def segmentation_loss(
    pred_masks: torch.Tensor,
    pred_ious: torch.Tensor,
    gt_masks: torch.Tensor,
    valid_token_mask: torch.Tensor,
    focal_weight: float = 20.0,
    dice_weight: float = 1.0,
    iou_weight: float = 0.01,
) -> torch.Tensor:
    """
    Combined segmentation loss (focal + dice + IoU prediction MSE).

    Multi-mask supervision (K masks per prompt):
      - Use expected IoU to select the best mask (avoids thresholding instability).
      - Apply focal + dice to the best mask, IoU prediction loss to all K.

    Args:
        pred_masks:       (B, K, N, P, P) predicted logits.
        pred_ious:        (B, K) predicted IoU values (from IoU head).
        gt_masks:         (B, N, P, P) GT mask proportions in foveated space.
        valid_token_mask: (B, N) bool.

    Returns:
        Scalar total loss averaged over batch.
    """
    B, K, N, P, _ = pred_masks.shape
    total = pred_masks.new_zeros(())

    for b in range(B):
        valid = valid_token_mask[b]      # (N,) bool
        n_valid = valid.sum().item()
        if n_valid == 0:
            continue

        # Flatten valid token pixels: (K, n_valid*P*P) and (n_valid*P*P,)
        pred_b = pred_masks[b, :, valid].flatten(1)    # (K, M)
        gt_b   = gt_masks[b, valid].flatten()           # (M,)
        piou_b = pred_ious[b]                           # (K,)

        # ── Select best mask via expected IoU ──────────────────────────────
        with torch.no_grad():
            e_ious = expected_iou(pred_b, gt_b)  # (K,)
        best_k = e_ious.argmax()

        # ── Focal + Dice on best mask ──────────────────────────────────────
        best_pred = pred_b[best_k]   # (M,)
        loss_f = sigmoid_focal_loss(best_pred, gt_b)
        loss_d = dice_loss(best_pred.unsqueeze(0), gt_b.unsqueeze(0))

        # ── IoU prediction MSE (all K masks) ──────────────────────────────
        # piou_b is raw (unbounded); apply sigmoid to get [0,1] prediction,
        # compare against expected IoU targets.
        loss_iou = F.mse_loss(piou_b.sigmoid(), e_ious.detach())

        total = total + focal_weight * loss_f + dice_weight * loss_d + iou_weight * loss_iou

    return total / B
