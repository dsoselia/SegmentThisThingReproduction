# Learning rate schedulers for STT training.

import math
import torch


class WarmupThenConstant(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup for `warmup_steps`, then hold LR constant.

    Used for MAE pre-training (paper: "10K step linear warm-up, after which
    it is held constant.  Instead of periodically dropping the learning rate,
    we double the batch size every 100K iterations.")
    """

    def __init__(self, optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = (step + 1) / self.warmup_steps
        else:
            scale = 1.0
        return [base_lr * scale for base_lr in self.base_lrs]


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup then cosine decay to `min_lr_scale * base_lr`.

    Used for segmentation fine-tuning (paper does not specify; cosine is
    standard for SAM-style training).
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_scale: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_scale = min_lr_scale
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = (step + 1) / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = self.min_lr_scale + 0.5 * (1 - self.min_lr_scale) * (
                1 + math.cos(math.pi * progress)
            )
        return [base_lr * scale for base_lr in self.base_lrs]
