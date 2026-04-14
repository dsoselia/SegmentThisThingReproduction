# Segment This Thing — Project Guide

Replication of [Segment This Thing (CVPR 2025)](https://github.com/facebookresearch/segment_this_thing)
on Nexus cluster. The official repo is **inference-only**; this project adds the full training pipeline.

---

## Directory Layout

```
SegmentThisThing/
├── repo/                    # Upstream facebookresearch/segment_this_thing clone
│   └── segment_this_thing/  # Importable package (model, foveation, predictor, utils)
│
├── training/                # Our training code (tracked by git)
│   ├── train_mae.py         # Phase 1: MAE pre-training
│   ├── train_seg.py         # Phase 2: Segmentation fine-tuning
│   ├── datasets.py          # COCO + SA-1B dataset loaders
│   ├── losses.py            # Focal + Dice + IoU prediction losses
│   └── lr_scheduler.py      # WarmupThenConstant (MAE) + WarmupCosine (seg)
│
├── slurm/                   # SLURM job scripts (tracked by git)
│   ├── pretrain_coco.sh     # MAE on 4× H200 (cml-scavenger)
│   ├── pretrain_coco_a6000.sh  # MAE on 2× A6000 (clip/high)
│   ├── finetune_coco.sh     # Seg fine-tune on 4× H200
│   ├── pretrain.sh          # MAE on SA-1B (H200)
│   └── finetune.sh          # Seg fine-tune on SA-1B (H200)
│
├── evaluation/
│   └── eval.py              # Zero-shot single-point mIoU evaluation
│
├── checkpoints/             # Saved model weights (git-ignored)
│   └── mae_coco/
│       ├── latest.pth       # Always-current resume checkpoint (saved every 500 steps)
│       ├── step_0010000.pth # Milestone checkpoints (kept last 3, saved every 25K steps)
│       ├── step_0015000.pth
│       └── step_0025000.pth
│
├── data/
│   └── sa1b/                # SA-1B shards (3 downloaded: ~33 GB)
│       ├── sa_000020.tar    # ~11 GB each
│       ├── sa_000021.tar
│       ├── sa_000022.tar
│       └── extracted/       # ~11K images extracted for quick testing
│
├── weights/
│   └── stt-b.pth            # Official STT-B reference weights (for eval baseline)
│
├── logs/                    # SLURM stdout/stderr (git-ignored)
├── wandb/                   # W&B local run cache (git-ignored)
└── cache/                   # COCO dataset index cache (git-ignored, NFS-safe)
```

---

## Two Git Repositories

```
SegmentThisThing/         ← OUR repo (training pipeline)
SegmentThisThing/repo/    ← facebookresearch upstream (inference only)
```

`repo/` is listed in `.gitignore` so the two are **independent** — not nested.
Never `git add repo/`.

---

## Training Pipeline

### Phase 1: MAE Pre-training

Pre-trains the ViT-B image encoder with masked autoencoding on COCO train2017 (118K images).
A lightweight 2-layer MLP decoder reconstructs masked foveated token patches; decoder is
discarded after training, only the encoder is kept.

**Paper hyperparameters:**
- 500K steps, AdamW lr=2⁻¹³ ≈ 1.22e-4, constant after 10K warmup
- Batch starts at 1024, doubles every 100K steps (up to 16384)
- Masking ratio: 75% of 172 foveated tokens
- 2 foveated views per image

### Phase 2: Segmentation Fine-tuning

Fine-tunes the full model (encoder + neck + mask decoder) on COCO instance annotations.

**Paper hyperparameters:**
- 250K steps, AdamW lr=2⁻¹⁶ ≈ 1.53e-5
- Batch starts at 2048, doubles every 50K steps
- Up to 16 foveated views per image
- Loss: focal (×20) + dice (×1) + IoU prediction (×0.01)

---

## Running Jobs

### Environment

```bash
source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate torch-py313
cd /cmlscratch/dsoselia/SegmentThisThing
```

### Submit MAE Pre-training

```bash
# H200 (preferred, 4× GPU, 72h + auto-requeue):
sbatch slurm/pretrain_coco.sh

# A6000 fallback (2× GPU, 24h max, no requeue):
sbatch slurm/pretrain_coco_a6000.sh

# Watch logs live:
tail -f logs/mae_coco_<JOBID>.out
```

Both scripts share `checkpoints/mae_coco/latest.pth` — jobs hand off seamlessly.
If A6000 finishes its 24h slot, just resubmit; it resumes from the last checkpoint.

### Submit Segmentation Fine-tuning

```bash
# Requires a completed MAE checkpoint at checkpoints/mae_coco/final.pth (or latest.pth)
sbatch slurm/finetune_coco.sh
```

### Check Running Jobs

```bash
squeue -u dsoselia
```

### Monitor GPU Utilization (on the node)

```bash
# Find which node your job is on:
squeue -j <JOBID> -o "%R"

# SSH to node and watch GPU:
ssh clip05          # or cml35, etc.
watch -n1 nvidia-smi
```

---

## Partitions & GPU Types

| Partition | Account | GPU | VRAM | Max time | Requeue | Notes |
|---|---|---|---|---|---|---|
| `cml-scavenger` | `cml-scavenger` | H200 SXM | 141 GB | 72h | Yes | Preemptable; auto-requeueed via `--signal=SIGUSR1@120` |
| `clip` | `clip` | RTX A6000 | 48 GB | 24h | No | QoS `high`; stable, not preempted |

**Scavenger jobs can be killed at any time.** The training scripts handle this: SIGUSR1
is caught 120 s before the deadline, a clean checkpoint is saved, and the job requeueues
automatically. On restart the job resumes from `latest.pth` without manual intervention.

---

## Checkpointing Strategy

`latest.pth` is saved every **500 steps** (resume safety). Named milestones
(`step_XXXXXXX.pth`) are saved every **25K steps** with only the last 3 kept,
so checkpoint storage stays under ~4 GB for the entire 500K-step run.

To manually resume from a specific checkpoint:

```bash
torchrun ... training/train_mae.py --resume checkpoints/mae_coco/step_0025000.pth ...
```

---

## Batch Size Accumulation

Effective batch size is maintained across GPU configs via gradient accumulation.
`accum_steps = ceil(local_target / samples_per_gpu)` so the actual global batch
always meets or slightly exceeds the paper target regardless of GPU count or
`--images-per-gpu`.

| Config | images/gpu | accum | actual global | target |
|---|---|---|---|---|
| 4× H200 | 128 | 1 | 1024 | 1024 |
| 2× A6000 | 96 | 3 | 1152 | 1024 |

Log line format: `bs=ACTUAL(tgt=TARGET)` — gap is always visible.

---

## W&B Monitoring

Project: `segment-this-thing` at [wandb.ai/dsoselia/segment-this-thing](https://wandb.ai/dsoselia/segment-this-thing)

Logged metrics:
- `train/loss` — MAE reconstruction MSE (masked tokens only)
- `train/lr`, `train/sps`, `train/global_bs`, `train/global_bs_target`, `train/accum_steps`
- `val/loss` — reconstruction MSE on 256 held-out COCO val2017 images (logged every 500 steps)
- `val/reconstruction` — image grid: GT patches | masked input | decoded reconstruction

---

## Disk Usage

| Path | Size | Notes |
|---|---|---|
| `/cmlscratch/dsoselia/` | 400 GB quota | Currently ~190 GB used |
| `SegmentThisThing/` | ~5 GB | Code + active checkpoints |
| `miniconda3/` | ~62 GB | Conda base + envs (torch-py313, vllm, py12, …) |
| `data/sa1b/` | ~33 GB | 3 SA-1B shards |
| `/nfshomes/dsoselia/` | 30 GB quota | Home dir; Claude memory + git config here |

**Critical:** Never let `/cmlscratch` fill to 100% — the training process will crash
mid-step (W&B console capture writes to disk). Old checkpoints are auto-rotated but
watch for other projects accumulating data.

COCO images are read from `/fs/cml-datasets/coco/` (shared NFS, read-only). The
training scripts rsync them to `/tmp/stt_coco_train2017` at job start for fast local
I/O; reruns skip the rsync if the files are already there.

---

## Current Status (2026-04-14)

| Run | Job | Node | Step | Notes |
|---|---|---|---|---|
| MAE pretrain (A6000) | 6585172 | clip05 | ~43K / 500K | Active |
| Other (unrelated) | 6577147 | clip12 | — | Unrelated job |

MAE pretrain on H200 will be resubmitted (`sbatch slurm/pretrain_coco.sh`) once
the scavenger partition becomes available. It will resume from the same `latest.pth`.
