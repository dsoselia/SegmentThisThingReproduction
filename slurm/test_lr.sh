#!/bin/bash
# End-to-end smoke test for Log-Rectilinear foveation.
# Runs MAE pretrain (200 steps) then seg fine-tune (100 steps) sequentially
# on the same node.  Uses COCO val2017 (5K images) to skip the 25-min rsync.
#
# Primary target: 2× A100 on cml32 (80 GB ea).
# Fallbacks (change --gres and drop --nodelist):
#   L40S:   --gres=gpu:l40s:2   --nodelist=cml34   (cml-scavenger)
#         or --gres=gpu:l40s:2   --nodelist=gammagpu18 --partition=scavenger --account=scavenger --qos=scavenger
#   A6000:  --gres=gpu:rtxa6000:2 --partition=clip --account=clip --qos=high (drop nodelist)
#
# Submit: sbatch slurm/test_lr.sh

#SBATCH --job-name=stt-lr-test
#SBATCH --output=logs/test_lr_%j.out
#SBATCH --error=logs/test_lr_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=cml-scavenger
#SBATCH --account=cml-scavenger
#SBATCH --gres=gpu:l40s:2
#SBATCH --nodelist=cml34
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@60

set -e

REPO_DIR=/cmlscratch/dsoselia/SegmentThisThingRectilinear/SegmentThisThingReproduction
COCO_ROOT=/fs/cml-datasets/coco
# Use val2017 (5K images, ~1 GB) to skip the 25-min train2017 rsync
COCO_IMAGES=${COCO_ROOT}/images/val2017
COCO_ANN=${COCO_ROOT}/annotations/instances_val2017.json

MAE_OUT=${REPO_DIR}/checkpoints/test_lr/mae
SEG_OUT=${REPO_DIR}/checkpoints/test_lr/seg
mkdir -p "${MAE_OUT}" "${SEG_OUT}" "${REPO_DIR}/logs"

# ── Stage val2017 to /tmp (~1 GB, takes seconds) ─────────────────────────────
LOCAL_IMAGES=/tmp/stt_coco_val2017
EXPECTED=5000
mkdir -p "${LOCAL_IMAGES}"
ACTUAL=$(find "${LOCAL_IMAGES}" -maxdepth 1 -name '*.jpg' | wc -l)
if [ "${ACTUAL}" -lt "${EXPECTED}" ]; then
    echo "[test_lr] Staging COCO val2017 to ${LOCAL_IMAGES} (have ${ACTUAL}) ..."
    rsync -a --no-perms --update "${COCO_IMAGES}/" "${LOCAL_IMAGES}/"
    echo "[test_lr] Staged: $(find ${LOCAL_IMAGES} -maxdepth 1 -name '*.jpg' | wc -l) images"
else
    echo "[test_lr] Reusing cached staging at ${LOCAL_IMAGES} (${ACTUAL} images)"
fi

# ── Environment ───────────────────────────────────────────────────────────────
source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate torch-py313
cd "${REPO_DIR}"

NPROC=2
MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 1000 ))

# ── Phase 1: MAE pre-training (200 steps) ────────────────────────────────────
echo "[test_lr] === Phase 1: MAE pre-training ==="
torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    training/train_mae.py \
    --data-root "${LOCAL_IMAGES}" \
    --data-source coco \
    --output-dir "${MAE_OUT}" \
    --model-size b \
    --total-steps 200 \
    --warmup-steps 20 \
    --lr 1.2207e-4 \
    --weight-decay 0.001 \
    --mask-ratio 0.75 \
    --images-per-gpu 64 \
    --num-workers 4 \
    --log-interval 10 \
    --save-interval 100 \
    --milestone-interval 100 \
    --keep-milestones 2 \
    --val-data-root "${COCO_IMAGES}" \
    --val-images 64 \
    --wandb-project segment-this-thing \
    --wandb-run-name mae-b-coco-lr-test

echo "[test_lr] MAE pre-training complete."

# ── Phase 2: Segmentation fine-tuning (100 steps) ────────────────────────────
echo "[test_lr] === Phase 2: Segmentation fine-tuning ==="
MAE_CKPT="${MAE_OUT}/final.pth"
PRETRAIN_ARG=""
if [ -f "${MAE_CKPT}" ]; then
    PRETRAIN_ARG="--pretrain-ckpt ${MAE_CKPT}"
    echo "[test_lr] Loading MAE encoder from ${MAE_CKPT}"
fi

torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="$(( MASTER_PORT + 1 ))" \
    training/train_seg.py \
    --data-root "${LOCAL_IMAGES}" \
    --data-source coco \
    --ann-file "${COCO_ANN}" \
    --output-dir "${SEG_OUT}" \
    --model-size b \
    --total-steps 100 \
    --warmup-steps 10 \
    --lr 1.5259e-5 \
    --weight-decay 0.001 \
    --max-fov 8 \
    --images-per-gpu 16 \
    --num-workers 4 \
    --log-interval 10 \
    --save-interval 50 \
    --wandb-project segment-this-thing \
    --wandb-run-name seg-b-coco-lr-test \
    ${PRETRAIN_ARG}

echo "[test_lr] === Both phases complete. ==="
