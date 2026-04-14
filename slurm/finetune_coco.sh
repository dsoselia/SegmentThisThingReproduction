#!/bin/bash
# Segmentation fine-tuning using COCO train2017 + instance annotations.
# 860K annotated instances across 118K images.
#
# Submit (H200): sbatch slurm/finetune_coco.sh
# Submit (any GPU): sbatch --partition=clip --gres=gpu:rtxa6000:4 slurm/finetune_coco.sh

#SBATCH --job-name=stt-seg-coco
#SBATCH --output=logs/seg_coco_%j.out
#SBATCH --error=logs/seg_coco_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=cml-scavenger
#SBATCH --gres=gpu:h200-sxm:4
#SBATCH --account=cml-scavenger
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

set -e

REPO_DIR=/cmlscratch/dsoselia/SegmentThisThing
COCO_ROOT=/fs/cml-datasets/coco
COCO_IMAGES=${COCO_ROOT}/images/train2017
COCO_ANN=${COCO_ROOT}/annotations/instances_train2017.json
OUTPUT_DIR=${REPO_DIR}/checkpoints/seg_coco
MAE_CKPT=${REPO_DIR}/checkpoints/mae_coco/final.pth
mkdir -p "${OUTPUT_DIR}" "${REPO_DIR}/logs"

# ── Stage images to local scratch ────────────────────────────────────────────
# Use a fixed path (no PID) so reruns on the same node skip the 25-min rsync.
LOCAL_IMAGES=/tmp/stt_coco_train2017
EXPECTED=118287
# Remove stale PID-based staging dirs left by cancelled jobs
rm -rf /tmp/coco_train2017_* 2>/dev/null || true
mkdir -p "${LOCAL_IMAGES}"
ACTUAL=$(find "${LOCAL_IMAGES}" -maxdepth 1 -name '*.jpg' | wc -l)
if [ "${ACTUAL}" -lt "${EXPECTED}" ]; then
    # Verify enough free space (~20 GB needed)
    FREE_KB=$(df -k /tmp | awk 'NR==2 {print $4}')
    if [ "${FREE_KB}" -lt 20971520 ]; then
        echo "[finetune_coco] ERROR: /tmp has only $((FREE_KB/1024)) MB free (need ~20 GB). Aborting."
        exit 1
    fi
    echo "[finetune_coco] Staging COCO train2017 to ${LOCAL_IMAGES} (have ${ACTUAL}/${EXPECTED}) ..."
    rsync -a --no-perms --update "${COCO_IMAGES}/" "${LOCAL_IMAGES}/"
    echo "[finetune_coco] Staged: $(find ${LOCAL_IMAGES} -maxdepth 1 -name '*.jpg' | wc -l) images"
else
    echo "[finetune_coco] Reusing cached staging at ${LOCAL_IMAGES} (${ACTUAL} images)"
fi

# ── Environment ───────────────────────────────────────────────────────────────
source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate torch-py313
cd "${REPO_DIR}"

NPROC=${SLURM_NTASKS_PER_NODE:-4}
MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 1000 ))

PRETRAIN_ARG=""
if [ -f "${MAE_CKPT}" ]; then
    PRETRAIN_ARG="--pretrain-ckpt ${MAE_CKPT}"
fi

RESUME_ARG=""
if [ -f "${OUTPUT_DIR}/latest.pth" ]; then
    RESUME_ARG="--resume ${OUTPUT_DIR}/latest.pth"
fi

torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    training/train_seg.py \
    --data-root "${LOCAL_IMAGES}" \
    --data-source coco \
    --ann-file "${COCO_ANN}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-size b \
    --total-steps 250000 \
    --warmup-steps 5000 \
    --lr 1.5259e-5 \
    --weight-decay 0.001 \
    --max-fov 16 \
    --images-per-gpu 64 \
    --num-workers 6 \
    --log-interval 50 \
    --save-interval 500 \
    --wandb-project segment-this-thing \
    --wandb-run-name seg-b-coco \
    ${PRETRAIN_ARG} \
    ${RESUME_ARG}
