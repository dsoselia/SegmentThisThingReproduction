#!/bin/bash
# Segmentation fine-tuning for Segment This Thing.
# Paper: 250K iters, lr=2^-16, warmup 5K, batch 2048→doubling every 50K, 16 fov/image
#
# Submit: sbatch slurm/finetune.sh
# For H200 nodes: sbatch --gres=gpu:h200:4 slurm/finetune.sh

#SBATCH --job-name=stt-seg
#SBATCH --output=logs/seg_%j.out
#SBATCH --error=logs/seg_%j.err
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=96:00:00

set -e

REPO_DIR=/cmlscratch/dsoselia/SegmentThisThing
NFS_DATA=${REPO_DIR}/data/sa1b
OUTPUT_DIR=${REPO_DIR}/checkpoints/seg
MAE_CKPT=${REPO_DIR}/checkpoints/mae/final.pth
mkdir -p "${OUTPUT_DIR}" "${REPO_DIR}/logs"

# ── Stage data to local scratch (NVMe >> NFS for random JPEG reads) ──────────
LOCAL_DATA=${TMPDIR:-/tmp}/stt_sa1b_$$
mkdir -p "${LOCAL_DATA}/extracted"
echo "[finetune] Staging SA-1B data to ${LOCAL_DATA}/extracted ..."

if ls "${NFS_DATA}"/*.tar &>/dev/null; then
    for TAR in "${NFS_DATA}"/*.tar; do
        echo "[finetune] Extracting $(basename ${TAR}) ..."
        tar -xf "${TAR}" -C "${LOCAL_DATA}/extracted" --strip-components=0 \
            --wildcards '*.jpg' '*.json' 2>/dev/null || true
    done
elif [ -d "${NFS_DATA}/extracted" ]; then
    rsync -a --no-perms "${NFS_DATA}/extracted/" "${LOCAL_DATA}/extracted/"
fi

DATA_ROOT="${LOCAL_DATA}/extracted"
echo "[finetune] Data staged: $(ls ${DATA_ROOT}/*.jpg 2>/dev/null | wc -l) images"

# ── Environment ───────────────────────────────────────────────────────────────
source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate torch-py313
cd "${REPO_DIR}"

NPROC=${SLURM_NTASKS_PER_NODE:-4}

PRETRAIN_ARG=""
if [ -f "${MAE_CKPT}" ]; then
    PRETRAIN_ARG="--pretrain-ckpt ${MAE_CKPT}"
fi

RESUME_ARG=""
if [ -f "${OUTPUT_DIR}/latest.pth" ]; then
    RESUME_ARG="--resume ${OUTPUT_DIR}/latest.pth"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port=29501 \
    training/train_seg.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-size b \
    --total-steps 250000 \
    --warmup-steps 5000 \
    --lr 1.5259e-5 \
    --weight-decay 0.001 \
    --max-fov 16 \
    --images-per-gpu 8 \
    --num-workers 14 \
    --log-interval 50 \
    --save-interval 2500 \
    --wandb-project segment-this-thing \
    ${PRETRAIN_ARG} \
    ${RESUME_ARG}

# ── Cleanup ───────────────────────────────────────────────────────────────────
rm -rf "${LOCAL_DATA}"
