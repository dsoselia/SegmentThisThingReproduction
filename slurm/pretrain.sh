#!/bin/bash
# MAE pre-training for Segment This Thing.
# Paper: 500K iters, lr=2^-13, warmup 10K, constant LR, batch 1024→doubling every 100K
#
# Submit: sbatch slurm/pretrain.sh
# For H200 nodes: sbatch --gres=gpu:h200:4 slurm/pretrain.sh

#SBATCH --job-name=stt-mae
#SBATCH --output=logs/mae_%j.out
#SBATCH --error=logs/mae_%j.err
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=96:00:00

set -e

REPO_DIR=/cmlscratch/dsoselia/SegmentThisThing
NFS_DATA=${REPO_DIR}/data/sa1b
OUTPUT_DIR=${REPO_DIR}/checkpoints/mae
mkdir -p "${OUTPUT_DIR}" "${REPO_DIR}/logs"

# ── Stage data to local scratch (NVMe >> NFS for random JPEG reads) ──────────
LOCAL_DATA=${TMPDIR:-/tmp}/stt_sa1b_$$
mkdir -p "${LOCAL_DATA}/extracted"
echo "[pretrain] Staging SA-1B data to ${LOCAL_DATA}/extracted ..."

# Always extract from tars to local scratch so all shards are available.
# Falls back to rsync of pre-extracted dir if no tars exist.
if ls "${NFS_DATA}"/*.tar &>/dev/null; then
    for TAR in "${NFS_DATA}"/*.tar; do
        echo "[pretrain] Extracting $(basename ${TAR}) ..."
        tar -xf "${TAR}" -C "${LOCAL_DATA}/extracted" --strip-components=0 \
            --wildcards '*.jpg' '*.json' 2>/dev/null || true
    done
elif [ -d "${NFS_DATA}/extracted" ]; then
    rsync -a --no-perms "${NFS_DATA}/extracted/" "${LOCAL_DATA}/extracted/"
fi

DATA_ROOT="${LOCAL_DATA}/extracted"
echo "[pretrain] Data staged: $(ls ${DATA_ROOT}/*.jpg 2>/dev/null | wc -l) images"

# ── Environment ───────────────────────────────────────────────────────────────
source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate torch-py313
cd "${REPO_DIR}"

NPROC=${SLURM_NTASKS_PER_NODE:-4}

RESUME_ARG=""
if [ -f "${OUTPUT_DIR}/latest.pth" ]; then
    RESUME_ARG="--resume ${OUTPUT_DIR}/latest.pth"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port=29500 \
    training/train_mae.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-size b \
    --total-steps 500000 \
    --warmup-steps 10000 \
    --lr 1.2207e-4 \
    --weight-decay 0.001 \
    --mask-ratio 0.75 \
    --images-per-gpu 64 \
    --num-workers 14 \
    --log-interval 50 \
    --save-interval 10000 \
    --wandb-project segment-this-thing \
    ${RESUME_ARG}

# ── Cleanup ───────────────────────────────────────────────────────────────────
rm -rf "${LOCAL_DATA}"
