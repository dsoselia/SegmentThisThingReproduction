#!/bin/bash
# MAE pre-training using COCO train2017 images (118K images, no masks needed).
# Same paper hyperparameters as SA-1B run; COCO gives a cleaner controlled experiment.
#
# Submit (H200): sbatch slurm/pretrain_coco.sh
# Submit (any GPU): sbatch --partition=clip --gres=gpu:rtxa6000:4 slurm/pretrain_coco.sh

#SBATCH --job-name=stt-mae-coco
#SBATCH --output=logs/mae_coco_%j.out
#SBATCH --error=logs/mae_coco_%j.err
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
COCO_IMAGES=/fs/cml-datasets/coco/images/train2017
OUTPUT_DIR=${REPO_DIR}/checkpoints/mae_coco
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
        echo "[pretrain_coco] ERROR: /tmp has only $((FREE_KB/1024)) MB free (need ~20 GB). Aborting."
        exit 1
    fi
    echo "[pretrain_coco] Staging COCO train2017 to ${LOCAL_IMAGES} (have ${ACTUAL}/${EXPECTED}) ..."
    rsync -a --no-perms --update "${COCO_IMAGES}/" "${LOCAL_IMAGES}/"
    echo "[pretrain_coco] Staged: $(find ${LOCAL_IMAGES} -maxdepth 1 -name '*.jpg' | wc -l) images"
else
    echo "[pretrain_coco] Reusing cached staging at ${LOCAL_IMAGES} (${ACTUAL} images)"
fi

# ── Environment ───────────────────────────────────────────────────────────────
source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate torch-py313
cd "${REPO_DIR}"

NPROC=${SLURM_NTASKS_PER_NODE:-4}
# Unique port per job to avoid collisions with zombie processes from cancelled runs
MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 1000 ))

RESUME_ARG=""
if [ -f "${OUTPUT_DIR}/latest.pth" ]; then
    RESUME_ARG="--resume ${OUTPUT_DIR}/latest.pth"
fi

torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    training/train_mae.py \
    --data-root "${LOCAL_IMAGES}" \
    --data-source coco \
    --output-dir "${OUTPUT_DIR}" \
    --model-size b \
    --total-steps 500000 \
    --warmup-steps 10000 \
    --lr 1.2207e-4 \
    --weight-decay 0.001 \
    --mask-ratio 0.75 \
    --images-per-gpu 128 \
    --num-workers 6 \
    --log-interval 50 \
    --save-interval 500 \
    --milestone-interval 25000 \
    --keep-milestones 3 \
    --val-data-root /fs/cml-datasets/coco/images/val2017 \
    --val-images 256 \
    --wandb-project segment-this-thing \
    --wandb-run-name mae-b-coco \
    ${RESUME_ARG}
