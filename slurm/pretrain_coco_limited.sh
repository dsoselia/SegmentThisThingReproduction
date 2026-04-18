#!/bin/bash
# MAE pre-training — limited run (~4h on 4×H200).
# 20K steps @ ~1.76 steps/sec = ~3.2h training + ~30min staging buffer.
#
# Submit: sbatch slurm/pretrain_coco_limited.sh

#SBATCH --job-name=stt-mae-lr-lim
#SBATCH --output=logs/mae_coco_limited_%j.out
#SBATCH --error=logs/mae_coco_limited_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=cml-scavenger
#SBATCH --gres=gpu:h200-sxm:4
#SBATCH --account=cml-scavenger
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

set -e

REPO_DIR=/cmlscratch/dsoselia/SegmentThisThingRectilinear/SegmentThisThingReproduction
COCO_IMAGES=/fs/cml-datasets/coco/images/train2017
OUTPUT_DIR=${REPO_DIR}/checkpoints/mae_coco_lr_limited
mkdir -p "${OUTPUT_DIR}" "${REPO_DIR}/logs"

# ── Stage images to local scratch ────────────────────────────────────────────
LOCAL_IMAGES=/tmp/stt_coco_train2017
EXPECTED=118287
rm -rf /tmp/coco_train2017_* 2>/dev/null || true
mkdir -p "${LOCAL_IMAGES}"
ACTUAL=$(find "${LOCAL_IMAGES}" -maxdepth 1 -name '*.jpg' | wc -l)
if [ "${ACTUAL}" -lt "${EXPECTED}" ]; then
    FREE_KB=$(df -k /tmp | awk 'NR==2 {print $4}')
    if [ "${FREE_KB}" -lt 20971520 ]; then
        echo "[pretrain_limited] ERROR: /tmp has only $((FREE_KB/1024)) MB free (need ~20 GB). Aborting."
        exit 1
    fi
    echo "[pretrain_limited] Staging COCO train2017 to ${LOCAL_IMAGES} (have ${ACTUAL}/${EXPECTED}) ..."
    rsync -a --no-perms --update "${COCO_IMAGES}/" "${LOCAL_IMAGES}/"
    echo "[pretrain_limited] Staged: $(find ${LOCAL_IMAGES} -maxdepth 1 -name '*.jpg' | wc -l) images"
else
    echo "[pretrain_limited] Reusing cached staging at ${LOCAL_IMAGES} (${ACTUAL} images)"
fi

# ── Environment ───────────────────────────────────────────────────────────────
source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate torch-py313
cd "${REPO_DIR}"

NPROC=${SLURM_NTASKS_PER_NODE:-4}
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
    --total-steps 20000 \
    --warmup-steps 1000 \
    --lr 1.2207e-4 \
    --weight-decay 0.001 \
    --mask-ratio 0.75 \
    --images-per-gpu 128 \
    --num-workers 6 \
    --log-interval 50 \
    --save-interval 500 \
    --milestone-interval 5000 \
    --keep-milestones 4 \
    --val-data-root /fs/cml-datasets/coco/images/val2017 \
    --val-images 256 \
    --wandb-project SegmentLogRectNexus \
    --wandb-run-name mae-b-coco-lr-limited \
    ${RESUME_ARG}
