#!/bin/bash
# MAE pre-training — actual full run, 2×RTX A6000 (48 GB each).
# images-per-gpu reduced to 64 vs H200 configs to stay within 48 GB VRAM.
# Grad accumulation auto-adjusts to maintain target batch size.
#
# Submit: sbatch slurm/pretrain_coco_actual_2xa6000.sh

#SBATCH --job-name=stt-mae-act-a6
#SBATCH --output=logs/mae_actual_2xa6000_%j.out
#SBATCH --error=logs/mae_actual_2xa6000_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=cml-scavenger
#SBATCH --account=cml-scavenger
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

set -e

REPO_DIR=/cmlscratch/dsoselia/SegmentThisThingRectilinear/SegmentThisThingReproduction
COCO_IMAGES=/fs/cml-datasets/coco/images/train2017
OUTPUT_DIR=${REPO_DIR}/checkpoints/mae_coco_lr_actual
mkdir -p "${OUTPUT_DIR}" "${REPO_DIR}/logs"

LOCAL_IMAGES=/tmp/stt_coco_train2017
EXPECTED=118287
rm -rf /tmp/coco_train2017_* 2>/dev/null || true
mkdir -p "${LOCAL_IMAGES}"
ACTUAL=$(find "${LOCAL_IMAGES}" -maxdepth 1 -name '*.jpg' | wc -l)
if [ "${ACTUAL}" -lt "${EXPECTED}" ]; then
    FREE_KB=$(df -k /tmp | awk 'NR==2 {print $4}')
    if [ "${FREE_KB}" -lt 20971520 ]; then
        echo "[pretrain_actual] ERROR: /tmp has only $((FREE_KB/1024)) MB free (need ~20 GB). Aborting."
        exit 1
    fi
    echo "[pretrain_actual] Staging COCO train2017 to ${LOCAL_IMAGES} (have ${ACTUAL}/${EXPECTED}) ..."
    rsync -a --no-perms --update "${COCO_IMAGES}/" "${LOCAL_IMAGES}/"
    echo "[pretrain_actual] Staged: $(find ${LOCAL_IMAGES} -maxdepth 1 -name '*.jpg' | wc -l) images"
else
    echo "[pretrain_actual] Reusing cached staging at ${LOCAL_IMAGES} (${ACTUAL} images)"
fi

source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate torch-py313
cd "${REPO_DIR}"

NPROC=${SLURM_NTASKS_PER_NODE:-2}
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
    --images-per-gpu 64 \
    --num-workers 4 \
    --log-interval 50 \
    --save-interval 1000 \
    --milestone-interval 25000 \
    --keep-milestones 4 \
    --val-data-root /fs/cml-datasets/coco/images/val2017 \
    --val-images 256 \
    --wandb-project SegmentLogRectNexus_actual \
    --wandb-run-name mae-b-coco-lr-actual \
    ${RESUME_ARG}
