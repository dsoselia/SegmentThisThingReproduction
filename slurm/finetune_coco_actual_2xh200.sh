#!/bin/bash
# Segmentation fine-tuning — actual full run, 2×H200.
# 250K steps. Requires mae_coco_lr_actual/final.pth from pretrain.
#
# Submit: sbatch slurm/finetune_coco_actual_2xh200.sh
# With dependency: sbatch --dependency=afterok:<pretrain_jobid> slurm/finetune_coco_actual_2xh200.sh

#SBATCH --job-name=stt-seg-act-2h
#SBATCH --output=logs/seg_actual_2xh200_%j.out
#SBATCH --error=logs/seg_actual_2xh200_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=cml-scavenger
#SBATCH --account=cml-scavenger
#SBATCH --gres=gpu:h200-sxm:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

set -e

REPO_DIR=/cmlscratch/dsoselia/SegmentThisThingRectilinear/SegmentThisThingReproduction
COCO_ROOT=/fs/cml-datasets/coco
COCO_IMAGES=${COCO_ROOT}/images/train2017
COCO_ANN=${COCO_ROOT}/annotations/instances_train2017.json
OUTPUT_DIR=${REPO_DIR}/checkpoints/seg_coco_lr_actual
MAE_CKPT=${REPO_DIR}/checkpoints/mae_coco_lr_actual/final.pth
mkdir -p "${OUTPUT_DIR}" "${REPO_DIR}/logs"

LOCAL_IMAGES=/tmp/stt_coco_train2017
EXPECTED=118287
rm -rf /tmp/coco_train2017_* 2>/dev/null || true
mkdir -p "${LOCAL_IMAGES}"
ACTUAL=$(find "${LOCAL_IMAGES}" -maxdepth 1 -name '*.jpg' | wc -l)
if [ "${ACTUAL}" -lt "${EXPECTED}" ]; then
    FREE_KB=$(df -k /tmp | awk 'NR==2 {print $4}')
    if [ "${FREE_KB}" -lt 20971520 ]; then
        echo "[finetune_actual] ERROR: /tmp has only $((FREE_KB/1024)) MB free (need ~20 GB). Aborting."
        exit 1
    fi
    echo "[finetune_actual] Staging COCO train2017 to ${LOCAL_IMAGES} (have ${ACTUAL}/${EXPECTED}) ..."
    rsync -a --no-perms --update "${COCO_IMAGES}/" "${LOCAL_IMAGES}/"
    echo "[finetune_actual] Staged: $(find ${LOCAL_IMAGES} -maxdepth 1 -name '*.jpg' | wc -l) images"
else
    echo "[finetune_actual] Reusing cached staging at ${LOCAL_IMAGES} (${ACTUAL} images)"
fi

source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate torch-py313
cd "${REPO_DIR}"

NPROC=${SLURM_NTASKS_PER_NODE:-2}
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
    --num-workers 4 \
    --log-interval 50 \
    --save-interval 1000 \
    --val-data-root /fs/cml-datasets/coco/images/val2017 \
    --val-ann-file /fs/cml-datasets/coco/annotations/instances_val2017.json \
    --val-images 128 \
    --val-interval 2500 \
    --wandb-project SegmentLogRectNexus_actual \
    --wandb-run-name seg-b-coco-lr-actual \
    ${PRETRAIN_ARG} \
    ${RESUME_ARG}
