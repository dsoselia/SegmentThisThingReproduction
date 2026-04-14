#!/bin/bash
# MAE pre-training on 2× RTX A6000 (48 GB each) — clip partition.
# Shares the same checkpoint dir as the H200 run so jobs can hand off
# seamlessly: whichever is available picks up from latest.pth.
#
# Submit: sbatch slurm/pretrain_coco_a6000.sh

#SBATCH --job-name=stt-mae-coco-a6k
#SBATCH --output=logs/mae_coco_%j.out
#SBATCH --error=logs/mae_coco_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@120
# Note: clip partition does not support --requeue; preemption is handled
# via SIGUSR1 → clean checkpoint save → manual resubmission if needed.

set -e

REPO_DIR=/cmlscratch/dsoselia/SegmentThisThing
COCO_IMAGES=/fs/cml-datasets/coco/images/train2017
# Same output dir as H200 run — jobs share latest.pth for seamless handoff
OUTPUT_DIR=${REPO_DIR}/checkpoints/mae_coco
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
        echo "[pretrain_a6000] ERROR: /tmp has only $((FREE_KB/1024)) MB free (need ~20 GB). Aborting."
        exit 1
    fi
    echo "[pretrain_a6000] Staging COCO train2017 to ${LOCAL_IMAGES} (have ${ACTUAL}/${EXPECTED}) ..."
    rsync -a --no-perms --update "${COCO_IMAGES}/" "${LOCAL_IMAGES}/"
    echo "[pretrain_a6000] Staged: $(find ${LOCAL_IMAGES} -maxdepth 1 -name '*.jpg' | wc -l) images"
else
    echo "[pretrain_a6000] Reusing cached staging at ${LOCAL_IMAGES} (${ACTUAL} images)"
fi

# ── Environment ───────────────────────────────────────────────────────────────
source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate torch-py313
cd "${REPO_DIR}"

NPROC=2  # hardcoded: 2x RTX A6000 requested via --gres
MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 1000 ))

RESUME_ARG=""
if [ -f "${OUTPUT_DIR}/latest.pth" ]; then
    RESUME_ARG="--resume ${OUTPUT_DIR}/latest.pth"
fi

# A6000 has 48 GB VRAM.  With the dummy-mask fix the encoder uses ~41 GB
# at images_per_gpu=128 on H200; leaving only ~7 GB headroom is risky over
# long runs (allocator fragmentation).  96 gives a comfortable buffer while
# still saturating GPU compute.
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
    --images-per-gpu 96 \
    --num-workers 4 \
    --log-interval 50 \
    --save-interval 500 \
    --milestone-interval 25000 \
    --keep-milestones 3 \
    --val-data-root /fs/cml-datasets/coco/images/val2017 \
    --val-images 256 \
    --wandb-project segment-this-thing \
    --wandb-run-name mae-b-coco \
    ${RESUME_ARG}
