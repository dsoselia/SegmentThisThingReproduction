#!/bin/bash
# One-shot evaluation: seg sam_miou + MAE crop-space reconstruction.
# Targets clip partition (A6000, non-preemptable).
#
# Submit: sbatch slurm/eval_a6000.sh

#SBATCH --job-name=stt-eval-lr
#SBATCH --output=logs/eval_lr_%j.out
#SBATCH --error=logs/eval_lr_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -e

REPO_DIR=/cmlscratch/dsoselia/SegmentThisThingRectilinear/SegmentThisThingReproduction
SEG_CKPT=${REPO_DIR}/checkpoints/seg_coco_lr_actual/latest.pth
MAE_CKPT_25K=${REPO_DIR}/checkpoints/mae_coco_lr_actual/step_0025000.pth
MAE_CKPT_50K=${REPO_DIR}/checkpoints/mae_coco_lr_actual/step_0050000.pth
MAE_CKPT_75K=${REPO_DIR}/checkpoints/mae_coco_lr_actual/step_0075000.pth
MAE_CKPT_LATEST=${REPO_DIR}/checkpoints/mae_coco_lr_actual/latest.pth

VAL_IMAGES=/fs/cml-datasets/coco/images/val2017
VAL_ANN=/fs/cml-datasets/coco/annotations/instances_val2017.json

mkdir -p "${REPO_DIR}/logs"

source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate torch-py313
cd "${REPO_DIR}"

echo "=== Log-rect Evaluation ==="
echo "Seg checkpoint: ${SEG_CKPT}"
echo "MAE checkpoints: 25K, 50K, 75K, latest"
echo "Eval images: 500"
echo ""

python evaluation/run_eval.py \
    --seg-ckpt      "${SEG_CKPT}" \
    --val-images-dir "${VAL_IMAGES}" \
    --val-ann-file   "${VAL_ANN}" \
    --eval-images   500 \
    --model-size    b

echo ""
echo "=== Done ==="
