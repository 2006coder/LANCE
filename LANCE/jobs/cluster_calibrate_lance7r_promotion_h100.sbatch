#!/usr/bin/env bash
# Freeze the selected LANCE-7R trainval recipe using random validation only.
# Heldout and OOD files are not opened by this job.

#SBATCH --job-name=Enz_Sub_L7R_calibrate
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/new_approach}"
RUN_STAMP="${RUN_STAMP:-20260724r1}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw/CatPred-DB}"
ESM_CACHE="${ESM_CACHE:-${PROJECT_ROOT}/data/new_approach/esmpp_large_per_residue.pt}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/lance}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${RUN_ROOT}/lance7r_screen_${RUN_STAMP}_summary}"
OUTPUT_NAME="${OUTPUT_NAME:-promotion_recipe.json}"
MAX_BLEND_ALPHA="${MAX_BLEND_ALPHA:-0.50}"

export PATH="${ENV_PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${ENV_PREFIX}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/LANCE:${PROJECT_ROOT}/shims:${PYTHONPATH:-}"
export DGLBACKEND=pytorch
export DGL_GRAPHBOLT_SKIP_LOAD=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

checkpoints=()
for seed in 0 1 2; do
  checkpoints+=(
    "${RUN_ROOT}/lance7r_screen_exact_nca_strict_${RUN_STAMP}_seed${seed}/checkpoint_best.pt"
  )
done

"${ENV_PREFIX}/bin/python" LANCE/lance7r/calibrate_promotion.py \
  --screen-summary "${SUMMARY_ROOT}/screen_summary.json" \
  --checkpoint "${checkpoints[@]}" \
  --data-root "${DATA_ROOT}" \
  --esm-cache "${ESM_CACHE}" \
  --output "${SUMMARY_ROOT}/${OUTPUT_NAME}" \
  --batch-size 128 \
  --num-workers 0 \
  --device cuda \
  --max-blend-alpha "${MAX_BLEND_ALPHA}"
