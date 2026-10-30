#!/usr/bin/env bash
# Build five final LANCE-7R members concurrently on one 80 GB H100.
#
# The promotion recipe fixes the selected architecture, epoch count, optimizer
# settings, and familiar/unseen cluster blend strengths. This job reads only
# random_trainval labels; it cannot perform validation or benchmark selection.

#SBATCH --job-name=Enz_Sub_L7R_final
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/new_approach}"
RUN_STAMP="${RUN_STAMP:-20260724}"
SCREEN_STAMP="${SCREEN_STAMP:-20260724r1}"
SEED_BASE="${SEED_BASE:?Set SEED_BASE for this five-member group}"
SEED_COUNT="${SEED_COUNT:-5}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw/CatPred-DB}"
ESM_CACHE="${ESM_CACHE:-${PROJECT_ROOT}/data/new_approach/esmpp_large_per_residue.pt}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/lance}"
RECIPE="${RECIPE:-${RUN_ROOT}/lance7r_screen_${SCREEN_STAMP}_summary/promotion_recipe.json}"
INCUMBENT_PREFIX="${INCUMBENT_PREFIX:-lance4_rate_trainval_eng_h100x1_2seed_b256_20260716_seed}"
RUN_PREFIX="${RUN_PREFIX:-lance7r_exact_nca_trainval_${RUN_STAMP}_seed}"

export PATH="${ENV_PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${ENV_PREFIX}"
cd "${PROJECT_ROOT}"

mkdir -p logs/slurm runs/lance
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/LANCE:${PROJECT_ROOT}/shims:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET=1
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/cache/torch}"
export TOKENIZERS_PARALLELISM=false
export DGLBACKEND=pytorch
export DGL_GRAPHBOLT_SKIP_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [[ ! -f "${RECIPE}" ]]; then
  echo "Missing frozen promotion recipe: ${RECIPE}" >&2
  exit 2
fi

pids=()
seeds=()
for (( offset=0; offset<SEED_COUNT; offset++ )); do
  seed=$((SEED_BASE + offset))
  incumbent="${RUN_ROOT}/${INCUMBENT_PREFIX}${seed}/checkpoint_best.pt"
  if [[ ! -f "${incumbent}" ]]; then
    echo "Missing LANCE-4R trainval seed ${seed}: ${incumbent}" >&2
    exit 2
  fi

  run_name="${RUN_PREFIX}${seed}"
  worker_out="${PROJECT_ROOT}/logs/slurm/lance7r_final_seed${seed}_${SLURM_JOB_ID}.out"
  worker_err="${PROJECT_ROOT}/logs/slurm/lance7r_final_seed${seed}_${SLURM_JOB_ID}.err"
  echo "[lance7r] launching fixed final seed=${seed} run=${run_name}"
  CUDA_VISIBLE_DEVICES=0 "${ENV_PREFIX}/bin/python" LANCE/lance7r/train.py \
    --protocol fixed-final \
    --promotion-recipe "${RECIPE}" \
    --data-root "${DATA_ROOT}" \
    --esm-cache "${ESM_CACHE}" \
    --incumbent-checkpoint "${incumbent}" \
    --incumbent-split trainval \
    --memory-split trainval \
    --output-dir "${RUN_ROOT}" \
    --run-name "${run_name}" \
    --device cuda \
    --seed "${seed}" \
    --batch-size 256 \
    --eval-batch-size 128 \
    --num-workers 0 >"${worker_out}" 2>"${worker_err}" &
  pids+=("$!")
  seeds+=("${seed}")
  sleep 3
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "[lance7r] final seed=${seeds[${index}]} completed=$(date -Is)"
  else
    status=$?
    echo "[lance7r] final seed=${seeds[${index}]} failed_exit=${status}" >&2
    failed=1
  fi
done

if (( failed != 0 )); then
  exit 1
fi
echo "[lance7r] final group completed=$(date -Is)"
