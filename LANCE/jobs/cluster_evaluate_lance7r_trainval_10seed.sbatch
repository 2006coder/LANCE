#!/usr/bin/env bash
# One-shot ten-member LANCE-7R benchmark after the final recipe is frozen.

#SBATCH --job-name=Enz_Sub_L7R_eval
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/new_approach}"
RUN_STAMP="${RUN_STAMP:-20260724}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/lance}"
RUN_PREFIX="${RUN_PREFIX:-lance7r_exact_nca_trainval_${RUN_STAMP}_seed}"
ENSEMBLE_NAME="${ENSEMBLE_NAME:-lance7r_exact_nca_trainval_${RUN_STAMP}_ensemble10}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw/CatPred-DB}"
ESM_CACHE="${ESM_CACHE:-${PROJECT_ROOT}/data/new_approach/esmpp_large_per_residue.pt}"

export PATH="${ENV_PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${ENV_PREFIX}"
cd "${PROJECT_ROOT}"

mkdir -p logs/slurm runs/lance
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/LANCE:${PROJECT_ROOT}/shims:${PYTHONPATH:-}"
export DGLBACKEND=pytorch
export DGL_GRAPHBOLT_SKIP_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

checkpoints=()
for seed in $(seq 0 9); do
  checkpoint="${RUN_ROOT}/${RUN_PREFIX}${seed}/checkpoint_best.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing final LANCE-7R checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  checkpoints+=("${checkpoint}")
done

summary_dir="${RUN_ROOT}/${ENSEMBLE_NAME}"
summary_path="${summary_dir}/evaluation_summary.json"
predictions_path="${summary_dir}/per_row_predictions.csv"
score_json="${summary_dir}/calibration_score.json"
score_csv="${summary_dir}/calibration_score.csv"
mkdir -p "${summary_dir}"

"${ENV_PREFIX}/bin/python" LANCE/lance7r/evaluate.py \
  --checkpoint "${checkpoints[@]}" \
  --data-root "${DATA_ROOT}" \
  --esm-cache "${ESM_CACHE}" \
  --output "${summary_path}" \
  --predictions-output "${predictions_path}" \
  --batch-size "${ENSEMBLE_EVAL_BATCH_SIZE:-256}" \
  --num-workers 0 \
  --device cuda

"${ENV_PREFIX}/bin/python" LANCE/calibrate_and_score.py \
  --predictions-csv "${predictions_path}" \
  --output-json "${score_json}" \
  --output-csv "${score_csv}"

echo "[lance7r] benchmark completed=$(date -Is) summary=${summary_path}"
