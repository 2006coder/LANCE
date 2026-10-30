#!/usr/bin/env bash
# LANCE-4R ten-seed evaluator for either the strict or data-parity track.
#SBATCH --job-name=Enz_Sub_pred_Bach_eval_L4R
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/new_approach}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${ENV_PREFIX}"
cd "${PROJECT_ROOT}"

mkdir -p logs/slurm runs/lance
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/LANCE:${PROJECT_ROOT}/shims:${PYTHONPATH:-}"
export DGLBACKEND=pytorch
export DGL_GRAPHBOLT_SKIP_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/lance}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
case "${TRAIN_SPLIT}" in
  train)
    DEFAULT_RUN_PREFIX="lance4_rate_train_eng_h100x1_2seed_b256_20260716_seed"
    DEFAULT_ENSEMBLE_NAME="lance4_rate_train_eng_h100x1_2seed_b256_20260716_seed_ensemble10"
    ;;
  trainval)
    DEFAULT_RUN_PREFIX="lance4_rate_trainval_eng_h100x1_2seed_b256_20260716_seed"
    DEFAULT_ENSEMBLE_NAME="lance4_rate_trainval_eng_h100x1_2seed_b256_20260716_seed_ensemble10"
    ;;
  *)
    echo "TRAIN_SPLIT must be train or trainval; got ${TRAIN_SPLIT}" >&2
    exit 2
    ;;
esac
RUN_PREFIX="${RUN_NAME_PREFIX:-${DEFAULT_RUN_PREFIX}}"
ENSEMBLE_RUN_NAME="${ENSEMBLE_RUN_NAME:-${DEFAULT_ENSEMBLE_NAME}}"
ESM_CACHE="${ESM_CACHE:-${PROJECT_ROOT}/data/new_approach/esmpp_large_per_residue.pt}"

checkpoints=()
for seed in $(seq 0 9); do
  checkpoint="${RUN_ROOT}/${RUN_PREFIX}${seed}/checkpoint_best.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing LANCE-4R checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  checkpoints+=("${checkpoint}")
done

summary_dir="${RUN_ROOT}/${ENSEMBLE_RUN_NAME}"
summary_path="${summary_dir}/evaluation_summary.json"
predictions_path="${summary_dir}/per_row_predictions.csv"
calibration_json="${summary_dir}/calibration_score.json"
calibration_csv="${summary_dir}/calibration_score.csv"
mkdir -p "${summary_dir}"

"${ENV_PREFIX}/bin/python" LANCE/evaluate.py \
  --checkpoint "${checkpoints[@]}" \
  --data-root "${DATA_ROOT:-${PROJECT_ROOT}/data/raw/CatPred-DB}" \
  --esm-cache "${ESM_CACHE}" \
  --output "${summary_path}" \
  --predictions-output "${predictions_path}" \
  --batch-size "${ENSEMBLE_EVAL_BATCH_SIZE:-128}" \
  --num-workers 0 \
  --device cuda \
  --target-vram-gb "${ENSEMBLE_TARGET_VRAM_GB:-72}" \
  --target-vram-tolerance-gb "${ENSEMBLE_TARGET_VRAM_TOLERANCE_GB:-4}"
"${ENV_PREFIX}/bin/python" LANCE/calibrate_and_score.py \
  --predictions-csv "${predictions_path}" \
  --output-json "${calibration_json}" \
  --output-csv "${calibration_csv}"
echo "[lance4-rate-ensemble] finished=$(date -Is) split=${TRAIN_SPLIT} summary=${summary_path}"
