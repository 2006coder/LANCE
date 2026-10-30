#!/usr/bin/env bash
# ==============================================================================
# LANCE-4R trainer: one all-task checkpoint with a kcat rate-preserving path.
#
# Relative to completed LANCE-3S:
#   - pool native-width frozen ESM++ before projection for the kcat head only,
#   - put exact-reaction kcat pairs in every batch and rank within those cohorts,
#   - keep binding JEPA for Km/Ki but remove its feature/loss from kcat,
#   - add a small one-log-unit boundary loss for the remaining Km p1mag miss,
#   - remove the inert mechanistic rate prior, EC1 ranking, and kcat tail loss.
#
# TRAIN_SPLIT=train is the strict scientific track.  TRAIN_SPLIT=trainval is a
# separate CatPred data-parity track because the released CatPred reproduction
# recipe trains on random_trainval while also supplying random_val.
# ==============================================================================
#SBATCH --job-name=Enz_Sub_pred_Bach_L4R
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=190G
#SBATCH --time=40:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/new_approach}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${ENV_PREFIX}"
cd "${PROJECT_ROOT}"

mkdir -p logs/slurm data/new_approach runs/lance
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

SEED_BASE="${SEED_BASE:?Set SEED_BASE to 0, 2, 4, 6, or 8}"
SEED_COUNT="${SEED_COUNT:-2}"
BATCH_SIZE_PER_SEED="${BATCH_SIZE_PER_SEED:-256}"
EVAL_BATCH_SIZE_PER_SEED="${EVAL_BATCH_SIZE_PER_SEED:-128}"
GPU_VISIBLE="${CUDA_VISIBLE_DEVICES:-0}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw/CatPred-DB}"
ESM_CACHE="${ESM_CACHE:-${PROJECT_ROOT}/data/new_approach/esmpp_large_per_residue.pt}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/lance}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"

case "${TRAIN_SPLIT}" in
  train)
    DEFAULT_RUN_PREFIX="lance4_rate_train_eng_h100x1_2seed_b256_20260716_seed"
    ;;
  trainval)
    DEFAULT_RUN_PREFIX="lance4_rate_trainval_eng_h100x1_2seed_b256_20260716_seed"
    ;;
  *)
    echo "TRAIN_SPLIT must be train or trainval; got ${TRAIN_SPLIT}" >&2
    exit 2
    ;;
esac
RUN_PREFIX="${RUN_NAME_PREFIX:-${DEFAULT_RUN_PREFIX}}"

if [[ "${SEED_COUNT}" != "2" ]]; then
  echo "SEED_COUNT must remain 2 for this shared-H100 script; got ${SEED_COUNT}" >&2
  exit 2
fi
if [[ ! -f "${ESM_CACHE}" ]]; then
  echo "Missing frozen ESM++ cache: ${ESM_CACHE}" >&2
  exit 2
fi

pids=()
seeds=()
for offset in $(seq 0 "$((SEED_COUNT - 1))"); do
  seed=$(( SEED_BASE + offset ))
  run_name="${RUN_PREFIX}${seed}"
  worker_out="${PROJECT_ROOT}/logs/slurm/lance4_rate_${TRAIN_SPLIT}_seed${seed}_${SLURM_JOB_ID}.out"
  worker_err="${PROJECT_ROOT}/logs/slurm/lance4_rate_${TRAIN_SPLIT}_seed${seed}_${SLURM_JOB_ID}.err"
  echo "[lance4-rate] launching split=${TRAIN_SPLIT} seed=${seed} run=${run_name}"
  CUDA_VISIBLE_DEVICES="${GPU_VISIBLE}" "${ENV_PREFIX}/bin/python" LANCE/train.py \
    --data-root "${DATA_ROOT}" \
    --train-split "${TRAIN_SPLIT}" \
    --esm-cache "${ESM_CACHE}" \
    --output-dir "${RUN_ROOT}" \
    --run-name "${run_name}" \
    --device cuda \
    --seed "${seed}" \
    --tasks kcat km ki \
    --epochs 150 \
    --early-stop-patience 20 \
    --batch-size "${BATCH_SIZE_PER_SEED}" \
    --eval-batch-size "${EVAL_BATCH_SIZE_PER_SEED}" \
    --num-workers 0 \
    --init-lr 0.00006 \
    --max-lr 0.00035 \
    --final-lr 0.000006 \
    --warmup-epochs 3 \
    --weight-decay 0.00008 \
    --grad-clip 1.0 \
    --kinetic-weight 1.0 \
    --jepa-weight 0.35 \
    --jepa-delta-weight 0.25 \
    --jepa-contrastive-weight 0.05 \
    --action-decorrelation-weight 0.01 \
    --sigreg-weight 0.02 \
    --binding-aux-weight 0.12 \
    --binding-jepa-weight 0.15 \
    --binding-jepa-delta-weight 0.10 \
    --binding-jepa-contrastive-weight 0.02 \
    --binding-jepa-exclude-kcat \
    --kcat-tail-weight 0.0 \
    --kcat-mechanism-ranking-weight 0.0 \
    --kcat-reaction-ranking-weight 0.03 \
    --kcat-ranking-temperature 0.35 \
    --kcat-ranking-min-gap 0.25 \
    --km-p1-weight 0.03 \
    --km-p1-temperature 0.15 \
    --gaussian-loss beta_nll \
    --beta-nll-beta 0.5 \
    --ec-aux-weight 0.10 \
    --max-residues 1024 \
    --dmpnn-hidden 300 \
    --dmpnn-depth 3 \
    --dmpnn-dropout 0.0 \
    --model-dim 512 \
    --action-dim 64 \
    --hidden-dim 1024 \
    --dropout 0.15 \
    --use-task-interface-gate \
    --use-task-residual-actions \
    --task-action-adapter-dim 256 \
    --use-gaussian-heads \
    --use-condition-features \
    --use-ec-aux \
    --use-kcat-global-rate-context \
    --use-binding-jepa \
    --reaction-cohort-sampler \
    --kcat-cohort-fraction 0.50 \
    --score-r2-weight 0.10 \
    --score-p1mag-weight 0.001 \
    --log-every 20 \
    --target-vram-gb "${TARGET_VRAM_PER_SEED_GB:-38.0}" \
    --target-vram-tolerance-gb 4.0 >"${worker_out}" 2>"${worker_err}" &
  pids+=("$!")
  seeds+=("${seed}")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "[lance4-rate] split=${TRAIN_SPLIT} seed=${seeds[${index}]} completed=$(date -Is)"
  else
    status=$?
    echo "[lance4-rate] split=${TRAIN_SPLIT} seed=${seeds[${index}]} failed_exit=${status}" >&2
    failed=1
  fi
done

if (( failed != 0 )); then
  exit 1
fi
echo "[lance4-rate] group_completed=$(date -Is) split=${TRAIN_SPLIT} seed_base=${SEED_BASE}"
