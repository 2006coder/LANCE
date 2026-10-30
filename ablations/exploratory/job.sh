#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$1"
STAGE="$2"
source "${PACKAGE_DIR}/configs/cluster.env"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${ENV_PREFIX}"
export PYTHONPATH="${PACKAGE_DIR}/runtime:${PACKAGE_DIR}:${PROJECT_ROOT}/shims"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export TOKENIZERS_PARALLELISM=false
export DGLBACKEND=pytorch
export DGL_GRAPHBOLT_SKIP_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PACKAGE_DIR}"
case "${STAGE}" in
  preflight)
    exec "${ENV_PREFIX}/bin/python" preflight.py --run-root "${RUN_ROOT}" --data-root "${DATA_ROOT}" --esm-cache "${ESM_CACHE}"
    ;;
  train|refit)
    COMMAND=("${ENV_PREFIX}/bin/python" run.py --run-root "${RUN_ROOT}" --esm-cache "${ESM_CACHE}" --index "${SLURM_ARRAY_TASK_ID:-0}" --index-map "${3:?}")
    if [[ "${STAGE}" == refit ]]; then COMMAND+=(--refit); fi
    exec "${COMMAND[@]}"
    ;;
  summarize)
    exec "${ENV_PREFIX}/bin/python" summarize.py --run-root "${RUN_ROOT}"
    ;;
  *) exit 2 ;;
esac
