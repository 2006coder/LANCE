#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${1:?Package directory is required}"
shift
export PATH="${ENV_PREFIX}/bin:${PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false DGLBACKEND=pytorch DGL_GRAPHBOLT_SKIP_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "${SCRIPT_DIR}"
PYTHON="${ENV_PREFIX}/bin/python"
case "$1" in
    preflight)
        exec "${PYTHON}" preflight.py --run-root "${RUN_ROOT}" --data-root "${DATA_ROOT}" --esm-cache "${ESM_CACHE}"
        ;;
    train)
        exec "${PYTHON}" run.py train --variant "$2" --seed "${SLURM_ARRAY_TASK_ID:?}" --run-root "${RUN_ROOT}" --data-root "${DATA_ROOT}" --esm-cache "${ESM_CACHE}"
        ;;
    evaluate)
        exec "${PYTHON}" run.py evaluate --variant "$2" --run-root "${RUN_ROOT}" --data-root "${DATA_ROOT}" --esm-cache "${ESM_CACHE}"
        ;;
    summarize)
        exec "${PYTHON}" summarize.py --run-root "${RUN_ROOT}"
        ;;
    *) exit 2 ;;
esac
