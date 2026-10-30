#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${PACKAGE_DIR}/configs/cluster.env"
export PROJECT_ROOT ENV_PREFIX DATA_ROOT ESM_CACHE RUN_ROOT
export SLURM_CPU_PARTITION SLURM_ACCOUNT SLURM_PARTITION SLURM_GRES SLURM_CPUS SLURM_MEM SLURM_TIME MAX_PARALLEL
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
exec "${PYTHON_BIN}" "${PACKAGE_DIR}/submit.py" "$@"
