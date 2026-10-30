#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${SCRIPT_DIR}/configs/cluster.env"
set +a
PYTHON="${PYTHON:-${ENV_PREFIX}/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Python not found: ${PYTHON}" >&2
    exit 2
fi
exec "${PYTHON}" "${SCRIPT_DIR}/submit.py" "$@"
