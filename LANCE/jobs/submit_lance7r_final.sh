#!/usr/bin/env bash
# Submit the frozen ten-member trainval final and its one-shot benchmark.

set -euo pipefail

RUN_STAMP="${RUN_STAMP:-20260724}"
SCREEN_STAMP="${SCREEN_STAMP:-20260724r1}"
CALIBRATION_JOB_ID="${CALIBRATION_JOB_ID:-}"
RUN_PREFIX="${RUN_PREFIX:-lance7r_exact_nca_trainval_${RUN_STAMP}_seed}"
ENSEMBLE_NAME="${ENSEMBLE_NAME:-lance7r_exact_nca_trainval_${RUN_STAMP}_ensemble10}"

dependency_args=()
if [[ -n "${CALIBRATION_JOB_ID}" ]]; then
  dependency_args=(--dependency="afterok:${CALIBRATION_JOB_ID}")
fi

group_a=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --export=ALL,RUN_STAMP="${RUN_STAMP}",SCREEN_STAMP="${SCREEN_STAMP}",SEED_BASE=0,SEED_COUNT=5,RUN_PREFIX="${RUN_PREFIX}" \
  LANCE/jobs/cluster_finalize_lance7r_trainval_h100.sbatch)
group_b=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --export=ALL,RUN_STAMP="${RUN_STAMP}",SCREEN_STAMP="${SCREEN_STAMP}",SEED_BASE=5,SEED_COUNT=5,RUN_PREFIX="${RUN_PREFIX}" \
  LANCE/jobs/cluster_finalize_lance7r_trainval_h100.sbatch)

evaluation=$(sbatch --parsable \
  --dependency="afterok:${group_a}:${group_b}" \
  --export=ALL,RUN_STAMP="${RUN_STAMP}",RUN_PREFIX="${RUN_PREFIX}",ENSEMBLE_NAME="${ENSEMBLE_NAME}" \
  LANCE/jobs/cluster_evaluate_lance7r_trainval_10seed.sbatch)

printf 'final_seeds_0_4=%s\n' "${group_a}"
printf 'final_seeds_5_9=%s\n' "${group_b}"
printf 'benchmark=%s\n' "${evaluation}"
