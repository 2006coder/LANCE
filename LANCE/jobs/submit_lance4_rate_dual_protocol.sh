#!/usr/bin/env bash
# Submit strict train-only and CatPred data-parity LANCE-4R ten-seed sweeps.

set -euo pipefail

train_script="${TRAIN_SCRIPT:-LANCE/jobs/cluster_train_lance4_rate_2seed_shared_h100.sbatch}"
eval_script="${EVAL_SCRIPT:-LANCE/jobs/cluster_ensemble_eval_lance4_rate_10seed.sbatch}"
account="${SBATCH_ACCOUNT:-${ACCOUNT:-}}"
account_args=()
if [[ -n "${account}" ]]; then
  account_args=(--account="${account}")
fi

read -r -a tracks <<< "${LANCE4_TRACKS:-train trainval}"
seed_bases=(0 2 4 6 8)

for train_split in "${tracks[@]}"; do
  case "${train_split}" in
    train)
      run_prefix="lance4_rate_train_eng_h100x1_2seed_b256_20260716_seed"
      ensemble_name="lance4_rate_train_eng_h100x1_2seed_b256_20260716_seed_ensemble10"
      ;;
    trainval)
      run_prefix="lance4_rate_trainval_eng_h100x1_2seed_b256_20260716_seed"
      ensemble_name="lance4_rate_trainval_eng_h100x1_2seed_b256_20260716_seed_ensemble10"
      ;;
    *)
      echo "Unknown LANCE4 track ${train_split}; expected train or trainval" >&2
      exit 2
      ;;
  esac

  job_ids=()
  for seed_base in "${seed_bases[@]}"; do
    raw_job_id="$(sbatch --parsable "${account_args[@]}" \
      --export=ALL,SEED_BASE="${seed_base}",SEED_COUNT=2,TRAIN_SPLIT="${train_split}",RUN_NAME_PREFIX="${run_prefix}" \
      "${train_script}")"
    job_id="${raw_job_id%%;*}"
    job_ids+=("${job_id}")
    echo "submitted LANCE-4R ${train_split} seeds ${seed_base}-$((seed_base + 1)): ${raw_job_id}"
  done

  dependency="$(IFS=:; echo "${job_ids[*]}")"
  eval_job_id="$(sbatch --parsable "${account_args[@]}" \
    --dependency=afterok:${dependency} \
    --export=ALL,TRAIN_SPLIT="${train_split}",RUN_NAME_PREFIX="${run_prefix}",ENSEMBLE_RUN_NAME="${ensemble_name}" \
    "${eval_script}")"
  echo "submitted LANCE-4R ${train_split} evaluator afterok:${dependency}: ${eval_job_id}"
done
