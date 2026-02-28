#!/usr/bin/env bash
# 2-GPU parallel runner: Papaya on GPU0, Cotton on GPU1
# Uses the same minimal args as the given single-run command.
set -euo pipefail

CODE_DIR="/root/autodl-tmp/RootSAM_Pro"
RUN_ROOT="/root/autodl-tmp/runs_rootsam_pro"
LOG_ROOT="/root/autodl-tmp/logs_rootsam_pro"

SUBSET0="Papaya_736x552_DPI150"
SUBSET1="Cotton_736x552_DPI150"   # 如果你真叫 Catton，就改这里

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
cd "${CODE_DIR}"

run_one () {
  local GPU="$1"
  local SUBSET="$2"
  local TAG="$3"

  echo "[$(date '+%F %T')] GPU${GPU} | ${SUBSET} | FULL | ${TAG}"

  CUDA_VISIBLE_DEVICES="${GPU}" python main.py \
    --run_root "${RUN_ROOT}" \
    --subset "${SUBSET}" \
    --ablation FULL \
    --run_tag "${TAG}" \
    --set TRAIN_MODE=SEQUENCE \
    --set TBPTT_CHUNK=4 \
    --set TRAIN_MAX_SEQ_LEN=0 \
    --set AMP=1 \
    --set USE_TASK_UNCERTAINTY=0
}

# Parallel launch
run_one 0 "${SUBSET0}" "abla_full_fair_s0_${SUBSET0}" > "${LOG_ROOT}/full_${SUBSET0}.log" 2>&1 &
pid0=$!

run_one 1 "${SUBSET1}" "abla_full_fair_s0_${SUBSET1}" > "${LOG_ROOT}/full_${SUBSET1}.log" 2>&1 &
pid1=$!

wait $pid0
wait $pid1

echo "✅ Done: 2-GPU subset-parallel finished."
echo "📄 Logs:"
echo "  - ${LOG_ROOT}/full_${SUBSET0}.log"
echo "  - ${LOG_ROOT}/full_${SUBSET1}.log"