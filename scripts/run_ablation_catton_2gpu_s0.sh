#!/usr/bin/env bash
# 2-GPU parallel ablation runner for AutoDL (one experiment per GPU).
# Tri-Adapter strict chain: SFA_ONLY ⊂ SFA_ASTA ⊂ FULL
set -euo pipefail

CODE_DIR="/root/autodl-tmp/RootSAM_Pro"
RUN_ROOT="/root/autodl-tmp/runs_rootsam_pro"
LOG_ROOT="/root/autodl-tmp/logs_rootsam_pro"

SUBSET="Catton_736x552_DPI150"
SEED=42

# Training defaults (you can adjust here)
EPOCHS=30
BATCH_SIZE=4
LR="1e-4"
AMP=1
USE_TASK_UNCERTAINTY=0

# Fair-loss switch (recommended for strict ablation comparability)
SOIL_PENALTY_ALL_MODES=1
SOIL_TOPK_RATIO="0.03"
SOIL_LAMBDA_MAX="50"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
cd "${CODE_DIR}"

run_one () {
  local GPU="$1"
  local ABL="$2"
  local TAG="$3"
  local TBPTT="$4"

  echo "[$(date '+%F %T')] GPU${GPU} | ${SUBSET} | ${ABL} | ${TAG} | TBPTT_CHUNK=${TBPTT}"

  CUDA_VISIBLE_DEVICES="${GPU}" python main.py \
    --run_root "${RUN_ROOT}" \
    --subset "${SUBSET}" \
    --ablation "${ABL}" \
    --run_tag "${TAG}" \
    --set SEED="${SEED}" \
    --set TRAIN_MODE="SEQUENCE" \
    --set TBPTT_CHUNK="${TBPTT}" \
    --set TRAIN_MAX_SEQ_LEN=0 \
    --set EPOCHS="${EPOCHS}" \
    --set BATCH_SIZE="${BATCH_SIZE}" \
    --set LR="${LR}" \
    --set AMP="${AMP}" \
    --set USE_TASK_UNCERTAINTY="${USE_TASK_UNCERTAINTY}" \
    --set SOIL_PENALTY_ALL_MODES="${SOIL_PENALTY_ALL_MODES}" \
    --set SOIL_TOPK_RATIO="${SOIL_TOPK_RATIO}" \
    --set SOIL_LAMBDA_MAX="${SOIL_LAMBDA_MAX}"
}

# ------------------------------------------------------------------------------
# Group 1 (GPU0 vs GPU1)
#   - SFA_ONLY uses TBPTT_CHUNK=0 (sequence loader; no ASTA runtime)
#   - SFA_ASTA uses TBPTT_CHUNK=4
# ------------------------------------------------------------------------------
run_one 0 SFA_ONLY "abla_sfa_only_s${SEED}" 0 > "${LOG_ROOT}/sfa_only_s${SEED}.log" 2>&1 &
pid0=$!
run_one 1 SFA_ASTA "abla_sfa_asta_s${SEED}" 4 > "${LOG_ROOT}/sfa_asta_s${SEED}.log" 2>&1 &
pid1=$!
wait $pid0
wait $pid1

# ------------------------------------------------------------------------------
# Group 2
#   - FULL uses TBPTT_CHUNK=4
#   - Optional: rerun FULL with AMP=0 for fp32 stability check (commented)
# ------------------------------------------------------------------------------
run_one 0 FULL "abla_full_s${SEED}" 4 > "${LOG_ROOT}/full_s${SEED}.log" 2>&1 &
pid0=$!

# If you want a fp32 sanity run (optional), uncomment:
# AMP_FP32=0
# CUDA_VISIBLE_DEVICES="1" python main.py \
#   --run_root "${RUN_ROOT}" \
#   --subset "${SUBSET}" \
#   --ablation "FULL" \
#   --run_tag "abla_full_fp32_s${SEED}" \
#   --set SEED="${SEED}" \
#   --set TRAIN_MODE="SEQUENCE" \
#   --set TBPTT_CHUNK=4 \
#   --set TRAIN_MAX_SEQ_LEN=0 \
#   --set EPOCHS="${EPOCHS}" \
#   --set BATCH_SIZE="${BATCH_SIZE}" \
#   --set LR="${LR}" \
#   --set AMP="${AMP_FP32}" \
#   --set USE_TASK_UNCERTAINTY="${USE_TASK_UNCERTAINTY}" \
#   --set SOIL_PENALTY_ALL_MODES="${SOIL_PENALTY_ALL_MODES}" \
#   --set SOIL_TOPK_RATIO="${SOIL_TOPK_RATIO}" \
#   --set SOIL_LAMBDA_MAX="${SOIL_LAMBDA_MAX}" \
#   > "${LOG_ROOT}/full_fp32_s${SEED}.log" 2>&1 &
# pid1=$!

# If fp32 run is disabled:
pid1=""
wait $pid0
# if [[ -n "${pid1}" ]]; then wait "${pid1}"; fi

echo "✅ Done: 2-GPU ablation finished."
# echo "➡️  Summarize:"
# echo "python -m tools.collect_ablation_table --runs_root ${RUN_ROOT} --subset ${SUBSET} --prefer test"
