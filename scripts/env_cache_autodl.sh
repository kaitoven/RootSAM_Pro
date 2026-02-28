#!/usr/bin/env bash
# Redirect common caches to AutoDL data disk (/root/autodl-tmp) to avoid filling the 30GB system disk.
set -euo pipefail
export XDG_CACHE_HOME=/root/autodl-tmp/.cache
export TORCH_HOME=/root/autodl-tmp/.cache/torch
export HF_HOME=/root/autodl-tmp/.cache/hf
export TRANSFORMERS_CACHE=/root/autodl-tmp/.cache/hf
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME"
echo "[OK] Cache env set:"
echo "  XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "  TORCH_HOME=$TORCH_HOME"
echo "  HF_HOME=$HF_HOME"
