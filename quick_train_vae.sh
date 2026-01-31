#!/usr/bin/env bash
set -euo pipefail

# Quick VAE training script (single node). Adjust paths and flags as needed.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"

DATA_PATH="${DATA_PATH:-/path/to/data}"
DATA_FORMAT="${DATA_FORMAT:-parquet}"   # imagefolder | parquet
VAE_PATH="${VAE_PATH:-/path/to/vae}" 
IMAGE_SIZE="${IMAGE_SIZE:-256}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-1000}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"              # fp32 | fp16 | bf16

TORCHRUN_ARGS=(
  --standalone
  --nproc_per_node="${NPROC_PER_NODE:-1}"
)

torchrun "${TORCHRUN_ARGS[@]}" \
  train_vae.py \
  --data-path "${DATA_PATH}" \
  --data-format "${DATA_FORMAT}" \
  --vae-path "${VAE_PATH}" \
  --image-size "${IMAGE_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --amp-dtype "${AMP_DTYPE}" \
  --log-every 100 \
  --ckpt-every 50000 \
  --save-code \
  --run-notes "quick VAE run"
