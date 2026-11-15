#!/usr/bin/env bash
# Gemma 4 as an OCR engine on vLLM 0.31 (venv-vllm), BF16, one GPU, localhost only.
# Image budget 1,120 visual tokens (Gemma 4 model card: use the highest budget for OCR / document parsing; default 280).
# Usage: GPU=<n> PORT=<p> serve_gemma.sh <weights dir> <served name>
D="$(dirname "$(readlink -f "$0")")"
CU=$D/venv-nemo/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU PATH=$CU/bin:$PATH LIBRARY_PATH=$CU/lib VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=${GPU:?set GPU} exec "$D/venv-vllm/bin/vllm" serve "$1" --served-model-name "$2" \
  --dtype bfloat16 --max-model-len 32768 --limit-mm-per-prompt '{"image":1,"audio":0,"video":0}' \
  --mm-processor-kwargs '{"max_soft_tokens":1120}' --reasoning-parser gemma4 \
  --gpu-memory-utilization 0.90 --host 127.0.0.1 --port ${PORT:?set PORT}
