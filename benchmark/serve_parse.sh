#!/usr/bin/env bash
# Nemotron 3.5 Parse on vLLM 0.31 + NVIDIA's plugin (MTP head), model card's serve flags, localhost:8119.
D="$(dirname "$(readlink -f "$0")")"   # the benchmark directory
cd "$D/dl/nemotron-3.5-parse"
CU=$D/venv-nemo/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU PATH=$CU/bin:$PATH LIBRARY_PATH=$CU/lib VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=${GPU:-2} exec "$D/venv-vllm/bin/vllm" serve . --served-model-name ${NAME:-nemotron-parse} \
  --dtype bfloat16 --distributed-executor-backend uni --tensor-parallel-size 1 --max-model-len 16384 \
  --max-num-seqs 192 --max-num-batched-tokens 16384 --no-enable-prefix-caching --limit-mm-per-prompt '{"image":1}' \
  --attention-config '{"flash_attn_version":3}' --trust-remote-code --host 127.0.0.1 --port ${PORT:-8119} \
  --chat-template chat_template.jinja --gpu-memory-utilization ${UTIL:-0.5}
