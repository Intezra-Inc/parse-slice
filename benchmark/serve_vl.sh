#!/usr/bin/env bash
# PaddleOCR-VL-1.6 on vLLM 0.31 (Paddle's recommended backend), GPU 2, localhost:8118.
cd "$(dirname "$(readlink -f "$0")")"   # the benchmark directory
CU=$PWD/venv-nemo/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU PATH=$CU/bin:$PATH LIBRARY_PATH=$CU/lib VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=${GPU:-2} exec venv-vllm/bin/vllm serve dl/paddleocr-vl-1.6 --served-model-name PaddleOCR-VL-1.6-0.9B \
  --trust-remote-code --host 127.0.0.1 --port ${PORT:-8118} --max-num-batched-tokens 16384 --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 --trust-request-mm-kwargs --gpu-memory-utilization ${UTIL:-0.5}
