#!/usr/bin/env bash
# Create the three OCR environments with the package versions used for the published results (env/*.txt are
# `uv pip freeze` of the benchmark venvs). Set TORCH_CUDA_ARCH_LIST for your GPU (9.0 Hopper, 12.0 Blackwell).
# Usage: bash env/setup_envs.sh   Log: stdout
set -eu
cd "$(dirname "$(readlink -f "$0")")/.."; export PATH=$HOME/.local/bin:$PATH
TORCH=https://download.pytorch.org/whl/cu130
PADDLE=https://www.paddlepaddle.org.cn/packages/stable/cu129/
say(){ echo "[$(date -u +%FT%TZ)] $*"; }

say "venv-nemo"
uv venv -q -p 3.12.13 venv-nemo
grep -v '^nemotron-ocr @' env/nemo.txt > env/nemo.req
uv pip install -q --python venv-nemo/bin/python -r env/nemo.req --extra-index-url $TORCH --index-strategy unsafe-best-match
CU=$PWD/venv-nemo/lib/python3.12/site-packages/nvidia/cu13
ln -sf libcudart.so.13 $CU/lib/libcudart.so
say "building Nemotron OCR v2 extension for sm_120"
(cd build/nemotron-ocr && rm -rf build && PATH=$CU/bin:$PATH CUDA_HOME=$CU LIBRARY_PATH=$CU/lib TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0} \
  BUILD_CPP_FORCE=1 MAX_JOBS=32 VIRTUAL_ENV=$PWD/../../venv-nemo uv pip install -q --no-build-isolation .)

say "venv-paddle"
uv venv -q -p 3.12.13 venv-paddle
uv pip install -q --python venv-paddle/bin/python -r env/paddle.txt --extra-index-url $PADDLE --index-strategy unsafe-best-match

say "venv-vllm"
uv venv -q -p 3.12.13 venv-vllm
grep -v '^nemotron-parse @' env/vllm.txt > env/vllm.req
uv pip install -q --python venv-vllm/bin/python -r env/vllm.req --extra-index-url $TORCH --index-strategy unsafe-best-match
uv pip install -q --python venv-vllm/bin/python --no-deps ./build/parse35

say "versions"
venv-nemo/bin/python -c "import torch,nemotron_ocr;print('nemo torch',torch.__version__, torch.cuda.get_device_capability(0))"
venv-paddle/bin/python -c "import paddle;print('paddle',paddle.__version__,paddle.version.cuda()); paddle.utils.run_check()" 2>&1 | tail -2
venv-vllm/bin/python -c "import vllm,torch,nemotron_parse;print('vllm',vllm.__version__,'torch',torch.__version__)"
say "SETUP DONE"
