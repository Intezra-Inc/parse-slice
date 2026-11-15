#!/usr/bin/env bash
# Download NVIDIA's Nemotron 3.5 Parse repository (weights + nemotron_parse package, ~2.4 GB) into ./model.
# Its files are licensed by NVIDIA under OpenMDW-1.1 (see model/LICENSE); they are not part of this repository.
set -eu
cd "$(dirname "$0")/.."
command -v hf >/dev/null || python3 -m pip install --user "huggingface_hub>=1.24"
hf download nvidia/Nemotron-3.5-Parse --local-dir model
echo "model/ ready: $(du -sh model | cut -f1)"
