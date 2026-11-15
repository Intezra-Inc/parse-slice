# Nemotron 3.5 Parse server + the parse-slice client in one image.
# Based on NVIDIA's own Dockerfile from the model repository, with vLLM 0.32.0 instead of NVIDIA's 0.26.0 (both within
# the package's supported range >=0.25,<0.33). Benchmarked on 0.31.0 and 0.32.0 with identical accuracy and speed;
# build on 0.31.0 with: docker compose build --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.31.0
# Build context = this repository, with the model repository downloaded to ./model (scripts/fetch_model.sh).
ARG VLLM_IMAGE=vllm/vllm-openai:v0.32.0
FROM ${VLLM_IMAGE}
WORKDIR /workspace
# NVIDIA's model repository: weights, chat template, the nemotron_parse package and its vLLM plugin.
COPY model /workspace
COPY docker/constraints.txt /tmp/constraints.txt
# (no `pip check`: the vllm/vllm-openai:v0.31.0 base image already fails it, on opentelemetry-sdk and nvidia-nccl-cu13)
RUN python3 -m pip install --no-cache-dir -c /tmp/constraints.txt "/workspace[client]"
COPY pyproject.toml README.md LICENSE /opt/parse-slice/
COPY parse_slice /opt/parse-slice/parse_slice
RUN python3 -m pip install --no-cache-dir -c /tmp/constraints.txt "/opt/parse-slice[pdf]" \
 && python3 -c "import vllm, nemotron_parse, vllm_tied_patch.plugin, parse_slice, pypdfium2; print('vllm', vllm.__version__, 'parse-slice', parse_slice.__version__)"
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
# NVIDIA's model-card serving flags.
CMD ["--model", "/workspace", "--served-model-name", "nemotron-parse", "--trust-remote-code", "--dtype", "bfloat16", \
     "--distributed-executor-backend", "uni", "--tensor-parallel-size", "1", "--max-model-len", "16384", \
     "--max-num-seqs", "192", "--max-num-batched-tokens", "16384", "--no-enable-prefix-caching", \
     "--attention-config", "{\"flash_attn_version\":3}", "--limit-mm-per-prompt", "{\"image\":1}", \
     "--chat-template", "/workspace/chat_template.jinja", "--host", "0.0.0.0", "--port", "8000"]
