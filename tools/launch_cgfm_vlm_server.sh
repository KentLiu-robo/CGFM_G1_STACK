#!/bin/bash
# vLLM OpenAI-compatible server for CGFM's VLM target selection.
#   Model : Qwen2.5-VL-3B-Instruct-AWQ (HF cache, ~3.4 GB)
#   Env   : a Python env with vLLM 0.7.2 + transformers 4.49.0 (set CGFM_VLM_PYTHON; tested with a separate conda env)
#   GPU   : 1 only (GPU 0 keeps CLIP / YOLO-World), capped at CGFM_VLM_GPU_UTIL of its memory
#   Port  : 8101  (OpenAI API at http://127.0.0.1:8101/v1, model name "qwen2.5-vl-3b")
# Logs to /tmp/go2_stack/cgfm_vlm.log when started in the background:
#   setsid bash launch_cgfm_vlm_server.sh > /tmp/go2_stack/cgfm_vlm.log 2>&1 &
# Stop:  pkill -f "[s]erved-model-name qwen2.5-vl-3b"   (the [s] keeps pkill from matching its own shell)
set -e
MODEL=${CGFM_VLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct-AWQ}
PORT=${CGFM_VLM_PORT:-8101}
export CUDA_VISIBLE_DEVICES=${CGFM_VLM_GPU:-1}
export HF_HUB_OFFLINE=1          # weights are already cached; never hit the network at start
# transformers-4.49.0 vs vLLM-0.7.2 name mismatch -> import shim (see vllm_compat/sitecustomize.py)
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vllm_compat${PYTHONPATH:+:$PYTHONPATH}"
exec "${CGFM_VLM_PYTHON:-python}" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen2.5-vl-3b \
    --host 127.0.0.1 --port "$PORT" \
    --quantization awq --dtype float16 \
    --max-model-len 12288 --gpu-memory-utilization ${CGFM_VLM_GPU_UTIL:-0.6} \
    --limit-mm-per-prompt image=32 \
    --disable-log-requests
