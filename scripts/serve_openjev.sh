#!/usr/bin/env bash
# Bring up openjev (Jev-compatible, DiffusionGemma-26B-A4B NVFP4) on the 24 GB laptop.
# Upstream: https://github.com/razorback16/openjev
#
#   bash scripts/serve_openjev.sh            # docker (recommended first attempt)
#   bash scripts/serve_openjev.sh native     # vllm serve from the pinned fork (see upstream README)
#
# 24 GB is the stated minimum for the NVFP4 checkpoint (~18 GB weights). Knobs that matter on a
# laptop GPU that also drives the display: lower --max-model-len, raise --gpu-memory-utilization,
# and close the browser. Check `nvidia-smi` for what the desktop already uses before starting.
set -euo pipefail
MODE="${1:-docker}"
PORT="${OPENJEV_PORT:-8080}"
MODEL="${OPENJEV_MODEL:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}"
GPU_UTIL="${OPENJEV_GPU_UTIL:-0.92}"
MAX_LEN="${OPENJEV_MAX_LEN:-8192}"     # upstream default 65536 is far more KV cache than we need

MAX_NUM_SEQS="${OPENJEV_MAX_NUM_SEQS:-4}"   # upstream 64. Sampler warmup allocates fp32 logits for
# max_num_seqs x canvas(64) tokens x 262k vocab: 16 -> 1 GiB, which OOMed the 24 GB GPU after KV alloc.
# First start JIT-compiles FlashInfer NVFP4 MoE kernels for sm_120. Without MAX_JOBS, ninja runs
# nproc+2 cicc processes at ~2.4 GB each (24 cores -> ~57 GB) and the host OOM-killer froze the
# desktop. Cap the parallelism, cap the container's RAM, and persist the JIT caches.
JIT_JOBS="${OPENJEV_JIT_JOBS:-4}"
MEM_LIMIT="${OPENJEV_MEM_LIMIT:-40g}"
# 0.1.0 predates image support (the API dropped `images`, and its vLLM pin crashed on image input);
# images arrived in 0.2.0.
IMAGE="${OPENJEV_IMAGE:-razorback16/openjev:0.5.0}"
MAX_IMAGES="${OPENJEV_MAX_IMAGES:-2}"      # per request; upstream 8. We send front+wrist at most

if [ "$MODE" = "docker" ]; then
  mkdir -p ~/.cache/huggingface ~/.cache/openjev/flashinfer ~/.cache/openjev/vllm
  docker run -d --name openjev --gpus all --ipc=host \
    --memory="$MEM_LIMIT" --memory-swap="$MEM_LIMIT" \
    -p 127.0.0.1:${PORT}:8080 \
    -e OPENJEV_MODEL="$MODEL" \
    -e OPENJEV_MAX_MODEL_LEN="$MAX_LEN" \
    -e OPENJEV_GPU_UTIL="$GPU_UTIL" \
    -e OPENJEV_MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    -e OPENJEV_MAX_IMAGES="$MAX_IMAGES" \
    -e MAX_JOBS="$JIT_JOBS" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/openjev/flashinfer:/root/.cache/flashinfer \
    -v ~/.cache/openjev/vllm:/root/.cache/vllm \
    "$IMAGE"
  echo "started container 'openjev' on http://127.0.0.1:${PORT}  (docker logs -f openjev)"
  echo "image=$IMAGE max_images=$MAX_IMAGES max_model_len=$MAX_LEN gpu_util=$GPU_UTIL max_num_seqs=$MAX_NUM_SEQS jit_jobs=$JIT_JOBS mem=$MEM_LIMIT hf_offline=${HF_HUB_OFFLINE:-0}"
  echo "TIP: once weights are cached, HF_HUB_OFFLINE=1 avoids startup failures from flaky Hub connections."
else
  # Native: requires the pinned vLLM fork (razorback16/vllm, branch structured-reads-54309) installed
  # in the current environment. See upstream README for the exact install steps; flags below are theirs.
  vllm serve "$MODEL" --served-model-name dgemma \
    --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 --enable-prefix-caching \
    --async-scheduling --attention-backend TRITON_ATTN \
    --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_LEN" --port 8001 &
  echo "vLLM on :8001 - now start the openjev API layer pointing at it (see upstream README), then:"
  echo "  dlb health --backend openjev && dlb smoke --backend openjev"
fi
