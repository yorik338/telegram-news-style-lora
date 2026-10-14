#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${VLLM_API_KEY:?Set a private API key in VLLM_API_KEY}"
test -f "$TASK_ROOT/adapter/adapter_model.safetensors"
python -m vllm.entrypoints.openai.api_server \
  --model unsloth/Qwen3-4B-Instruct-2507-bnb-4bit \
  --revision f12db89cd5156e090618dded9b4367f23f8f3b33 \
  --tokenizer-revision f12db89cd5156e090618dded9b4367f23f8f3b33 \
  --chat-template "$TASK_ROOT/inference_chat_template.jinja" \
  --served-model-name telegram-base --host 127.0.0.1 --port 8000 --api-key "$VLLM_API_KEY" \
  --dtype half --quantization bitsandbytes --kv-cache-dtype auto --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" --max-model-len 4096 \
  --max-num-seqs "${MAX_NUM_SEQS:-15}" --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-1024}" \
  --enable-chunked-prefill --enforce-eager --generation-config vllm \
  --enable-lora --max-loras 1 --max-cpu-loras 1 --max-lora-rank 16 \
  --lora-modules "telegram-style-v2=$TASK_ROOT/adapter"
