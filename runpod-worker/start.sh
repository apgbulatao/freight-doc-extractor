#!/bin/bash
# RunPod runs this container's CMD once per worker boot (cold start), then
# keeps the process alive across multiple jobs until RunPod scales the
# worker back down. So llama-server only needs to load the model once per
# cold start, not once per request -- same as your local docker-compose
# setup, just started by this script instead of compose.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen2-VL-2B-Instruct-Q4_K_M.gguf}"
MMPROJ_PATH="${MMPROJ_PATH:-/models/mmproj-Qwen2-VL-2B-Instruct-f16.gguf}"

echo "Starting llama-server..."
llama-server \
    -m "$MODEL_PATH" \
    --mmproj "$MMPROJ_PATH" \
    --host 127.0.0.1 \
    --port 8080 \
    -ngl 999 &

# Wait for llama-server to actually be ready before the handler starts
# accepting RunPod jobs -- otherwise the first job on a cold start can
# race the model still loading and fail with a connection error.
echo "Waiting for llama-server to become healthy..."
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
        echo "llama-server is up."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "llama-server did not become healthy in time." >&2
        exit 1
    fi
    sleep 2
done

echo "Starting RunPod handler..."
exec python3 -u handler.py
