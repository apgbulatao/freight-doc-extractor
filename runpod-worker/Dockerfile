# Base: official llama.cpp CUDA server image (same one used in
# docker-compose.yml for local dev), so behavior matches what you already
# tested locally. Check https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp
# for the current tag before building -- this project's image names/tags
# have moved before.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages runpod httpx

WORKDIR /worker

# Model baked into the image: at ~2.3GB combined (2B Q4_K_M + f16 mmproj)
# this is small enough that baking in is simpler than managing a RunPod
# Network Volume, at the cost of a slower image push/cold pull. If you
# later move to a bigger model, switch to a Network Volume + download-on-
# first-boot instead (see the RunPod docs) rather than keep growing this
# image.
COPY models/Qwen2-VL-2B-Instruct-Q4_K_M.gguf /models/Qwen2-VL-2B-Instruct-Q4_K_M.gguf
COPY models/mmproj-Qwen2-VL-2B-Instruct-f16.gguf /models/mmproj-Qwen2-VL-2B-Instruct-f16.gguf

COPY runpod-worker/handler.py .
COPY runpod-worker/start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
