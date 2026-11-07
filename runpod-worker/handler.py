"""
RunPod Serverless entrypoint for the VLM.

RunPod Serverless doesn't expose your container on a plain HTTP port --
it hands your process one job at a time via `runpod.serverless.start`,
and expects a return value back. This handler's only job is to be a thin
proxy: take the job's `input` (which is exactly the OpenAI-style payload
extract.py already builds -- model/messages/temperature/max_tokens),
forward it to llama-server running on localhost inside this same
container, and hand back its response untouched.

This means extract.py's payload-building logic (_build_prompt, message
shape, etc.) doesn't need to know or care whether it's talking to a local
llama-server or to RunPod -- only the transport layer (how the payload
gets there and back) differs, and that split lives in extract.py's
_call_model_via_runpod / _call_model_via_openai.
"""

import os

import httpx
import runpod

LLAMA_SERVER_URL = os.environ.get(
    "LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1/chat/completions"
)
# Generous timeout: cold-start-adjacent first calls on a freshly spun-up
# worker can be slow even after llama-server itself reports healthy.
REQUEST_TIMEOUT_S = 180


def handler(job):
    payload = job["input"]
    resp = httpx.post(LLAMA_SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


runpod.serverless.start({"handler": handler})
