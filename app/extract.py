"""
Extraction step: turns normalized ingest output (page images or raw text)
into structured JSON matching one of the schemas in schemas.py.

Two paths into the model:
  - text prompt  -> DOCX text, or a PDF with a usable native text layer
  - image (VLM)  -> PDF pages with no usable text layer (scans, handwriting)

Both build the same OpenAI-style payload. Where that payload actually gets
sent is a separate concern, controlled by MODEL_BACKEND:
  - "openai" (default) -> POST straight to {MODEL_API_BASE}/chat/completions.
    Used for local dev and docker-compose, talking directly to llama-server.
  - "runpod"            -> POST to RunPod's job-queue API instead, since
    RunPod Serverless doesn't expose an arbitrary HTTP path -- it hands
    your container one job at a time via a handler. The RunPod worker
    image (runpod-worker/) proxies that job straight through to its own
    local llama-server, so the payload shape and response parsing here
    are identical either way -- only the transport differs.
"""

import base64
import json
import os
import time
from pathlib import Path

import httpx
import pdfplumber
from pydantic import ValidationError

from app.schemas import SCHEMA_BY_DOC_TYPE, DocType, schema_json_hint

MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "openai")  # "openai" | "runpod"
MODEL_API_BASE = os.environ.get("MODEL_API_BASE", "http://localhost:8080/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2-vl")
REQUEST_TIMEOUT_S = 120

RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_POLL_INTERVAL_S = 2
RUNPOD_POLL_TIMEOUT_S = 180

# Below this many characters of native text per page, treat the PDF as
# "no usable text layer" (e.g. scanned/handwritten) and fall back to the
# VLM image path instead of the cheaper text-prompt path.
MIN_CHARS_PER_PAGE_FOR_TEXT_PATH = 40


def pdf_has_text_layer(pdf_bytes: bytes) -> tuple[bool, str]:
    """Check whether a PDF has an extractable native text layer.

    Returns (has_text, extracted_text). A scanned or handwritten PDF will
    have little to no extractable text even though it "has pages".

    """
    import io

    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)

    full_text = "\n".join(text_parts)
    avg_chars_per_page = len(full_text) / max(len(text_parts), 1)
    return avg_chars_per_page >= MIN_CHARS_PER_PAGE_FOR_TEXT_PATH, full_text


FIELD_SPECIFIC_HINTS: dict[str, str] = {
    "incoterm": (
        "The 'incoterm' field must be exactly one of these 3-letter codes: "
        "EXW, FCA, FAS, FOB, CFR, CIF, CPT, CIP, DAP, DPU, DDP. "
        "Do not put a field label (like 'B/L / AWB No.' or any other document "
        "heading) into this field. If no valid Incoterm code is visible on the "
        "document, use null."
    ),
}


def _build_prompt(doc_type: DocType) -> str:
    schema = SCHEMA_BY_DOC_TYPE[doc_type]
    relevant_hints = [
        hint for field, hint in FIELD_SPECIFIC_HINTS.items() if field in schema.model_fields
    ]
    hints_block = ("\n\nField-specific rules:\n- " + "\n- ".join(relevant_hints)) if relevant_hints else ""

    return (
        f"Extract fields from this {doc_type.replace('_', ' ')} into JSON "
        f"matching exactly this shape (use null for missing fields, do not "
        f"invent values, respond with ONLY the JSON object and nothing else). "
        f"Read field VALUES only -- never copy a document's field LABELS or "
        f"headings into a field if the actual value isn't present or isn't "
        f"the right kind of data for that field; use null instead."
        f"{hints_block}\n\n"
        f"{schema_json_hint(doc_type)}"
    )


def _call_model(messages: list[dict]) -> dict:
    """Build the OpenAI-style payload, send it via whichever backend is
    configured, and parse the JSON object out of the response content."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0,  # deterministic extraction, not creative generation
        "max_tokens": 2000,
    }

    if MODEL_BACKEND == "runpod":
        response_json = _call_model_via_runpod(payload)
    else:
        response_json = _call_model_via_openai(payload)

    content = response_json["choices"][0]["message"]["content"]

    # Models sometimes wrap JSON in ```json fences despite instructions.
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.split("\n", 1)[1] if "\n" in content else content
        content = content.rsplit("```", 1)[0]

    return json.loads(content)


def _call_model_via_openai(payload: dict) -> dict:
    """Direct call to an OpenAI-compatible /chat/completions endpoint --
    local llama-server today, or anything else exposing that same API."""
    resp = httpx.post(
        f"{MODEL_API_BASE}/chat/completions",
        json=payload,
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def _call_model_via_runpod(payload: dict) -> dict:
    """Call via RunPod Serverless's job-queue API instead of a direct HTTP
    path. /runsync blocks and returns the result inline if the job finishes
    within RunPod's own budget; on a cold start (worker still spinning up)
    it can come back still IN_QUEUE/IN_PROGRESS, in which case we fall back
    to polling /status ourselves until the result is ready or we time out.
    """
    if not RUNPOD_ENDPOINT_ID or not RUNPOD_API_KEY:
        raise RuntimeError(
            "MODEL_BACKEND=runpod requires RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY to be set."
        )

    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    resp = httpx.post(
        f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync",
        headers=headers,
        json={"input": payload},
        timeout=RUNPOD_POLL_TIMEOUT_S,
    )
    resp.raise_for_status()
    result = resp.json()

    status = result.get("status")
    if status == "COMPLETED":
        return result["output"]
    if status in ("IN_QUEUE", "IN_PROGRESS"):
        return _poll_runpod_status(result["id"], headers)
    raise RuntimeError(f"RunPod job did not complete successfully: {result}")


def _poll_runpod_status(job_id: str, headers: dict) -> dict:
    status_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
    deadline = time.time() + RUNPOD_POLL_TIMEOUT_S
    while time.time() < deadline:
        resp = httpx.get(status_url, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        status = result.get("status")
        if status == "COMPLETED":
            return result["output"]
        if status == "FAILED":
            raise RuntimeError(f"RunPod job {job_id} failed: {result}")
        time.sleep(RUNPOD_POLL_INTERVAL_S)
    raise TimeoutError(
        f"RunPod job {job_id} did not complete within {RUNPOD_POLL_TIMEOUT_S}s "
        f"(likely an unusually slow cold start)"
    )


def extract_from_text(text: str, doc_type: DocType):
    """Text-prompt extraction path: DOCX content or a PDF with a native
    text layer. Cheaper and more reliable than the VLM path -- use it
    whenever the source isn't a scan."""
    messages = [
        {"role": "system", "content": "You are a precise data-extraction assistant for freight documents."},
        {"role": "user", "content": f"{_build_prompt(doc_type)}\n\nDocument text:\n{text}"},
    ]
    raw = _call_model(messages)
    return _validate(raw, doc_type)


def extract_from_images(image_paths: list[Path], doc_type: DocType):
    """VLM extraction path: page images with no usable text layer (scans,
    handwriting). Sends all pages of one document in a single call so the
    model can reason across pages if fields span more than one."""
    content = [{"type": "text", "text": _build_prompt(doc_type)}]
    for path in image_paths:
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )

    messages = [
        {"role": "system", "content": "You are a precise data-extraction assistant for freight documents."},
        {"role": "user", "content": content},
    ]
    raw = _call_model(messages)
    return _validate(raw, doc_type)


def _validate(raw: dict, doc_type: DocType):
    """Validate model output against the schema. Raises pydantic
    ValidationError on bad data -- caller decides whether to retry,
    flag for manual review, or reject."""
    schema = SCHEMA_BY_DOC_TYPE[doc_type]
    try:
        return schema.model_validate(raw)
    except ValidationError as e:
        # Re-raise with the raw payload attached so a caller/log can see
        # exactly what the model returned before validation stripped it.
        raise ValidationError.from_exception_data(
            title=f"{doc_type} extraction failed validation",
            line_errors=e.errors(),
        ) from e
