# Freight Document Extractor

Upload a freight document (PDF or DOCX — commercial invoice, packing list,
or bill of lading) and get back structured, validated JSON: shipper/
consignee, weights, Incoterms, line items, and doc-type-specific fields.
Built as a portfolio piece to demonstrate deploying and operating an AI
system end-to-end, not just calling a model.

## Why this exists

I spent a year as a Customer Associate in cargo/freight forwarding before
moving into AI Engineering, and this project sits deliberately at that
intersection: it's a small, real MLOps pipeline (ingest → route → extract
→ validate → store → serve) built around a domain I know well enough to
judge whether the output is actually right, not just well-formatted.

## Architecture

```
                    ┌─────────────┐
   upload (PDF/DOCX)│   FastAPI   │
   ─────────────────▶   app       │
                    │ (CPU VPS)   │
                    └──────┬──────┘
                           │ routes by file type / text layer
              ┌────────────┴────────────┐
              │                         │
      native text available     no text layer
      (DOCX, digital PDF)       (scanned PDF)
              │                         │
       text-prompt path          rasterize → images
              │                         │
              └────────────┬────────────┘
                            ▼
                 OpenAI-compatible chat API
                            │
              ┌─────────────┴─────────────┐
              │                           │
        local dev:                  production:
     llama-server on              RunPod Serverless
     your own GPU box          (scale-to-zero GPU worker
                                running the same llama-server
                                     internally)
                            │
                            ▼
                pydantic schema validation
              (rejects bad output -- never
               silently written to the DB)
                            │
                            ▼
                  SQLite (Postgres-ready)
              original files kept for audit
```

The FastAPI app and the VLM are deliberately split across two different
compute types: the app itself is cheap and CPU-only, so it runs on a
free/low-cost VPS; the VLM needs a GPU but only for the seconds it takes to
run inference, so it runs on RunPod Serverless and scales to zero between
requests instead of idling on a rented GPU.

## What's built

- **Ingest** — FastAPI endpoint accepts PDF/DOCX, detects type, checks
  size/emptiness
- **Routing** — DOCX → text parse (including table cells); PDF checked via
  `pdfplumber` for a native text layer (→ cheap text-prompt path) vs.
  scanned/no text layer (→ rasterized to PNG for the VLM)
- **Extraction** — self-hosted Qwen2-VL-2B-Instruct (Q4_K_M GGUF) via
  llama.cpp's `llama-server`, against pydantic schemas for three doc types
- **Validation** — schemas enforce real business rules (valid Incoterm
  codes, non-negative weight, no field-label text masquerading as a value);
  bad output → HTTP 422, never silently written to the DB
- **Storage** — SQLite via SQLAlchemy (one-line swap to Postgres), original
  files kept on disk for audit, both successful and failed attempts
  persisted
- **Frontend** — single HTML file, served same-origin by FastAPI
- **Rate limiting** — `/ingest` capped per-IP, defense-in-depth against
  cost abuse on metered GPU compute
- **Two deployable backends for the VLM call** — direct HTTP to a local
  `llama-server` for dev, or RunPod Serverless's job-queue API for
  production, switched with one environment variable (`MODEL_BACKEND`)

## Running it locally

```bash
docker compose up --build
```

Puts your GGUF model files in `./models/` first — see
[`docker-compose.yml`](./docker-compose.yml) for expected filenames. Then
open `http://localhost:8000`.

## Deploying it for real

See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for the full RunPod Serverless +
VPS walkthrough.

## Known limitations

- No formal accuracy evaluation has been run yet — this has been tested
  against real documents during development, not benchmarked at scale.
- Not yet tested against handwritten documents. The routing and VLM path
  exist for scanned/no-text-layer PDFs, but a 2B, further-quantized model
  is a materially harder starting point for handwriting than for printed
  text, and this hasn't been validated.
- `doc_type` is provided by the caller; there's no auto-classification
  step yet.

## Tech stack

FastAPI, pymupdf, python-docx, pdfplumber, pydantic, SQLAlchemy, slowapi,
llama.cpp / llama-server, Qwen2-VL-2B-Instruct, RunPod Serverless, vanilla
HTML/JS frontend.
