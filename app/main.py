"""
Ingest + extraction + storage for the freight document pipeline.

Accepts an uploaded PDF or DOCX file, detects its type, normalizes it, and
routes it to the correct extraction path:
  - DOCX                        -> text-prompt extraction
  - PDF with native text layer  -> text-prompt extraction (cheaper, more reliable)
  - PDF with no text layer      -> VLM image extraction (scans, handwriting)

Every attempt (success or validation failure) is persisted: the original
file is kept for audit, and successful extractions are stored as structured,
queryable records.
"""

import io
import os
from pathlib import Path

import httpx
import pymupdf as fitz  # PyMuPDF
from docx import Document
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import storage
from app.extract import extract_from_images, extract_from_text, pdf_has_text_layer
from app.schemas import DocType

app = FastAPI(title="Freight Doc Extractor")

# Optional shared-secret gate on the data-bearing endpoints (/ingest,
# /documents*). If API_KEY isn't set, this is a no-op -- local dev and
# docker-compose keep working exactly as before with no key required.
# Once deployed publicly, set API_KEY so uploaded documents (real shipper/
# consignee/invoice data) aren't readable by anyone who finds the URL.
API_KEY = os.environ.get("API_KEY")


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    if API_KEY is None:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")

# Rate limiting: /ingest is the expensive endpoint (triggers GPU inference
# on a per-second-billed cloud instance once deployed), so it gets a tight
# per-IP cap. This is a cost-control measure, not just abuse prevention --
# an unbounded public endpoint on metered GPU compute is a real bill risk.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS is no longer required for the bundled frontend (served same-origin
# below), but left permissive here in case you call this API from a
# separately-hosted frontend or a tool like Postman during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/")
async def serve_frontend():
    """Serve the bundled frontend at the API's own root, so the whole app
    (API + UI) runs from a single `uvicorn` process on one port."""
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found.")
    return FileResponse(FRONTEND_PATH)


@app.on_event("startup")
def on_startup():
    storage.init_db()


# Where rasterized page images land. Separate from storage.ORIGINALS_DIR
# (which holds the raw uploaded file) since these are intermediate
# artifacts, not the source of record.
PAGES_DIR = Path("storage/pages")
PAGES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

MAX_FILE_SIZE_MB = 25


def detect_doc_type(filename: str, content_type: str | None) -> str:
    """Determine doc type from content-type first, falling back to extension.

    Browsers/clients don't always set content_type correctly for DOCX, so we
    don't trust it alone.
    """
    if content_type in ALLOWED_CONTENT_TYPES:
        return ALLOWED_CONTENT_TYPES[content_type]

    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"

    raise HTTPException(
        status_code=415,
        detail=f"Unsupported file type for '{filename}'. Only PDF and DOCX are accepted.",
    )


def rasterize_pdf(file_bytes: bytes, dpi: int = 200) -> list[Path]:
    """Render each PDF page to a PNG image. Returns saved image paths.

    dpi=200 balances legibility of small print (HS codes, weights) against
    file size / VLM input cost. Bump to 300 if extraction accuracy suffers
    on dense docs.
    """
    import uuid

    call_id = uuid.uuid4().hex[:12]
    out_dir = PAGES_DIR / call_id
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    zoom = dpi / 72  # PDF default is 72 dpi
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        if pdf.page_count == 0:
            raise HTTPException(status_code=422, detail="PDF has no pages.")
        for page_index, page in enumerate(pdf):
            pix = page.get_pixmap(matrix=matrix)
            page_path = out_dir / f"page_{page_index + 1:03d}.png"
            pix.save(str(page_path))
            image_paths.append(page_path)

    return image_paths


def extract_docx_text(file_bytes: bytes) -> str:
    """Pull plain text from a DOCX: paragraphs + table cells.

    Freight docs (packing lists especially) often carry key fields in
    tables, so we can't just read paragraphs.
    """
    document = Document(io.BytesIO(file_bytes))

    parts = [p.text for p in document.paragraphs if p.text.strip()]

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    text = "\n".join(parts)
    if not text.strip():
        raise HTTPException(status_code=422, detail="DOCX contained no extractable text.")
    return text


@app.post("/ingest")
@limiter.limit("10/hour")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    doc_type: DocType = Form(..., description="commercial_invoice | packing_list | bill_of_lading"),
    _: None = Depends(require_api_key),
):
    """Ingest + extract in one call. doc_type classification is manual for
    v1 (user/caller specifies it) -- auto-classifying doc type is a
    reasonable later addition, not required to prove the pipeline out.

    Rate-limited to 10/hour per IP -- this endpoint triggers model inference,
    which is metered GPU cost once deployed, so it's capped independent of
    RunPod's own worker/spend limits (defense in depth, not a replacement)."""
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    file_format = detect_doc_type(file.filename, file.content_type)

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File is {size_mb:.1f}MB; max allowed is {MAX_FILE_SIZE_MB}MB.",
        )
    if size_mb == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    extraction_path = "unknown"
    try:
        if file_format == "pdf":
            has_text, native_text = pdf_has_text_layer(file_bytes)
            if has_text:
                result = extract_from_text(native_text, doc_type)
                extraction_path = "pdf_text_layer"
            else:
                page_images = rasterize_pdf(file_bytes)
                result = extract_from_images(page_images, doc_type)
                extraction_path = "pdf_vlm_image"
        else:
            # file_format == "docx"
            native_text = extract_docx_text(file_bytes)
            result = extract_from_text(native_text, doc_type)
            extraction_path = "docx_text"
    except ValidationError as e:
        # e.errors() can include a 'ctx' dict holding the raw exception object
        # (not JSON-serializable) -- strip it, message text already has the detail.
        clean_errors = [
            {k: v for k, v in err.items() if k != "ctx"} for err in e.errors()
        ]
        # Persist the failed attempt too -- keeps an audit trail of what was
        # uploaded and why extraction didn't pass validation, without a
        # bogus extraction record.
        doc_id = storage.save_extraction_result(
            filename=file.filename,
            file_format=file_format,
            doc_type=doc_type,
            extraction_path=extraction_path,
            file_bytes=file_bytes,
            extracted_data=None,
            status="validation_failed",
        )
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Model output failed schema validation.",
                "errors": clean_errors,
                "document_id": doc_id,
            },
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Model API call failed: {e}")

    extracted = result.model_dump()
    doc_id = storage.save_extraction_result(
        filename=file.filename,
        file_format=file_format,
        doc_type=doc_type,
        extraction_path=extraction_path,
        file_bytes=file_bytes,
        extracted_data=extracted,
        status="extracted",
    )

    return JSONResponse(
        {
            "document_id": doc_id,
            "filename": file.filename,
            "file_format": file_format,
            "doc_type": doc_type,
            "extraction_path": extraction_path,
            "extracted": extracted,
        }
    )


@app.get("/documents")
async def list_documents(
    doc_type: DocType | None = None, limit: int = 50, _: None = Depends(require_api_key)
):
    """List stored documents, newest first. Optionally filter by doc_type."""
    return storage.list_documents(doc_type=doc_type, limit=limit)


@app.get("/documents/{document_id}")
async def get_document(document_id: str, _: None = Depends(require_api_key)):
    """Fetch one document's full record, including its extracted data if
    extraction succeeded."""
    document = storage.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"No document found with id '{document_id}'.")
    return document


@app.get("/health")
async def health():
    return {"status": "ok"}
