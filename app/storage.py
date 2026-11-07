"""
Storage step: persists validated extraction results to a DB, keeping the
original uploaded file on disk for audit/reference.

SQLite for v1 -- zero setup, fits a free-tier CPU VPS. Swapping to Postgres
later is a one-line change: set DATABASE_URL to a postgres:// connection
string (e.g. "postgresql://user:pass@host/dbname") and nothing else in this
file needs to change, since SQLAlchemy abstracts the dialect.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./freight_docs.db")
ORIGINALS_DIR = Path(os.environ.get("ORIGINALS_DIR", "storage/originals"))
ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)

# check_same_thread=False is SQLite-specific and safe here: FastAPI can call
# from different threads for sync endpoints, and we open a fresh session per
# request rather than sharing one across threads.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)


class Base(DeclarativeBase):
    pass


class Document(Base):
    """One row per uploaded file, independent of whether extraction
    succeeded -- keeps a record of failed attempts too, for audit."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_format: Mapped[str] = mapped_column(String(10))  # pdf | docx
    doc_type: Mapped[str] = mapped_column(String(30))  # commercial_invoice | packing_list | bill_of_lading
    extraction_path: Mapped[str] = mapped_column(String(30))  # pdf_text_layer | pdf_vlm_image | docx_text
    original_file_path: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(20))  # extracted | validation_failed
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    extraction: Mapped["Extraction | None"] = relationship(
        back_populates="document", uselist=False, cascade="all, delete-orphan"
    )


class Extraction(Base):
    """The validated, structured result for a successfully-extracted
    document. Stored as JSON since fields differ per doc_type -- querying
    by specific field can use JSON operators later if needed (Postgres
    supports this natively; SQLite has json_extract)."""

    __tablename__ = "extractions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    data: Mapped[dict] = mapped_column(JSON)

    document: Mapped["Document"] = relationship(back_populates="extraction")


def init_db():
    Base.metadata.create_all(engine)


def save_original_file(doc_id: str, filename: str, file_bytes: bytes) -> str:
    """Save the raw uploaded file to disk under its document id, for audit
    -- lets you go back and check what the model actually saw."""
    ext = Path(filename).suffix
    out_path = ORIGINALS_DIR / f"{doc_id}{ext}"
    out_path.write_bytes(file_bytes)
    return str(out_path)


def save_extraction_result(
    *,
    filename: str,
    file_format: str,
    doc_type: str,
    extraction_path: str,
    file_bytes: bytes,
    extracted_data: dict | None,
    status: str,
) -> str:
    """Persist one ingest+extract attempt: the document record always
    gets written (even on validation_failed), and the extraction record
    only exists when status == "extracted"."""
    doc_id = uuid.uuid4().hex[:12]
    original_path = save_original_file(doc_id, filename, file_bytes)

    with Session(engine) as session:
        document = Document(
            id=doc_id,
            filename=filename,
            file_format=file_format,
            doc_type=doc_type,
            extraction_path=extraction_path,
            original_file_path=original_path,
            status=status,
        )
        session.add(document)

        if status == "extracted" and extracted_data is not None:
            extraction = Extraction(
                id=uuid.uuid4().hex[:12],
                document_id=doc_id,
                data=extracted_data,
            )
            session.add(extraction)

        session.commit()

    return doc_id


def get_document(doc_id: str) -> dict | None:
    with Session(engine) as session:
        document = session.get(Document, doc_id)
        if document is None:
            return None
        return {
            "id": document.id,
            "filename": document.filename,
            "file_format": document.file_format,
            "doc_type": document.doc_type,
            "extraction_path": document.extraction_path,
            "status": document.status,
            "uploaded_at": document.uploaded_at.isoformat(),
            "extracted": document.extraction.data if document.extraction else None,
        }


def list_documents(doc_type: str | None = None, limit: int = 50) -> list[dict]:
    with Session(engine) as session:
        stmt = select(Document).order_by(Document.uploaded_at.desc()).limit(limit)
        if doc_type:
            stmt = stmt.where(Document.doc_type == doc_type)
        documents = session.execute(stmt).scalars().all()
        return [
            {
                "id": d.id,
                "filename": d.filename,
                "doc_type": d.doc_type,
                "status": d.status,
                "uploaded_at": d.uploaded_at.isoformat(),
            }
            for d in documents
        ]
