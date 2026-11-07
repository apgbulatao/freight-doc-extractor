FROM python:3.11-slim

# libmagic-ish deps aren't needed here (pymupdf/pdfplumber ship their own
# binaries), but pdfplumber pulls in pillow, which wants a couple of image
# libs at the OS level on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY frontend/ ./frontend/

# Where uploaded originals, rasterized pages, and the SQLite DB live.
# Mounted as a volume in docker-compose / at deploy time so data survives
# container restarts and redeploys.
RUN mkdir -p storage

ENV MODEL_API_BASE=http://localhost:8080/v1
ENV MODEL_NAME=Qwen2-VL-2B-Instruct-Q4_K_M.gguf

EXPOSE 8000

# No --reload: this is the deploy image, not the dev server.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
