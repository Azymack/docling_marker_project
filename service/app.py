"""
Docling PDF → text microservice.

Run:
  uvicorn service.app:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import logging
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from docling.document_converter import DocumentConverter
from docling_pipeline import (
    ConversionError,
    ConvertResult,
    PipelineConfig,
    build_converter,
    convert_pdf,
    cuda_info,
    initialize_converter,
)
from service.settings import ServiceSettings

_log = logging.getLogger(__name__)
_settings = ServiceSettings()
_base_config = PipelineConfig.from_env()

_converters: dict[bool, DocumentConverter] = {}
_converter_init_seconds: dict[bool, float] = {}
_init_lock = threading.Lock()
_convert_lock = threading.Lock()
_ready = False


class ResponseFormat(str, Enum):
    text = "text"
    json = "json"


class ConvertMetadata(BaseModel):
    status: str
    pages: int
    convert_seconds: float
    pages_per_second: float | None = None
    warnings: list[str] = Field(default_factory=list)
    ocr: bool
    source_filename: str | None = None


class ConvertJsonResponse(BaseModel):
    text: str
    metadata: ConvertMetadata


def _config_for_ocr(do_ocr: bool) -> PipelineConfig:
    cfg = replace(_base_config, do_ocr=do_ocr)
    if do_ocr:
        cfg.force_backend_text = False
    return cfg


def _ensure_converter(do_ocr: bool) -> DocumentConverter:
    if do_ocr in _converters:
        return _converters[do_ocr]

    with _init_lock:
        if do_ocr in _converters:
            return _converters[do_ocr]

        label = "OCR on" if do_ocr else "OCR off"
        _log.info("Initializing Docling pipeline (%s)…", label)
        config = _config_for_ocr(do_ocr)
        converter = build_converter(config)
        seconds = initialize_converter(converter)
        _converters[do_ocr] = converter
        _converter_init_seconds[do_ocr] = seconds
        _log.info("%s ready in %.2fs", label, seconds)
        return converter


def _response_headers(result: ConvertResult, *, ocr: bool) -> dict[str, str]:
    return {
        "X-Docling-Status": result.status,
        "X-Docling-Pages": str(result.pages),
        "X-Docling-Seconds": str(result.convert_seconds),
        "X-Docling-OCR": "true" if ocr else "false",
    }


def _convert_locked(pdf_path: Path, *, do_ocr: bool) -> ConvertResult:
    converter = _ensure_converter(do_ocr)
    with _convert_lock:
        return convert_pdf(converter, pdf_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready

    logging.basicConfig(level=_settings.log_level.upper())
    logging.getLogger("docling").setLevel(logging.WARNING)
    _log.info("Loading default Docling pipeline (device=%s)…", _base_config.device)

    _ensure_converter(_base_config.do_ocr)
    _ready = True

    info = cuda_info()
    if info.get("available"):
        _log.info("Service ready — GPU %s", info.get("name"))
    else:
        _log.info("Service ready — no CUDA GPU detected")

    yield

    _ready = False
    _converters.clear()
    _converter_init_seconds.clear()


app = FastAPI(
    title="Docling PDF Service",
    description="Convert insurance PDFs to plain text (markdown) for downstream processing.",
    version="1.1.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "conversion", "description": "PDF upload → text (use in your pipeline)"},
        {"name": "health", "description": "Liveness and readiness"},
    ],
)


@app.get("/", tags=["health"])
def root() -> dict:
    """Identify this service and list endpoints (useful if /docs shows unexpected routes)."""
    return {
        "service": "docling_marker_project",
        "title": app.title,
        "version": app.version,
        "docs": "/docs",
        "endpoints": {
            "convert": "POST /v1/convert",
            "ready": "GET /ready",
            "health": "GET /health",
        },
    }


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready", tags=["health"])
def ready() -> dict:
    if not _ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return {
        "status": "ready",
        "loaded_pipelines": {
            "ocr_off": False in _converters,
            "ocr_on": True in _converters,
        },
        "pipeline_init_seconds": _converter_init_seconds,
        "default_config": {
            "device": _base_config.device,
            "do_ocr_default": _base_config.do_ocr,
            "do_table_structure": _base_config.do_table_structure,
            "table_mode": _base_config.table_mode,
            "force_backend_text": _base_config.force_backend_text,
            "layout_batch_size": _base_config.layout_batch_size,
        },
        "cuda": cuda_info(),
    }


@app.post(
    "/v1/convert",
    tags=["conversion"],
    summary="Convert PDF to text",
    operation_id="convert_pdf",
    responses={
        200: {
            "description": "Plain text body (default) or JSON with text + metadata",
            "content": {
                "text/plain": {},
                "application/json": {},
            },
        },
    },
)
async def convert(
    file: Annotated[UploadFile, File(description="PDF file to convert")],
    ocr: Annotated[
        bool,
        Query(description="Enable OCR for scanned/image PDFs (slower)"),
    ] = False,
    format: Annotated[
        ResponseFormat,
        Query(description="`text`: raw string body; `json`: text + metadata"),
    ] = ResponseFormat.text,
):
    """
  Upload a PDF and receive extracted content as **inline text** (not a file download).

  - **ocr=false** (default): text-selectable PDFs — uses embedded text, fastest.
  - **ocr=true**: runs OCR — use for scanned documents.
  - **format=text** (default): response body is the markdown string (`text/plain`).
  - **format=json**: `{"text": "...", "metadata": {...}}` for the next pipeline stage.
    """
    if not _ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    max_bytes = _settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {_settings.max_upload_mb} MB limit",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    suffix = Path(file.filename).suffix or ".pdf"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            result = _convert_locked(tmp_path, do_ocr=ocr)
        except ConversionError as exc:
            raise HTTPException(
                status_code=422,
                detail={"message": str(exc), "status": exc.status, "pages": exc.pages},
            ) from exc

        meta = ConvertMetadata(
            status=result.status,
            pages=result.pages,
            convert_seconds=result.convert_seconds,
            pages_per_second=result.pages_per_second,
            warnings=result.warnings,
            ocr=ocr,
            source_filename=file.filename,
        )

        if format == ResponseFormat.json:
            return ConvertJsonResponse(text=result.markdown, metadata=meta)

        return PlainTextResponse(
            content=result.markdown,
            media_type="text/plain; charset=utf-8",
            headers=_response_headers(result, ocr=ocr),
        )

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
