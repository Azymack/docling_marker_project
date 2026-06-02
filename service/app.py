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

import traceback

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from docling.document_converter import DocumentConverter
from docling_pipeline import (
    ConversionError,
    ConvertResult,
    OcrMode,
    PipelineConfig,
    build_converter,
    convert_pdf,
    cuda_info,
    initialize_converter,
    parse_ocr_mode,
)
from service.settings import ServiceSettings

_log = logging.getLogger(__name__)
# Bump when deploying; shown on GET / for version checks
SERVICE_BUILD = "2026-06-02-furniture-export-v3"
_settings = ServiceSettings()
_base_config = PipelineConfig.from_env()

_converters: dict[OcrMode, DocumentConverter] = {}
_converter_init_seconds: dict[OcrMode, float] = {}
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
    ocr_mode: str
    include_page_furniture: bool = True
    source_filename: str | None = None


class ConvertJsonResponse(BaseModel):
    text: str
    metadata: ConvertMetadata


def _config_for_mode(ocr_mode: OcrMode) -> PipelineConfig:
    cfg = replace(_base_config, ocr_mode=ocr_mode)
    if ocr_mode != "off":
        cfg.force_backend_text = False
    return cfg


def _ensure_converter(ocr_mode: OcrMode) -> DocumentConverter:
    if ocr_mode in _converters:
        return _converters[ocr_mode]

    with _init_lock:
        if ocr_mode in _converters:
            return _converters[ocr_mode]

        _log.info("Initializing Docling pipeline (ocr_mode=%s)…", ocr_mode)
        config = _config_for_mode(ocr_mode)
        converter = build_converter(config)
        seconds = initialize_converter(converter)
        _converters[ocr_mode] = converter
        _converter_init_seconds[ocr_mode] = seconds
        _log.info("ocr_mode=%s ready in %.2fs", ocr_mode, seconds)
        return converter


def _response_headers(result: ConvertResult, *, ocr_mode: OcrMode) -> dict[str, str]:
    return {
        "X-Docling-Status": result.status,
        "X-Docling-Pages": str(result.pages),
        "X-Docling-Seconds": str(result.convert_seconds),
        "X-Docling-OCR-Mode": ocr_mode,
    }


def _convert_locked(
    pdf_path: Path,
    *,
    ocr_mode: OcrMode,
    include_page_furniture: bool,
) -> ConvertResult:
    converter = _ensure_converter(ocr_mode)
    with _convert_lock:
        result = convert_pdf(
            converter,
            pdf_path,
            include_furniture=include_page_furniture,
        )
        result.ocr_mode = ocr_mode
        return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready

    logging.basicConfig(level=_settings.log_level.upper())
    logging.getLogger("docling").setLevel(logging.WARNING)
    _log.info("Loading default Docling pipeline (device=%s)…", _base_config.device)

    _ensure_converter(_base_config.ocr_mode)
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
    version="1.2.0",
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
        "build": SERVICE_BUILD,
        "title": app.title,
        "version": app.version,
        "ocr_modes": ["off", "auto", "full"],
        "docs": "/docs",
        "endpoints": {
            "convert": "POST /v1/convert",
            "ready": "GET /ready",
            "health": "GET /health",
        },
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    _log.error("Unhandled %s: %s\n%s", type(exc).__name__, exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready", tags=["health"])
def ready() -> dict:
    if not _ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return {
        "status": "ready",
        "build": SERVICE_BUILD,
        "loaded_pipelines": list(_converters.keys()),
        "pipeline_init_seconds": _converter_init_seconds,
        "default_config": {
            "device": _base_config.device,
            "ocr_mode_default": _base_config.ocr_mode,
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
        str,
        Query(
            description="OCR: off | auto | full (aliases: false→off, true→auto)",
        ),
    ] = "off",
    format: Annotated[
        ResponseFormat,
        Query(description="`text`: raw string body; `json`: text + metadata"),
    ] = ResponseFormat.text,
    include_page_furniture: Annotated[
        bool,
        Query(
            description=(
                "Include page headers/footers (Docling 'furniture') in returned text. "
                "Recommended: true for insurance docs with required date/header fields."
            )
        ),
    ] = True,
):
    """
  Upload a PDF and receive extracted content as **inline text** (not a file download).

  - **ocr=off** (default): embedded PDF text only — misses text inside images.
  - **ocr=auto**: OCR on image regions (try for carrier logos in mixed PDFs).
  - **ocr=full**: full-page OCR — use for fully scanned PDFs; slower.
  - **include_page_furniture=true** (default): include page header/footer lines in text output.
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
            ocr_mode = parse_ocr_mode(ocr)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            result = _convert_locked(
                tmp_path,
                ocr_mode=ocr_mode,
                include_page_furniture=include_page_furniture,
            )
        except ConversionError as exc:
            raise HTTPException(
                status_code=422,
                detail={"message": str(exc), "status": exc.status, "pages": exc.pages},
            ) from exc

        output_text = result.markdown

        meta = ConvertMetadata(
            status=result.status,
            pages=result.pages,
            convert_seconds=result.convert_seconds,
            pages_per_second=result.pages_per_second,
            warnings=result.warnings,
            ocr_mode=ocr_mode,
            include_page_furniture=include_page_furniture,
            source_filename=file.filename,
        )

        if format == ResponseFormat.json:
            return ConvertJsonResponse(text=output_text, metadata=meta)

        return PlainTextResponse(
            content=output_text,
            media_type="text/plain; charset=utf-8",
            headers=_response_headers(result, ocr_mode=ocr_mode),
        )

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
