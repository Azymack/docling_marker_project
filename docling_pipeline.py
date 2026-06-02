"""Shared Docling PDF → markdown pipeline."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    RapidOcrOptions,
    TableFormerMode,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

OcrMode = Literal["off", "auto", "full"]
OK_STATUSES = {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}


def parse_ocr_mode(value: str | bool) -> OcrMode:
    """Normalize API / env values to off | auto | full."""
    if isinstance(value, bool):
        return "auto" if value else "off"
    v = str(value).lower().strip()
    if v in ("false", "0", "off", "no", "none"):
        return "off"
    if v in ("full", "force", "all", "force_full"):
        return "full"
    if v in ("true", "1", "auto", "yes", "on"):
        return "auto"
    raise ValueError(f"Invalid ocr mode {value!r}; use off, auto, or full")


@dataclass
class PipelineConfig:
    device: str = "cuda"
    layout_batch_size: int = 64
    ocr_batch_size: int = 4
    table_batch_size: int = 4
    do_table_structure: bool = True
    table_mode: str = "accurate"
    force_backend_text: bool = True
    ocr_mode: OcrMode = "off"
    # Only OCR image regions larger than this fraction of the page (auto mode)
    bitmap_area_threshold: float = 0.01

    @classmethod
    def from_env(cls) -> PipelineConfig:
        def _bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        ocr_mode: OcrMode = "off"
        if os.environ.get("DOCLING_OCR_MODE"):
            ocr_mode = parse_ocr_mode(os.environ["DOCLING_OCR_MODE"])
        elif _bool("DOCLING_DO_OCR", False):
            ocr_mode = "auto"

        return cls(
            device=os.environ.get("DOCLING_DEVICE", "cuda"),
            layout_batch_size=int(os.environ.get("DOCLING_LAYOUT_BATCH_SIZE", "64")),
            ocr_batch_size=int(os.environ.get("DOCLING_OCR_BATCH_SIZE", "4")),
            table_batch_size=int(os.environ.get("DOCLING_TABLE_BATCH_SIZE", "4")),
            do_table_structure=_bool("DOCLING_DO_TABLE_STRUCTURE", True),
            table_mode=os.environ.get("DOCLING_TABLE_MODE", "accurate"),
            force_backend_text=_bool("DOCLING_FORCE_BACKEND_TEXT", True),
            ocr_mode=ocr_mode,
            bitmap_area_threshold=float(
                os.environ.get("DOCLING_BITMAP_AREA_THRESHOLD", "0.01")
            ),
        )


@dataclass
class ConvertResult:
    markdown: str
    status: str
    pages: int
    convert_seconds: float
    pages_per_second: float | None
    warnings: list[str] = field(default_factory=list)
    doc_dict: dict[str, Any] | None = None
    ocr_mode: OcrMode = "off"


class ConversionError(Exception):
    def __init__(self, message: str, *, status: str, pages: int = 0) -> None:
        super().__init__(message)
        self.status = status
        self.pages = pages


def _map_device(name: str) -> AcceleratorDevice:
    mapping = {
        "cuda": AcceleratorDevice.CUDA,
        "cpu": AcceleratorDevice.CPU,
        "auto": AcceleratorDevice.AUTO,
        "mps": AcceleratorDevice.MPS,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unknown device {name!r}; use one of {list(mapping)}")
    return mapping[key]


def build_converter(config: PipelineConfig) -> DocumentConverter:
    if config.device == "cuda":
        os.environ.setdefault("DOCLING_DEVICE", "cuda")

    table_enum = (
        TableFormerMode.FAST
        if config.table_mode.lower() == "fast"
        else TableFormerMode.ACCURATE
    )

    do_ocr = config.ocr_mode != "off"
    force_backend_text = config.force_backend_text if config.ocr_mode == "off" else False

    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=_map_device(config.device)),
        ocr_batch_size=config.ocr_batch_size,
        layout_batch_size=config.layout_batch_size,
        table_batch_size=config.table_batch_size,
        do_ocr=do_ocr,
        do_table_structure=config.do_table_structure,
        force_backend_text=force_backend_text,
    )
    pipeline_options.table_structure_options.mode = table_enum

    if do_ocr:
        ocr_options = RapidOcrOptions(
            backend="torch",
            force_full_page_ocr=(config.ocr_mode == "full"),
            bitmap_area_threshold=config.bitmap_area_threshold,
        )
        pipeline_options.ocr_options = ocr_options

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )


def initialize_converter(converter: DocumentConverter) -> float:
    """Load models once. Returns init time in seconds."""
    t0 = time.perf_counter()
    converter.initialize_pipeline(InputFormat.PDF)
    return time.perf_counter() - t0


def export_document_markdown(document: Any, *, include_furniture: bool = True) -> str:
    """
    Export Docling document to markdown.

    By default Docling exports BODY only, which drops page headers/footers
  (SBC title, coverage period, plan name). Insurance PDFs need FURNITURE too.
    """
    from docling_core.types.doc.document import ContentLayer

    if include_furniture:
        layers = {ContentLayer.BODY, ContentLayer.FURNITURE}
    else:
        layers = {ContentLayer.BODY}
    return document.export_to_markdown(included_content_layers=layers)


def convert_pdf(
    converter: DocumentConverter,
    pdf_path: str | Path,
    *,
    include_furniture: bool = True,
) -> ConvertResult:
    """Convert a PDF file to markdown."""
    path = Path(pdf_path)
    t0 = time.perf_counter()
    conv = converter.convert(str(path))
    elapsed = time.perf_counter() - t0

    status = conv.status.name if conv.status else "UNKNOWN"
    pages = len(conv.pages) if conv.pages else 0

    if conv.status not in OK_STATUSES:
        raise ConversionError(
            f"Conversion failed with status {status}",
            status=status,
            pages=pages,
        )

    warnings: list[str] = []
    if conv.status == ConversionStatus.PARTIAL_SUCCESS:
        warnings.append("partial_success_some_pages_failed")

    markdown = export_document_markdown(
        conv.document,
        include_furniture=include_furniture,
    )
    pps = round(pages / elapsed, 3) if elapsed > 0 and pages else None

    return ConvertResult(
        markdown=markdown,
        status=status,
        pages=pages,
        convert_seconds=round(elapsed, 3),
        pages_per_second=pps,
        warnings=warnings,
        doc_dict=conv.document.export_to_dict(),
    )


def cuda_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return {
            "available": True,
            "device_index": idx,
            "name": torch.cuda.get_device_name(idx),
            "vram_gb": round(props.total_memory / (1024**3), 2),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    except ImportError:
        return {"available": False, "error": "torch not installed"}
