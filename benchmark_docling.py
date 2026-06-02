#!/usr/bin/env python3
"""
Benchmark Docling on insurance PDF fixtures.

Measures:
  - Pipeline initialization time (one-time model load)
  - Per-document conversion time and pages/sec
  - Field-level text recall vs ground-truth JSON in test_fixtures/

Default OCR mode is **full** (full-page OCR). Use --ocr off|auto for faster runs.

Example:
  python benchmark_docling.py --device cuda --warmup
  python benchmark_docling.py --device cuda --ocr off --force-backend-text --warmup
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from docling.datamodel.base_models import InputFormat

from docling_pipeline import (
    ConversionError,
    OcrMode,
    PipelineConfig,
    build_converter,
    convert_pdf,
    cuda_info,
    initialize_converter,
    parse_ocr_mode,
)

_log = logging.getLogger("benchmark_docling")


# ---------------------------------------------------------------------------
# Ground-truth accuracy (text presence in Docling output)
# ---------------------------------------------------------------------------


def _normalize_for_match(text: str) -> str:
    """Loose normalization so '$1,650' and '1650' can still match."""
    t = text.lower().strip()
    t = re.sub(r"[\s\-_/\\]+", " ", t)
    t = re.sub(r"[^\w\s.$%]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _digits_only(text: str) -> str:
    return re.sub(r"\D", "", text)


def _value_variants(value: str) -> list[str]:
    """Generate normalized strings to search for in extracted text."""
    variants: list[str] = []
    base = _normalize_for_match(value)
    if base:
        variants.append(base)
    digits = _digits_only(value)
    if len(digits) >= 2:
        variants.append(digits)
    # Strip currency / punctuation variants: "$1,650" -> "1650"
    stripped = re.sub(r"[$,\s]", "", value.lower())
    if stripped and stripped not in variants:
        variants.append(stripped)
    return list(dict.fromkeys(v for v in variants if v))


def field_recall(
    ground_truth: dict[str, Any],
    extracted_text: str,
) -> dict[str, Any]:
    """
    For each non-empty ground-truth field, check whether its value appears
    in the Docling markdown/plain text. This proxies extraction accuracy
    before any structured LLM parsing step.
    """
    norm_doc = _normalize_for_match(extracted_text)
    norm_doc_digits = _digits_only(extracted_text)

    hits: list[str] = []
    misses: list[dict[str, str]] = []
    skipped_empty = 0

    for key, raw in ground_truth.items():
        if raw is None:
            skipped_empty += 1
            continue
        value = str(raw).strip()
        if not value:
            skipped_empty += 1
            continue

        found = False
        for variant in _value_variants(value):
            if variant.isdigit():
                if variant in norm_doc_digits:
                    found = True
                    break
            elif variant in norm_doc or variant in extracted_text.lower():
                found = True
                break

        if found:
            hits.append(key)
        else:
            misses.append({"field": key, "expected": value})

    evaluated = len(hits) + len(misses)
    recall = (len(hits) / evaluated) if evaluated else 0.0

    return {
        "fields_evaluated": evaluated,
        "fields_skipped_empty": skipped_empty,
        "fields_found": len(hits),
        "fields_missing": len(misses),
        "recall": round(recall, 4),
        "hit_fields": hits,
        "misses": misses,
    }


# ---------------------------------------------------------------------------
# Fixture discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PdfFixture:
    category: str
    stem: str
    pdf_path: Path
    ground_truth_path: Path | None


def discover_fixtures(fixtures_root: Path) -> list[PdfFixture]:
    fixtures: list[PdfFixture] = []
    for pdf_path in sorted(fixtures_root.rglob("*.pdf")):
        rel = pdf_path.relative_to(fixtures_root)
        category = rel.parts[0] if len(rel.parts) > 1 else "root"
        stem = pdf_path.stem
        gt = pdf_path.with_suffix(".json")
        fixtures.append(
            PdfFixture(
                category=category,
                stem=stem,
                pdf_path=pdf_path.resolve(),
                ground_truth_path=gt.resolve() if gt.is_file() else None,
            )
        )
    return fixtures


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------


@dataclass
class DocumentResult:
    category: str
    name: str
    pdf_path: str
    status: str
    pages: int
    convert_seconds: float
    pages_per_second: float | None
    accuracy: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class BenchmarkSummary:
    started_at: str
    finished_at: str
    device: str
    fixtures_root: str
    output_dir: str
    pipeline_init_seconds: float
    total_convert_seconds: float
    total_pages: int
    overall_pages_per_second: float | None
    config: dict[str, Any] = field(default_factory=dict)
    documents: list[DocumentResult] = field(default_factory=list)
    aggregate_accuracy: dict[str, Any] = field(default_factory=dict)


def _load_ground_truth(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _print_cuda_info() -> None:
    info = cuda_info()
    if info.get("available"):
        _log.info(
            "CUDA: %s (device %s, %.1f GB VRAM, torch %s, cuda %s)",
            info["name"],
            info["device_index"],
            info["vram_gb"],
            info["torch_version"],
            info["cuda_version"],
        )
    else:
        _log.warning("CUDA not available — Docling will use configured device")


def run_benchmark(args: argparse.Namespace) -> int:
    fixtures_root = Path(args.fixtures_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id

    fixtures = discover_fixtures(fixtures_root)
    if not fixtures:
        _log.error("No PDF files under %s", fixtures_root)
        return 1

    _log.info("Found %d PDF fixture(s) under %s", len(fixtures), fixtures_root)
    _print_cuda_info()

    ocr_mode: OcrMode = parse_ocr_mode(args.ocr)
    pipeline_config = PipelineConfig(
        device=args.device,
        layout_batch_size=args.layout_batch_size,
        ocr_batch_size=args.ocr_batch_size,
        table_batch_size=args.table_batch_size,
        do_table_structure=not args.no_tables,
        table_mode=args.table_mode,
        force_backend_text=args.force_backend_text,
        ocr_mode=ocr_mode,
    )
    _log.info("OCR mode: %s", ocr_mode)
    converter = build_converter(pipeline_config)

    init_seconds = initialize_converter(converter)
    _log.info("Pipeline initialized in %.2f s", init_seconds)

    if args.warmup and fixtures:
        warmup_pdf = fixtures[0].pdf_path
        _log.info("Warmup conversion (not timed): %s", warmup_pdf.name)
        converter.convert(str(warmup_pdf))

    doc_results: list[DocumentResult] = []
    total_convert = 0.0
    total_pages = 0
    all_hits = 0
    all_evaluated = 0

    for fx in fixtures:
        doc_out = run_dir / fx.category / fx.stem
        doc_out.mkdir(parents=True, exist_ok=True)
        _log.info("Converting %s / %s …", fx.category, fx.stem)

        t1 = time.perf_counter()
        try:
            try:
                result = convert_pdf(converter, fx.pdf_path)
            except ConversionError as conv_exc:
                elapsed = time.perf_counter() - t1
                doc_results.append(
                    DocumentResult(
                        category=fx.category,
                        name=fx.stem,
                        pdf_path=str(fx.pdf_path),
                        status=conv_exc.status,
                        pages=conv_exc.pages,
                        convert_seconds=round(elapsed, 3),
                        pages_per_second=None,
                        error=str(conv_exc),
                    )
                )
                continue

            elapsed = result.convert_seconds
            status = result.status
            pages = result.pages
            warnings = list(result.warnings)
            markdown = result.markdown
            doc_dict = result.doc_dict or {}

            _write_text(doc_out / "output.md", markdown)
            with (doc_out / "output_docling.json").open("w", encoding="utf-8") as f:
                json.dump(doc_dict, f, indent=2, ensure_ascii=False)

            gt = _load_ground_truth(fx.ground_truth_path)
            accuracy = None
            if gt is not None:
                accuracy = field_recall(gt, markdown)
                with (doc_out / "accuracy.json").open("w", encoding="utf-8") as f:
                    json.dump(accuracy, f, indent=2, ensure_ascii=False)
                all_hits += accuracy["fields_found"]
                all_evaluated += accuracy["fields_evaluated"]

            pps = result.pages_per_second
            total_convert += elapsed
            total_pages += pages

            timing = {
                "convert_seconds": round(elapsed, 3),
                "pages": pages,
                "pages_per_second": pps,
            }
            with (doc_out / "timing.json").open("w", encoding="utf-8") as f:
                json.dump(timing, f, indent=2)

            doc_results.append(
                DocumentResult(
                    category=fx.category,
                    name=fx.stem,
                    pdf_path=str(fx.pdf_path),
                    status=status,
                    pages=pages,
                    convert_seconds=round(elapsed, 3),
                    pages_per_second=pps,
                    accuracy=accuracy,
                    warnings=warnings,
                )
            )
            recall_str = (
                f"{accuracy['recall']:.1%}" if accuracy else "n/a"
            )
            warn_str = f" [{', '.join(warnings)}]" if warnings else ""
            _log.info(
                "  done: %.2fs, %d pages (%.2f pg/s), field recall %s%s",
                elapsed,
                pages,
                pps or 0.0,
                recall_str,
                warn_str,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t1
            _log.exception("Failed on %s: %s", fx.pdf_path, exc)
            doc_results.append(
                DocumentResult(
                    category=fx.category,
                    name=fx.stem,
                    pdf_path=str(fx.pdf_path),
                    status="FAILED",
                    pages=0,
                    convert_seconds=round(elapsed, 3),
                    pages_per_second=None,
                    error=str(exc),
                )
            )

    finished = datetime.now(timezone.utc).isoformat()
    overall_pps = (
        round(total_pages / total_convert, 3)
        if total_convert > 0 and total_pages
        else None
    )
    aggregate_accuracy = {
        "fields_found": all_hits,
        "fields_evaluated": all_evaluated,
        "recall": round(all_hits / all_evaluated, 4) if all_evaluated else None,
    }

    summary = BenchmarkSummary(
        started_at=run_id,
        finished_at=finished,
        device=args.device,
        fixtures_root=str(fixtures_root),
        output_dir=str(run_dir),
        pipeline_init_seconds=round(init_seconds, 3),
        total_convert_seconds=round(total_convert, 3),
        total_pages=total_pages,
        overall_pages_per_second=overall_pps,
        config={
            "ocr_mode": ocr_mode,
            "layout_batch_size": args.layout_batch_size,
            "ocr_batch_size": args.ocr_batch_size,
            "table_batch_size": args.table_batch_size,
            "do_table_structure": not args.no_tables,
            "table_mode": args.table_mode,
            "force_backend_text": args.force_backend_text,
            "warmup": args.warmup,
        },
        documents=doc_results,
        aggregate_accuracy=aggregate_accuracy,
    )

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)

    _print_summary(summary)
    _log.info("Full report: %s", summary_path)
    return 0


def _print_summary(summary: BenchmarkSummary) -> None:
    print("\n" + "=" * 72)
    ocr_mode = summary.config.get("ocr_mode", "off")
    print(f"DOCLING BENCHMARK SUMMARY (ocr={ocr_mode})")
    print("=" * 72)
    print(f"Output folder     : {summary.output_dir}")
    print(f"Device            : {summary.device}")
    print(f"Pipeline init     : {summary.pipeline_init_seconds:.2f} s")
    print(f"Total conversion  : {summary.total_convert_seconds:.2f} s")
    print(f"Total pages       : {summary.total_pages}")
    if summary.overall_pages_per_second is not None:
        print(f"Throughput        : {summary.overall_pages_per_second:.2f} pages/s")
    acc = summary.aggregate_accuracy
    if acc.get("fields_evaluated"):
        print(
            f"Field text recall : {acc['fields_found']}/{acc['fields_evaluated']} "
            f"({acc['recall']:.1%})"
        )
    print("-" * 72)
    print(f"{'Category':<10} {'Doc':<6} {'Pages':>5} {'Time(s)':>8} {'Pg/s':>8} {'Recall':>8}")
    print("-" * 72)
    for d in summary.documents:
        recall = ""
        if d.accuracy is not None:
            recall = f"{d.accuracy['recall']:.1%}"
            if d.warnings:
                recall += "*"
        elif d.error:
            recall = "ERR"
        pps = f"{d.pages_per_second:.2f}" if d.pages_per_second is not None else "-"
        print(
            f"{d.category:<10} {d.name:<6} {d.pages:>5} {d.convert_seconds:>8.2f} "
            f"{pps:>8} {recall:>8}"
        )
    print("=" * 72 + "\n")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark Docling speed and field-level text recall on insurance PDF fixtures.",
    )
    p.add_argument(
        "--fixtures-dir",
        default="test_fixtures",
        help="Root folder with Category/*.pdf and matching .json ground truth",
    )
    p.add_argument(
        "--output-dir",
        default="results",
        help="Parent directory for timestamped run outputs",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "auto", "mps"],
        help="Accelerator for layout/table models (default: cuda)",
    )
    p.add_argument(
        "--layout-batch-size",
        type=int,
        default=64,
        help="Layout model GPU batch size (increase if VRAM allows)",
    )
    p.add_argument(
        "--ocr",
        default="full",
        choices=["off", "auto", "full"],
        help="OCR mode: off (embedded text), auto (image regions), full (full-page OCR, default)",
    )
    p.add_argument(
        "--ocr-batch-size",
        type=int,
        default=4,
        help="OCR batch size when OCR is enabled",
    )
    p.add_argument(
        "--table-batch-size",
        type=int,
        default=4,
        help="Table structure batch size",
    )
    p.add_argument(
        "--table-mode",
        default="accurate",
        choices=["accurate", "fast"],
        help="TableFormer mode — accurate for insurance tables",
    )
    p.add_argument(
        "--no-tables",
        action="store_true",
        help="Disable table structure extraction (faster, less accurate on tables)",
    )
    p.add_argument(
        "--force-backend-text",
        action="store_true",
        help="Use embedded PDF text instead of layout-model text (text PDFs)",
    )
    p.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="CPU threads for non-GPU stages",
    )
    p.add_argument(
        "--warmup",
        action="store_true",
        help="Run one untimed conversion before measuring",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("docling").setLevel(logging.WARNING if not args.verbose else logging.INFO)

    # Optional env override used by Docling
    if args.device == "cuda":
        import os

        os.environ.setdefault("DOCLING_DEVICE", "cuda")

    sys.exit(run_benchmark(args))


if __name__ == "__main__":
    main()
