#!/usr/bin/env python3
"""
Analyze Docling benchmark results across all runs in results/.

Reads every results/*/summary.json and produces:
  - Cross-run recall comparison table
  - Per-document recall trend
  - Persistent miss analysis (fields missed in every run)
  - Miss categorization with root-cause hints
  - Actionable recommendations for accuracy improvement

Usage:
  python analyze_results.py
  python analyze_results.py --results-dir results --min-runs 1
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Miss categorization
# ---------------------------------------------------------------------------

_IMAGE_LIKELY_FIELDS = {"Carrier Name", "Network Name"}


def _categorize_miss(field: str, expected: str) -> str:
    """Heuristic root-cause category for a missing field value."""
    if field in _IMAGE_LIKELY_FIELDS:
        return "likely_image_text"
    stripped = expected.strip()
    if stripped.isdigit() and len(stripped) == 4:
        return "year_in_coverage_period_image"
    if " / " in expected:
        return "slash_compound_value"
    if re.search(r"\d[\-\.]\d{3}[\-\.]\d{4}", expected):
        return "phone_format_variant"
    return "unknown"


_CATEGORY_NOTES = {
    "likely_image_text": (
        "Carrier/network name is often a logo image. "
        "Fix: --ocr auto (try first) or --ocr full."
    ),
    "year_in_coverage_period_image": (
        "Coverage-period header (e.g. '01/01/2025') is frequently an image at the top "
        "of SBC pages. Fix: --ocr auto to OCR the header image; verify furniture is included."
    ),
    "slash_compound_value": (
        "Ground truth uses ' / '-separated tiers (e.g. '$20 / $90 / $130') but the PDF "
        "stores each tier in its own table row. The compound matcher in benchmark_docling.py "
        "now checks each component individually — re-run to see updated recall."
    ),
    "phone_format_variant": (
        "Phone number format in the PDF (dashes/dots/parens) may differ from ground truth. "
        "Check the raw output.md and adjust the ground-truth JSON format if needed."
    ),
    "unknown": "Inspect output.md for the document to determine why the value is missing.",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_runs(results_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted(results_dir.glob("*/summary.json")):
        try:
            with summary_path.open(encoding="utf-8") as f:
                data = json.load(f)
            data["_run_id"] = summary_path.parent.name
            runs.append(data)
        except Exception as exc:
            print(f"  [warn] Could not load {summary_path}: {exc}")
    return runs


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _rec_str(recall: float | None) -> str:
    return f"{recall:.1%}" if recall is not None else "n/a"


def _trunc(s: str, n: int = 45) -> str:
    return (s[:n] + "…") if len(s) > n + 1 else s


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _print_run_table(runs: list[dict]) -> None:
    print("=" * 80)
    print("RUN COMPARISON")
    print("=" * 80)
    header = f"{'Run ID':<22} {'OCR':<6} {'FBT':<4} {'Furniture':<10} {'Pg/s':>6} {'Recall':>8} {'Docs':>5}"
    print(header)
    print("-" * 80)
    for r in runs:
        cfg = r.get("config", {})
        ocr = cfg.get("ocr_mode", cfg.get("do_ocr", "?"))
        if isinstance(ocr, bool):
            ocr = "auto" if ocr else "off"
        fbt = "T" if cfg.get("force_backend_text") else "F"
        furn = "T" if cfg.get("include_page_furniture", True) else "F"
        pps = r.get("overall_pages_per_second")
        pps_str = f"{pps:.2f}" if pps else "-"
        acc = r.get("aggregate_accuracy", {})
        recall = acc.get("recall")
        n_docs = len(r.get("documents", []))
        print(
            f"{r['_run_id']:<22} {str(ocr):<6} {fbt:<4} {furn:<10} "
            f"{pps_str:>6} {_rec_str(recall):>8} {n_docs:>5}"
        )
    print()


def _print_per_doc_table(runs: list[dict]) -> None:
    print("=" * 80)
    print("PER-DOCUMENT RECALL ACROSS RUNS")
    print("=" * 80)

    run_ids = [r["_run_id"] for r in runs]
    doc_recalls: dict[str, list[float | None]] = defaultdict(list)
    doc_pages: dict[str, int] = {}

    for r in runs:
        seen: set[str] = set()
        for doc in r.get("documents", []):
            dk = f"{doc['category']}/{doc['name']}"
            acc = doc.get("accuracy") or {}
            doc_recalls[dk].append(acc.get("recall"))
            doc_pages[dk] = doc.get("pages", 0)
            seen.add(dk)
        # pad missing docs with None
        for dk in doc_recalls:
            if dk not in seen and len(doc_recalls[dk]) < len(run_ids):
                doc_recalls[dk].append(None)

    col_w = max(20, min(26, len(run_ids[0]) + 2)) if run_ids else 20
    header = f"{'Document':<16} {'Pg':>3}"
    for rid in run_ids:
        header += f"  {rid[:col_w]:>{col_w}}"
    print(header)
    print("-" * (16 + 3 + len(run_ids) * (col_w + 2) + 2))

    for dk in sorted(doc_recalls):
        row = f"{dk:<16} {doc_pages.get(dk, 0):>3}"
        for rec in doc_recalls[dk]:
            row += f"  {_rec_str(rec):>{col_w}}"
        print(row)
    print()


def _build_miss_map(runs: list[dict]) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Returns {(doc_key, field): [(run_id, expected), ...]}"""
    miss_map: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for r in runs:
        run_id = r["_run_id"]
        for doc in r.get("documents", []):
            dk = f"{doc['category']}/{doc['name']}"
            for miss in (doc.get("accuracy") or {}).get("misses", []):
                miss_map[(dk, miss["field"])].append((run_id, miss["expected"]))
    return dict(miss_map)


def _print_miss_analysis(miss_map: dict, runs: list[dict], min_runs: int) -> None:
    n_runs = len(runs)
    print("=" * 80)
    print(f"PERSISTENT MISSES (missed in >= {min_runs}/{n_runs} run(s))")
    print("=" * 80)

    filtered = {k: v for k, v in miss_map.items() if len(v) >= min_runs}
    if not filtered:
        print("  None — all fields extracted in every qualifying run.\n")
        return

    # Group by category for readability
    by_category: dict[str, list[tuple]] = defaultdict(list)
    for (dk, field), occ in sorted(filtered.items(), key=lambda x: (-len(x[1]), x[0])):
        expected = occ[-1][1]
        cat = occ[-1][1] and _categorize_miss(field, expected)
        # Check if the saved miss already has a category from the JSON
        by_category[cat].append((dk, field, expected, len(occ), n_runs))

    for cat, items in sorted(by_category.items()):
        print(f"\n  [{cat}]")
        print(f"  {_CATEGORY_NOTES.get(cat, '')}")
        print()
        print(f"    {'Document':<14} {'Runs':>7}  {'Field':<38} Expected")
        print(f"    {'-'*14} {'-'*7}  {'-'*38} {'-'*20}")
        for dk, field, expected, n_missed, total in items:
            print(
                f"    {dk:<14} {n_missed:>3}/{total:<3}  "
                f"{field:<38} {_trunc(expected)}"
            )
    print()


def _print_recommendations(miss_map: dict, runs: list[dict]) -> None:
    n_runs = len(runs)
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    rec: list[str] = []

    # Check for force_backend_text=False runs (should match service default=True)
    bad_fbt = [r["_run_id"] for r in runs if not r.get("config", {}).get("force_backend_text")]
    if bad_fbt:
        rec.append(
            f"force_backend_text was OFF in runs: {', '.join(bad_fbt)}\n"
            "  The service defaults DOCLING_FORCE_BACKEND_TEXT=true. The benchmark now\n"
            "  defaults force_backend_text=True — re-run to confirm consistency."
        )

    # Check for missing include_page_furniture key (old runs)
    no_furn = [
        r["_run_id"]
        for r in runs
        if "include_page_furniture" not in r.get("config", {})
    ]
    if no_furn:
        rec.append(
            f"Runs missing 'include_page_furniture' in config: {', '.join(no_furn)}\n"
            "  These are older runs. The key is now recorded — re-run for full traceability."
        )

    # Categorize persistent misses (missed in all runs)
    all_misses = {k: v for k, v in miss_map.items() if len(v) == n_runs}
    image_docs: set[str] = set()
    slash_count = 0
    year_docs: set[str] = set()

    for (dk, field), occ in all_misses.items():
        expected = occ[-1][1]
        cat = _categorize_miss(field, expected)
        if cat == "likely_image_text":
            image_docs.add(dk)
        elif cat == "slash_compound_value":
            slash_count += 1
        elif cat == "year_in_coverage_period_image":
            year_docs.add(dk)

    if image_docs:
        rec.append(
            f"Carrier/network name in image for: {', '.join(sorted(image_docs))}\n"
            "  Run with --ocr auto (adds OCR only on image regions, fast).\n"
            "  If still missing, try --ocr full (re-OCRs full pages, slower).\n"
            "  Note: full OCR can break table structure on text PDFs (see Health/4 analysis)."
        )

    if year_docs:
        rec.append(
            f"Plan Year missing for: {', '.join(sorted(year_docs))}\n"
            "  Coverage period (e.g. '01/01/2025 – 12/31/2025') is typically an image\n"
            "  at the top of SBC pages. Run with --ocr auto to capture it.\n"
            "  Alternatively check if the year appears in the footer furniture."
        )

    if slash_count > 0:
        rec.append(
            f"{slash_count} slash-compound field(s) still unresolved.\n"
            "  The compound matcher in benchmark_docling.py now checks each ' / '-separated\n"
            "  component individually. Re-run the benchmark to see updated recall.\n"
            "  If still failing, inspect output.md — the individual values may not be\n"
            "  present (e.g. mail-order tier in a footnote not captured by Docling)."
        )

    if not rec:
        rec.append("No systemic issues detected. Review individual output.md files for remaining misses.")

    for i, r in enumerate(rec, 1):
        print(f"  {i}. {r}\n")


def _print_output_artifact_warning(runs: list[dict]) -> None:
    """Warn about the letter-by-letter artifact seen in some SBC outputs."""
    print("=" * 80)
    print("KNOWN DOCLING ARTIFACT")
    print("=" * 80)
    print(
        "  Some SBC PDFs (e.g. Health/1) have a vertical form code printed sideways\n"
        "  (e.g. '605718_ACMaromS2TglinesLdcp'). Docling extracts this as individual\n"
        "  single-character lines at the end of the markdown. This is harmless for\n"
        "  downstream LLM extraction but adds ~50 junk lines to the output.\n\n"
        "  Fix options:\n"
        "    1. Post-process: strip lines that are single characters or <3 chars.\n"
        "    2. Check output_docling.json to identify the element and filter by\n"
        "       bounding-box position (right margin of last page).\n"
        "    3. No action needed if the LLM ignores single-character tokens.\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze Docling benchmark results across all runs."
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing timestamped run subdirectories (default: results)",
    )
    p.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="Show misses that occur in at least this many runs (default: 1 = all misses)",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        print(f"Results directory not found: {results_dir}")
        return

    runs = _load_runs(results_dir)
    if not runs:
        print(f"No summary.json files found under {results_dir}")
        return

    print(f"\nDocling Benchmark Analysis — {len(runs)} run(s) from {results_dir}\n")

    _print_run_table(runs)
    _print_per_doc_table(runs)

    miss_map = _build_miss_map(runs)
    _print_miss_analysis(miss_map, runs, args.min_runs)
    _print_recommendations(miss_map, runs)
    _print_output_artifact_warning(runs)


if __name__ == "__main__":
    main()
