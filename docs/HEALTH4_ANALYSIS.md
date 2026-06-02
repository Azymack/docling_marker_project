# Health/4.pdf — why extraction and layout look wrong

This document explains root causes for `test_fixtures/Health/4.pdf` (Blue Cross Blue Shield Arizona SBC), not a single bug in our API wrapper.

## Symptoms

- Missing or wrong header fields (plan name, coverage period, network name)
- Broken top “Important Questions” table (question/answer columns mixed)
- Full OCR (`ocr=full`) makes layout **worse** (garbled text, wrong cells)

## Core issue 1 — Docling drops header/footer layer by default

Docling splits PDF content into layers:

| Layer | Examples on Health/4 |
|-------|----------------------|
| **BODY** | Main SBC tables and paragraphs |
| **FURNITURE** | Page headers: `Summary of Benefits and Coverage`, `EverydayHealth PPO Gold 3500`, coverage period |

**Default `export_to_markdown()` only exports BODY.**  
Headers are parsed (visible in `output_docling.json`) but **omitted from markdown**.

That is why fields like **Network Name** (`EverydayHealth`) and **Plan Name** (`PPO Gold 3500`) are missing from API/benchmark text even though Docling detected them.

**Fix (implemented):** export with both layers:

```python
from docling_core.types.doc.document import ContentLayer

md = doc.export_to_markdown(
    included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
)
```

Service and benchmark now use this by default (`include_page_furniture=true`).

## Core issue 2 — SBC tables are visual grids, not clean PDF tables

Health/4 uses the federal **Summary of Benefits and Coverage** layout:

- Multi-column grid with merged cells and wrapped text
- “Important Questions | Answers | Why This Matters” block
- Many benefit tables spanning pages with repeated headers

Docling pipeline:

1. Layout model detects regions  
2. **TableFormer** rebuilds table structure  
3. Markdown export serializes rows/columns  

When the PDF has **visual alignment without simple table tags**, TableFormer can **assign text to the wrong cell**. Example from `output.md` row 17:

- Question text and dollar amounts from different columns end up in one cell
- “Why This Matters” column gets shuffled to the wrong row  

This is a **table-structure reconstruction limit**, not fixed by turning OCR on.

## Core issue 3 — full-page OCR harms this PDF

Health/4 already has an embedded text layer. With `ocr=full`:

- Embedded text is replaced by OCR on rendered page images
- OCR noise appears (e.g. corrupted words in cells)
- Table detection gets worse

**Use `ocr=off` for Health/4** (and most text SBC PDFs).  
Use `ocr=auto` only for logos/images; use `ocr=full` only for scanned PDFs.

## Core issue 4 — some fields are images or formatted text

| Field | Issue |
|-------|--------|
| **Carrier Name** | Logo/wordmark area is often a **picture**, not extractable text at top of page 1 |
| **Carrier Name** (footer) | Appears later as `Blue Cross ® Blue Shield ® of Arizona` — different string than ground truth `Blue Cross Blue Shield Arizona` |
| **Generic RX** | Text exists as `Tier 1a: $3 copay... Tier 1b: $20 copay...` — ground truth uses `Tier 1a: $3 / Tier 1b: $20` (format mismatch in benchmark matcher) |

## What to expect after the furniture export fix

After pulling latest code and re-running:

- You should see **SBC header lines** and **EverydayHealth PPO Gold 3500** at the top of markdown
- Benefit tables may still have **column alignment issues** in markdown (issue 2)
- Carrier logo name may still need **`ocr=auto`** or manual mapping from footer text

## Recommended settings for Health/4

```bash
curl -X POST "http://host:8001/v1/convert?format=json&ocr=off&include_page_furniture=true" \
  -F "file=@test_fixtures/Health/4.pdf"
```

```bash
python benchmark_docling.py --device cuda --ocr off --warmup
```

## Longer-term options (if markdown tables stay broken)

1. Send **linear text** (all text sorted by position) to the LLM instead of markdown tables  
2. Use Docling JSON (`output_docling.json`) and walk `texts` + `tables` with custom logic  
3. Try a different extractor for SBC grid pages (e.g. pdfplumber area-based extraction) for comparison  
4. For carrier logo, use `ocr=auto` on page 1 or a dedicated logo OCR crop  
