# Docling insurance PDF benchmark

Benchmark [Docling](https://github.com/docling-project/docling) on the insurance PDF fixtures in `test_fixtures/`, measuring **processing time** and **field-level text recall** against the paired JSON ground truth.

**Assumption for this phase:** all PDFs are **text-selectable** (embedded text layer). OCR is disabled (`do_ocr=False`).

## What gets measured

| Metric | Description |
|--------|-------------|
| Pipeline init | One-time model load before any PDF |
| Per-PDF time | Seconds and pages/second per document |
| Field recall | For each non-empty value in the fixture JSON, whether that text appears in Docling’s markdown output |

Field recall is a **proxy** for extraction quality: it checks that plan values (deductibles, copays, etc.) are present in Docling’s text. A later LLM or rules step would map that text into structured JSON.

## GPU server setup (CUDA 12.x / 12.9)

Use Python 3.10–3.12 and a virtual environment.

```bash
cd docling_marker_project
python -m venv .venv
source .venv/bin/activate   # Linux
# .venv\Scripts\activate    # Windows

# PyTorch with CUDA 12.x wheels (driver 12.9 works with cu126/cu128 builds)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

Verify GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

First run downloads layout/table models (~GB); allow network access once.

## Run benchmark

```bash
# Recommended: GPU, accurate tables, warmup, native PDF text
python benchmark_docling.py \
  --device cuda \
  --layout-batch-size 64 \
  --table-mode accurate \
  --force-backend-text \
  --warmup \
  -v
```

Lower VRAM (e.g. 8–12 GB): use `--layout-batch-size 16`.

Speed-only (tables off, not recommended for accuracy):

```bash
python benchmark_docling.py --device cuda --no-tables --warmup
```

CPU baseline:

```bash
python benchmark_docling.py --device cpu --warmup
```

## Outputs

Each run creates `results/<UTC-timestamp>/`:

```
results/20260528T120000Z/
  summary.json              # timings + aggregate recall
  Health/1/
    output.md               # Docling markdown
    output_docling.json     # structured document
    timing.json
    accuracy.json           # per-field hits/misses
  Dental/1/
  ...
```

Console prints a table of time and recall per document.

## Fixtures

| Folder | PDFs | Ground truth |
|--------|------|--------------|
| `test_fixtures/Health/` | 5 | `*.json` |
| `test_fixtures/Dental/` | 2 | `*.json` |
| `test_fixtures/Vision/` | 2 | `*.json` |

## Interpreting results

- **Pages/second** — compare GPU vs CPU and batch sizes; Docling docs cite ~4–8 pg/s (standard pipeline, no OCR) on recent GPUs for similar workloads.
- **Field recall** — aim high before adding structured extraction. Misses often mean values live only in table cells, images, or odd formatting; try `--table-mode accurate` and avoid `--no-tables`.
- **Pipeline init** — exclude from per-document SLA if you keep the converter process warm in production.
- **PARTIAL_SUCCESS** — some pages failed (often `std::bad_alloc` on CPU/low RAM). Outputs and recall are still written; recall marked with `*` in the table. Use GPU + enough RAM for full Health PDFs.

## Microservice (GPU server)

HTTP API that converts an uploaded PDF and returns **plain text** in the response body (markdown string) for the next pipeline stage.

### Start the service

```bash
export DOCLING_DEVICE=cuda
export DOCLING_LAYOUT_BATCH_SIZE=64
export DOCLING_FORCE_BACKEND_TEXT=true
export DOCLING_TABLE_MODE=accurate

uvicorn service.app:app --host 0.0.0.0 --port 8000
```

Default pipeline (OCR off) loads at startup (~6–10 s). OCR-on pipeline loads on first `ocr=true` request.

```bash
curl http://localhost:8000/ready
```

### Convert a PDF → text (for next stage)

```bash
# Text-selectable PDF (default) — response body is the extracted text
curl -X POST "http://localhost:8000/v1/convert" \
  -F "file=@test_fixtures/Health/1.pdf"

# Scanned PDF — enable OCR
curl -X POST "http://localhost:8000/v1/convert?ocr=true" \
  -F "file=@scanned.pdf"

# JSON: { "text": "...", "metadata": { pages, timing, ocr, ... } }
curl -X POST "http://localhost:8000/v1/convert?format=json" \
  -F "file=@test_fixtures/Health/1.pdf"
```

| Query param | Default | Description |
|-------------|---------|-------------|
| `ocr` | `false` | `true` = OCR for scanned/image PDFs |
| `format` | `text` | `text` = raw body; `json` = `{ "text", "metadata" }` |

Response headers (`format=text`): `X-Docling-Status`, `X-Docling-Pages`, `X-Docling-Seconds`, `X-Docling-OCR`.

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness |
| `GET /ready` | Models loaded |
| `POST /v1/convert` | PDF upload → text |

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_DEVICE` | `cuda` | `cuda`, `cpu`, `auto` |
| `DOCLING_LAYOUT_BATCH_SIZE` | `64` | GPU layout batch |
| `DOCLING_FORCE_BACKEND_TEXT` | `true` | Embedded PDF text when OCR off |
| `DOCLING_DO_OCR` | `false` | Default pipeline at startup if `true` |
| `DOCLING_TABLE_MODE` | `accurate` | `accurate` or `fast` |
| `SERVICE_MAX_UPLOAD_MB` | `50` | Max upload size |

One request runs at a time per process (GPU lock). Scale with multiple workers/replicas if needed.

## Next steps

- Scanned PDFs: set `DOCLING_DO_OCR=true` and configure RapidOCR torch backend.
- Structured JSON: send markdown from the service to an LLM with your schema.
- Compare with Marker or other pipelines in the same repo layout.
