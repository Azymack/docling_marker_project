# Docling PDF service

GPU microservice that converts insurance PDFs to **plain text** (markdown) for a downstream extraction step (LLM, rules, etc.).

This service does **not** return structured fields (`Carrier Name`, deductibles, …). It only runs **stage 1** of the pipeline.

## Pipeline

```
PDF  →  POST /v1/convert (this service)  →  text string
       →  your next stage (LLM / API)     →  JSON (e.g. test_fixtures/*.json)
```

| Stage | Responsibility |
|-------|----------------|
| **This service** | PDF → `text` (tables and headings as markdown) |
| **Your next stage** | `text` → structured JSON matching your schema |

Use `test_fixtures/*.json` as the target schema for stage 2, not as output from this API.

## Setup (GPU server)

```bash
cd docling_marker_project
python -m venv venv
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

python -c "import docling; print('docling ok')"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```bash
export DOCLING_DEVICE=cuda
export DOCLING_LAYOUT_BATCH_SIZE=64
export DOCLING_FORCE_BACKEND_TEXT=true
export DOCLING_TABLE_MODE=accurate

python -m uvicorn service.app:app --host 0.0.0.0 --port 8001
```

Use **`python -m uvicorn`**, not `~/.local/bin/uvicorn`, so the venv that has `docling` is used.

```bash
which python   # .../venv/bin/python
python -m uvicorn service.app:app --host 0.0.0.0 --port 8001
```

Wait until models are loaded:

```bash
curl http://localhost:8001/ready
```

First startup downloads models (~GB). Default pipeline is OCR-off (~6–10 s). OCR-on loads on the first `ocr=true` request.

## API for the next stage

### Endpoints

| Method | Path | Use |
|--------|------|-----|
| `GET` | `/ready` | Call once before batch jobs — must return `200` |
| `POST` | **`/v1/convert`** | **Main API** — upload PDF, get text |
| `GET` | `/health` | Process liveness only |

### `POST /v1/convert`

**Request**

- Body: `multipart/form-data`, field name **`file`** (PDF bytes)
- Query params:

| Param | Default | Description |
|-------|---------|-------------|
| `format` | `text` | `json` recommended for code — see below |
| `ocr` | `off` | `off` \| `auto` \| `full` — see [OCR modes](#ocr-modes) |

### OCR modes

| `ocr` | When to use | Carrier name in **image** on page 1? |
|-------|-------------|--------------------------------------|
| `off` | PDF has selectable text everywhere | No — ignored |
| `auto` | Mixed PDF: text + logos/images | Sometimes — OCR runs only on image regions ≥1% of page |
| `full` | Scanned PDF, or image headers must be read | Yes — re-OCRs entire page (slower; replaces embedded text) |

**Why `ocr=false` and `ocr=true` looked the same:** with the old flag, `true` only ran OCR on large image blocks (default 5% of page). A small carrier logo was skipped, so output matched `off`.

For **Dental/1.pdf**-style image carrier names, try:

```bash
curl -s -X POST "http://localhost:8001/v1/convert?format=json&ocr=auto" \
  -F "file=@test_fixtures/Dental/1.pdf" | jq -r '.text' | head -20

# If still missing, use full-page OCR:
curl -s -X POST "http://localhost:8001/v1/convert?format=json&ocr=full" \
  -F "file=@test_fixtures/Dental/1.pdf" | jq -r '.text' | head -20
```

Check the response header `X-Docling-OCR-Mode` matches what you requested (`off`, `auto`, or `full`).

**Response (`format=json`)** — use this in automated pipelines:

```json
{
  "text": "# Plan summary\n\n| Deductible | $1,650 |\n...",
  "metadata": {
    "status": "SUCCESS",
    "pages": 8,
    "convert_seconds": 3.35,
    "pages_per_second": 2.39,
    "warnings": [],
    "ocr_mode": "off",
    "source_filename": "plan.pdf"
  }
}
```

Pass **`text`** to your LLM or extractor. Ignore structured insurance fields until stage 2.

**Response (`format=text`)** — body is the string only (`text/plain`). Optional headers: `X-Docling-Status`, `X-Docling-Pages`, `X-Docling-Seconds`, `X-Docling-OCR-Mode`.

**Errors**

| Status | Meaning |
|--------|---------|
| `503` | Service or pipeline not ready |
| `422` | Conversion failed |
| `413` | File over `SERVICE_MAX_UPLOAD_MB` |

### Python example (stage 1 → stage 2)

```python
import requests

DOCLING_URL = "http://gpu-host:8001"

def pdf_to_text(pdf_path: str, *, ocr: str = "off") -> str:
    """ocr: off | auto | full"""
    with open(pdf_path, "rb") as f:
        r = requests.post(
            f"{DOCLING_URL}/v1/convert",
            params={"format": "json", "ocr": ocr},
            files={"file": (pdf_path.rsplit("/", 1)[-1], f, "application/pdf")},
            timeout=300,
        )
    r.raise_for_status()
    return r.json()["text"]


def process_plan(pdf_path: str, plan_type: str) -> dict:
    document_text = pdf_to_text(pdf_path, ocr="auto")  # or "full" if logos are images
    # Stage 2: your LLM / API — schema from test_fixtures/{Health,Dental,Vision}/*.json
    return your_extractor(document_text, plan_type=plan_type)
```

### curl

```bash
# Extract text for inspection
curl -s -X POST "http://localhost:8001/v1/convert?format=json&ocr=false" \
  -F "file=@test_fixtures/Health/1.pdf" \
  | jq -r '.text' | head -50
```

### Batch flow

```
1. GET /ready  → 200
2. For each PDF:
     POST /v1/convert?format=json&ocr=off|auto|full  →  text
     your stage-2 service(text, schema)       →  JSON
```

One conversion at a time per process (GPU lock). Scale with multiple replicas if needed.

## Run permanently (systemd)

On Linux GPU servers, use **systemd** so the service starts on boot and restarts after crashes.

**1. Copy and edit the unit file** (paths, user, port):

```bash
cp deploy/docling-pdf.service.example /tmp/docling-pdf.service
nano /tmp/docling-pdf.service   # set User, WorkingDirectory, ExecStart paths, port
sudo cp /tmp/docling-pdf.service /etc/systemd/system/docling-pdf.service
```

**2. Enable and start**

```bash
sudo systemctl daemon-reload
sudo systemctl enable docling-pdf
sudo systemctl start docling-pdf
```

**3. Check status and logs**

```bash
sudo systemctl status docling-pdf
journalctl -u docling-pdf -f
curl http://127.0.0.1:8001/ready
curl http://127.0.0.1:8001/
```

**Useful commands**

| Command | Purpose |
|---------|---------|
| `sudo systemctl restart docling-pdf` | Restart after code/config change |
| `sudo systemctl stop docling-pdf` | Stop service |
| `journalctl -u docling-pdf -n 100` | Last 100 log lines |

**Notes**

- First start can take **several minutes** (model download/load). `TimeoutStartSec=600` in the unit file allows for that.
- Keep **one process per GPU** (`--workers 1`). The app serializes conversions with a lock.
- After `git pull`, run `sudo systemctl restart docling-pdf`.
- Open the port in your firewall only if other machines must call the API (e.g. `ufw allow 8001/tcp`).

**Optional: reverse proxy** — put nginx in front for TLS and auth; proxy `POST /v1/convert` to `http://127.0.0.1:8001`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_DEVICE` | `cuda` | `cuda`, `cpu`, `auto` |
| `DOCLING_LAYOUT_BATCH_SIZE` | `64` | Layout batch size (lower if low VRAM) |
| `DOCLING_FORCE_BACKEND_TEXT` | `true` | Embedded PDF text when OCR is off |
| `DOCLING_TABLE_MODE` | `accurate` | `accurate` or `fast` |
| `SERVICE_MAX_UPLOAD_MB` | `50` | Max upload size |

## Local evaluation (optional)

`benchmark_docling.py` runs Docling on `test_fixtures/` PDFs and reports speed plus field-level text recall vs the JSON fixtures. Use it to tune GPU settings, not as the production API.

```bash
python benchmark_docling.py --device cuda --layout-batch-size 64 --force-backend-text --warmup
```
