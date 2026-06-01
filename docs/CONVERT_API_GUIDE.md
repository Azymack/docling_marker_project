# Convert API guide

How to call **`POST /v1/convert`** on the Docling PDF service.

**Base URL (example):** `http://gpu-host:8001`  
**Interactive docs:** `http://gpu-host:8001/docs`

---

## What this API does

Upload a **PDF** → receive **plain text** (markdown-style) for your next step (LLM, rules, database, etc.).

It does **not** return structured insurance JSON (`Carrier Name`, deductibles, …). That is a separate stage in your pipeline.

```
PDF file  →  POST /v1/convert  →  text string  →  your extractor  →  JSON
```

---

## Before you call convert

### 1. Check the service is ready

```bash
curl -s http://gpu-host:8001/ready
```

Expect HTTP **200** and `"status": "ready"`.

If you get **503**, wait for models to finish loading (first start can take several minutes).

### 2. Confirm you hit the right service (optional)

```bash
curl -s http://gpu-host:8001/ | jq .
```

Expect:

- `"service": "docling_marker_project"`
- `"ocr_modes": ["off", "auto", "full"]`

---

## `POST /v1/convert`

### Request

| Part | Value |
|------|--------|
| **Method** | `POST` |
| **Path** | `/v1/convert` |
| **Content-Type** | `multipart/form-data` |
| **Body field** | `file` — the PDF bytes (required) |

### Query parameters

| Parameter | Default | Values | Description |
|-----------|---------|--------|-------------|
| `format` | `text` | `text`, `json` | Response shape (see below) |
| `ocr` | `off` | `off`, `auto`, `full` | How text is extracted (see OCR modes) |

**Aliases for `ocr`:** `false` → `off`, `true` → `auto`.

### OCR modes

| `ocr` | Use when | Notes |
|-------|----------|--------|
| `off` | PDF has selectable/copyable text | Fastest. **Skips text inside images** (logos, scanned headers). |
| `auto` | Mixed PDF: normal text + images/logos | OCR on image regions (about ≥1% of page area). Try this for carrier names in graphics. |
| `full` | Fully scanned PDF, or `auto` still misses text | Re-OCRs the **whole page**. Slower; replaces embedded text with OCR on each page. |

**Choosing `ocr` for insurance PDFs**

- Mostly text, all fields selectable → `ocr=off`
- Text body + carrier logo as image on page 1 → try `ocr=auto`, then `ocr=full` if needed
- Entire document is a scan → `ocr=full`

---

## Response formats

### `format=json` (recommended for code)

**Content-Type:** `application/json`

```json
{
  "text": "# Plan name\n\n| Deductible | $1,650 |\n...",
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

Use **`text`** as input to your next pipeline stage.

| `metadata` field | Meaning |
|------------------|---------|
| `status` | `SUCCESS` or `PARTIAL_SUCCESS` (some pages failed) |
| `pages` | Page count processed |
| `convert_seconds` | Processing time |
| `ocr_mode` | Actual mode used: `off`, `auto`, or `full` |
| `warnings` | e.g. `partial_success_some_pages_failed` |

### `format=text` (default)

**Content-Type:** `text/plain`

The response **body is only the extracted string** (no JSON wrapper).

Useful headers:

| Header | Example |
|--------|---------|
| `X-Docling-Status` | `SUCCESS` |
| `X-Docling-Pages` | `8` |
| `X-Docling-Seconds` | `3.35` |
| `X-Docling-OCR-Mode` | `off` |

---

## Examples

### curl — JSON (typical for pipelines)

```bash
curl -s -X POST "http://gpu-host:8001/v1/convert?format=json&ocr=off" \
  -F "file=@/path/to/plan.pdf"
```

Save only the text:

```bash
curl -s -X POST "http://gpu-host:8001/v1/convert?format=json&ocr=off" \
  -F "file=@plan.pdf" \
  | jq -r '.text' > extracted.txt
```

With OCR for image headers:

```bash
curl -s -X POST "http://gpu-host:8001/v1/convert?format=json&ocr=auto" \
  -F "file=@plan.pdf" \
  | jq -r '.text'
```

### curl — plain text body

```bash
curl -s -X POST "http://gpu-host:8001/v1/convert?ocr=off" \
  -F "file=@plan.pdf"
```

### Python (requests)

```python
import requests

DOCLING_URL = "http://gpu-host:8001"

def pdf_to_text(pdf_path: str, *, ocr: str = "off", timeout: int = 300) -> str:
    """
    ocr: "off" | "auto" | "full"
    Returns extracted markdown/plain text.
    """
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{DOCLING_URL}/v1/convert",
            params={"format": "json", "ocr": ocr},
            files={"file": (pdf_path.rsplit("/", 1)[-1], f, "application/pdf")},
            timeout=timeout,
        )
    response.raise_for_status()
    return response.json()["text"]


# Example
text = pdf_to_text("plan.pdf", ocr="auto")
# next: send `text` to your LLM / extractor
```

### Python — check readiness first

```python
import requests

base = "http://gpu-host:8001"

ready = requests.get(f"{base}/ready", timeout=30)
ready.raise_for_status()

with open("plan.pdf", "rb") as f:
    r = requests.post(
        f"{base}/v1/convert",
        params={"format": "json", "ocr": "off"},
        files={"file": ("plan.pdf", f, "application/pdf")},
        timeout=300,
    )
r.raise_for_status()
data = r.json()
print(data["metadata"]["ocr_mode"], data["metadata"]["convert_seconds"])
document_text = data["text"]
```

### JavaScript (fetch)

```javascript
const form = new FormData();
form.append("file", pdfFile); // File from <input type="file">

const res = await fetch(
  "http://gpu-host:8001/v1/convert?format=json&ocr=off",
  { method: "POST", body: form }
);
if (!res.ok) throw new Error(await res.text());

const { text, metadata } = await res.json();
console.log(metadata.ocr_mode, metadata.convert_seconds);
// use `text` in your next stage
```

### Swagger UI

1. Open `http://gpu-host:8001/docs`
2. Expand **conversion** → **POST /v1/convert**
3. Click **Try it out**
4. Upload a PDF under **file**
5. Set **ocr** to `off`, `auto`, or `full`
6. Set **format** to `json` or `text`
7. Click **Execute**

---

## HTTP status codes

| Code | Meaning | What to do |
|------|---------|------------|
| **200** | Success | Read `text` or response body |
| **400** | Bad request (not PDF, empty file, invalid `ocr`) | Fix request |
| **413** | File too large | Default max 50 MB (`SERVICE_MAX_UPLOAD_MB`) |
| **422** | Conversion failed | See `detail` in JSON body |
| **500** | Server error | Check service logs (`journalctl -u docling-pdf`) |
| **503** | Not ready | Wait and retry `GET /ready` |

**422 example**

```json
{
  "detail": {
    "message": "Conversion failed with status FAILED",
    "status": "FAILED",
    "pages": 0
  }
}
```

---

## Recommended workflow (batch / production)

```
1. GET  /ready          → wait for 200
2. POST /v1/convert     → format=json, ocr=off|auto|full
3. Read response.text   → pass to your extractor
4. Repeat step 2 per PDF
```

**Timeouts:** allow **60–300 seconds** per document depending on page count and `ocr` mode.

**Concurrency:** one conversion runs at a time per service process (GPU lock). For more throughput, run multiple service instances on different ports/GPUs.

---

## Troubleshooting

| Problem | Check |
|---------|--------|
| Internal Server Error | `journalctl -u docling-pdf -f` while reproducing |
| Same output for all `ocr` values | Response header `X-Docling-OCR-Mode` — must change per request |
| Missing carrier name in image | Try `ocr=auto`, then `ocr=full` |
| Old API in `/docs` (boolean `ocr`) | `curl http://host:8001/` — confirm `ocr_modes` list exists |
| Connection refused | Service not running or wrong port |

---

## Related endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Service info, build id, endpoint list |
| `GET` | `/health` | Liveness (process up) |
| `GET` | `/ready` | Models loaded, ready for traffic |
| `POST` | `/v1/convert` | **Convert PDF → text** |

Server setup and systemd: see [README.md](../README.md).
