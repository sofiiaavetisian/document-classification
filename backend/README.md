# Document Intelligence Backend

FastAPI backend that connects the DiT document classifier and LayoutLMv3
field extractor to the React frontend.

Invoice extraction now uses a **hybrid parallel architecture**:
- one-stage and two-stage pipelines run in parallel
- fields are merged with fixed routing:
  - `invoice_number` -> two-stage preferred
  - `invoice_date` / `due_date` -> one-stage
  - `issuer_name` -> two-stage with garbage guard
  - `recipient_name` -> one-stage
  - `total_amount` -> two-stage with currency symbol preservation

## Prerequisites

### System packages

```bash
# macOS
brew install tesseract poppler

# Ubuntu / Debian
sudo apt install tesseract-ocr poppler-utils
```

`tesseract` is required for OCR on image invoices.
`poppler` is required by `pdf2image` to rasterise scanned PDFs.

### Python

Python 3.11 (matches the `.venv311` training environment).

## Install

```bash
cd backend
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
# .env already has the correct values — no edits needed for a local run
```

## Run

**Important:** `TOKENIZERS_PARALLELISM` must be set to `false` before
uvicorn starts. The `.env` file handles this when loaded, but the safest
approach is to set it explicitly on the command line:

```bash
TOKENIZERS_PARALLELISM=false uvicorn main:app --reload --port 8000
```

Or source the `.env` file first:

```bash
set -a && source .env && set +a
uvicorn main:app --reload --port 8000
```

## Test the health endpoint

```bash
curl http://localhost:8000/health
```

Expected response when both models are loaded:

```json
{
  "status": "ok",
  "device": "mps",
  "models_loaded": true
}
```

## Test a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/invoice.pdf" | python3 -m json.tool
```

## Expected startup output

```
[pipeline] device: mps
[pipeline] loading DiT classifier ...
[pipeline] DiT loaded OK
[pipeline] loading LayoutLMv3 ...
[pipeline] LayoutLMv3 loaded OK
[pipeline] InvoiceCleaner ready
[pipeline] startup complete — all models loaded
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Both models load at startup (~10–25 seconds on Apple Silicon). If either
model fails to load the server starts but every `/predict` call returns a
503 with the error message — fix the problem and restart.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Liveness check |
| POST | `/predict` | Classify document + extract invoice fields |

### POST /predict

- Content-Type: `multipart/form-data`
- Field name: `file`
- Accepted formats: `jpg jpeg png tif tiff bmp webp avif pdf`

#### Response schema

```json
{
  "predicted_class": "invoice",
  "confidence": 0.97,
  "all_probabilities": {
    "invoice": 0.97,
    "form": 0.01,
    "budget": 0.01,
    "email": 0.005,
    "resume": 0.005
  },
  "is_invoice": true,
  "fields": {
    "invoice_number": "INV-2024-001",
    "invoice_date": "22-Aug-1995",
    "due_date": "14-Jan-2001",
    "issuer_name": "Acme Corp",
    "recipient_name": "John Smith",
    "total_amount": "1098.28 USD"
  },
  "processing_mode": "hybrid_parallel_two_stage",
  "filename": "invoice.pdf",
  "error": null
}
```

`processing_mode` is `"hybrid_parallel_two_stage"` for images and scanned PDFs,
`"native_pdf_text"` for PDFs with extractable text.

For non-invoice documents `fields` is `null`.

## Notes on AVIF support

AVIF images are tried with Pillow first. If Pillow was not compiled with
AVIF support, the backend falls back to ImageMagick's `convert` command
(`brew install imagemagick`). Alternatively install the Pillow plugin:

```bash
pip install pillow-avif-plugin
```
