"""
main.py
=======
FastAPI application for the Document Intelligence backend.

Routes
------
GET  /              — service info
GET  /health        — liveness check (reports device + model status)
POST /predict       — multipart file upload → classification + field extraction

Start with:
    TOKENIZERS_PARALLELISM=false uvicorn main:app --reload --port 8000
"""

import os
import sys
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Must be set before pipeline (and therefore transformers) is imported.
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Image / PDF helpers ────────────────────────────────────────────────────
from PIL import Image

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None  # handled at request time with a clear error

from pdf_extractor import (
    _extract_fields_from_text,
    extract_text_from_pdf,
    is_native_pdf,
)
from pipeline import DocumentPipeline

# ── Accepted file extensions ───────────────────────────────────────────────
_ACCEPTED_EXTENSIONS = frozenset({
    '.jpg', '.jpeg', '.png', '.tif', '.tiff',
    '.bmp', '.webp', '.avif', '.pdf',
})

# ── Application state ──────────────────────────────────────────────────────
# Both models are loaded once at startup and shared across all requests.
# Loading per-request would add ~10–30 seconds of latency each time.
_pipeline: Optional[DocumentPipeline] = None
_startup_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both models before the server starts accepting requests."""
    global _pipeline, _startup_error
    try:
        _pipeline = DocumentPipeline()
    except Exception as exc:
        _startup_error = str(exc)
        # Print the error loudly so the operator sees it immediately.
        print(f'\n[FATAL] Model loading failed: {exc}', file=sys.stderr)
        print('The application will start but every /predict call will return an error.', file=sys.stderr)
        print('Fix the model path / environment and restart.\n', file=sys.stderr)
    yield
    # Nothing to clean up on shutdown.


# ── FastAPI application ────────────────────────────────────────────────────
app = FastAPI(
    title='Document Intelligence API',
    version='1.0',
    lifespan=lifespan,
)

# CORS — allow all origins so the Vite dev server (localhost:5173) and any
# other origin can reach the backend without CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── Utility helpers ────────────────────────────────────────────────────────

def _error_response(
    message: str,
    filename: str = '',
    status_code: int = 400,
) -> JSONResponse:
    """Return a well-formed error JSON response (never raw FastAPI HTML)."""
    body = {
        'predicted_class':   None,
        'confidence':        None,
        'all_probabilities': None,
        'is_invoice':        False,
        'fields':            None,
        'processing_mode':   None,
        'filename':          filename,
        'error':             message,
    }
    return JSONResponse(content=body, status_code=status_code)


def _load_avif(path: Path) -> Image.Image:
    """
    Try to open an AVIF file with Pillow first; fall back to ImageMagick's
    `convert` command if Pillow does not have AVIF support compiled in.
    Raises RuntimeError if neither method works.
    """
    try:
        return Image.open(path).convert('RGB')
    except Exception:
        pass

    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.check_call(
            ['convert', str(path), tmp_path],
            stderr=subprocess.DEVNULL,
        )
        img = Image.open(tmp_path).convert('RGB')
        os.unlink(tmp_path)
        return img
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            f'Cannot open AVIF file {path.name}. '
            'Install pillow-avif-plugin (pip install pillow-avif-plugin) '
            'or ImageMagick (brew install imagemagick).'
        )


def _load_image_from_upload(data: bytes, suffix: str) -> Image.Image:
    """
    Convert raw uploaded bytes into a PIL RGB image.

    Handles: JPEG, PNG, TIFF, BMP, WebP, AVIF (with fallback), PDF (page 1).
    For PDFs we check for native text first; if scanned we rasterise with
    pdf2image at 200 DPI.
    """
    suffix = suffix.lower()

    # Write bytes to a temp file so we can pass a path to pdf2image / fitz.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        if suffix == '.avif':
            return _load_avif(tmp_path)

        if suffix == '.pdf':
            # Rasterise first page only — multi-page PDFs are supported but
            # we classify and extract from page 1 only (per spec).
            if convert_from_path is None:
                raise RuntimeError(
                    'pdf2image is not installed. Run: pip install pdf2image\n'
                    'Also install poppler: brew install poppler'
                )
            pages = convert_from_path(str(tmp_path), dpi=200, first_page=1, last_page=1)
            if not pages:
                raise ValueError('pdf2image returned no pages.')
            return pages[0].convert('RGB')

        return Image.open(tmp_path).convert('RGB')

    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get('/')
def root():
    return {'message': 'Document Intelligence API', 'version': '1.0'}


@app.get('/health')
def health():
    if _startup_error:
        return JSONResponse(
            content={
                'status':        'error',
                'device':        'unknown',
                'models_loaded': False,
                'error':         _startup_error,
            },
            status_code=503,
        )
    device_str = str(_pipeline.device) if _pipeline else 'unknown'
    return {
        'status':        'ok',
        'device':        device_str,
        'models_loaded': _pipeline is not None,
    }


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    Classify a document image and (for invoices) extract structured fields.

    Accepts multipart/form-data with a single `file` field.

    Pipeline:
      1. Validate extension
      2. If native-text PDF → pymupdf extraction → regex fields (no ML)
      3. Otherwise → DiT classifier → (if invoice) OCR + LayoutLMv3 + InvoiceCleaner
    """
    filename = file.filename or 'upload'
    suffix   = Path(filename).suffix.lower()

    # ── Extension validation ───────────────────────────────────────────────
    if suffix not in _ACCEPTED_EXTENSIONS:
        return _error_response(
            f'Unsupported file type "{suffix}". '
            f'Accepted formats: {", ".join(sorted(_ACCEPTED_EXTENSIONS))}.',
            filename=filename,
            status_code=400,
        )

    # ── Model availability check ───────────────────────────────────────────
    if _pipeline is None:
        return _error_response(
            f'Models failed to load at startup: {_startup_error}',
            filename=filename,
            status_code=503,
        )

    # ── Read file bytes ────────────────────────────────────────────────────
    data = await file.read()
    if not data:
        return _error_response('Uploaded file is empty.', filename=filename)

    # ── Native PDF fast-path ───────────────────────────────────────────────
    if suffix == '.pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        try:
            if is_native_pdf(str(tmp_path)):
                text   = extract_text_from_pdf(str(tmp_path))
                fields = _extract_fields_from_text(text)
                return {
                    'predicted_class':   'invoice',
                    'confidence':        1.0,
                    'all_probabilities': {
                        'invoice': 1.0,
                        'form':    0.0,
                        'budget':  0.0,
                        'email':   0.0,
                        'resume':  0.0,
                    },
                    'is_invoice':      True,
                    'fields':          fields or None,
                    'processing_mode': 'native_pdf_text',
                    'filename':        filename,
                    'error':           None,
                }
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    # ── Load image (rasterise scanned PDFs + all image formats) ───────────
    try:
        image = _load_image_from_upload(data, suffix)
    except Exception as exc:
        return _error_response(
            f'Could not open image: {exc}',
            filename=filename,
            status_code=422,
        )

    # ── ML pipeline ───────────────────────────────────────────────────────
    try:
        result = _pipeline.predict(image, filename)
    except Exception as exc:
        return _error_response(
            f'Inference error: {exc}',
            filename=filename,
            status_code=500,
        )

    return result
