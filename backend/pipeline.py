"""
pipeline.py
===========
DocumentPipeline: loads both models once at startup, exposes predict().

IMPORTANT: TOKENIZERS_PARALLELISM must be set before any transformers import.
This file sets it at the very top via os.environ so it takes effect even when
the module is imported before the caller has set the environment variable.
"""

import os

# Must be set before any transformers / tokenizers import — macOS Python 3.9+
# deadlock prevention for the fast LayoutLMv3 tokenizer.
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ── Stdlib ─────────────────────────────────────────────────────────────────
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ── Third-party ────────────────────────────────────────────────────────────
import numpy as np
import pytesseract
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)

# ── Local ──────────────────────────────────────────────────────────────────
# Path layout:
#   document-classification/       ← _DOC_CLASS
#   ├── backend/                   ← this file lives here (_BACKEND_DIR)
#   ├── src/invoice_cleaner.py
#   └── models/experimental/...

_BACKEND_DIR  = Path(__file__).parent
_DOC_CLASS    = _BACKEND_DIR.parent   # document-classification/
_SRC_DIR      = str(_DOC_CLASS / 'src')

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from invoice_cleaner import InvoiceCleaner  # noqa: E402  (must follow sys.path insert)
from hybrid_field_extractor import HybridInvoiceFieldExtractor  # noqa: E402
from extraction_improvements import warmup_doctr_predictor  # noqa: E402

# ── Model paths ────────────────────────────────────────────────────────────
_DIT_DIR        = _DOC_CLASS / 'models' / 'experimental' / 'dit_fatura'
_DIT_STATE_DICT = _DIT_DIR / 'dit_fatura_state_dict.pt'
_DIT_CONFIG     = _DIT_DIR / 'dit_fatura_training_config.json'
_LLM_CKPT       = _DOC_CLASS / 'models' / 'experimental' / 'layoutlmv3_fatura' / 'best_checkpoint'
_LABEL2ID_PATH  = _DOC_CLASS / 'data' / 'processed' / 'layoutlmv3_dataset' / 'label2id.json'
_ID2LABEL_PATH  = _DOC_CLASS / 'data' / 'processed' / 'layoutlmv3_dataset' / 'id2label.json'

# DiT base model identifier — must be present in the HuggingFace local cache.
# The notebook downloaded this once from the Hub; subsequent loads use
# local_files_only=True to prevent any network access at inference time.
_DIT_BASE_MODEL = 'microsoft/dit-large-finetuned-rvlcdip'

# ── DiT label set (order matches training config) ──────────────────────────
_DIT_LABELS  = ['invoice', 'form', 'resume', 'email', 'budget']
_DIT_LABEL2ID = {l: i for i, l in enumerate(_DIT_LABELS)}
_DIT_ID2LABEL = {i: l for l, i in _DIT_LABEL2ID.items()}

# ── LayoutLMv3 constants ───────────────────────────────────────────────────
_MAX_LENGTH  = 512
_OCR_CONF_THRESHOLD = 10  # minimum Tesseract confidence score (0–100)

# Mapping from model BIO uppercase keys to InvoiceCleaner lowercase keys
_CLEAN_KEY = {
    'INVOICE_NUMBER': 'invoice_number',
    'INVOICE_DATE':   'invoice_date',
    'DUE_DATE':       'due_date',
    'ISSUER_NAME':    'issuer_name',
    'RECIPIENT_NAME': 'recipient_name',
    'TOTAL_AMOUNT':   'total_amount',
}


def _detect_device() -> torch.device:
    """Auto-detect the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class DocumentPipeline:
    """
    Loads DiT classifier and LayoutLMv3 extractor once at startup.

    Both models are moved to the best available device (CUDA / MPS / CPU)
    and set to eval mode. All DataLoader operations use num_workers=0 to
    avoid macOS multiprocessing deadlocks (not applicable here since we run
    single-image inference, but noted for consistency).

    Usage
    -----
        pipeline = DocumentPipeline()
        result = pipeline.predict(pil_image, 'invoice.jpg')
    """

    def __init__(self) -> None:
        self.device = _detect_device()
        print(f'[pipeline] device: {self.device}')

        self._load_dit()
        self._load_layoutlmv3()

        self.cleaner = InvoiceCleaner()
        self.hybrid_invoice_extractor = HybridInvoiceFieldExtractor(
            model=self.layoutlm_model,
            processor=self.layoutlm_processor,
            device=self.device,
            id2label=self._ner_id2label,
            cleaner=self.cleaner,
            ocr_engine='doctr',
            max_length=_MAX_LENGTH,
        )
        print('[pipeline] HybridInvoiceFieldExtractor ready')
        if warmup_doctr_predictor():
            print('[pipeline] DocTR predictor warmup OK (cached)')
        else:
            print('[pipeline] DocTR predictor unavailable — Tesseract fallback will be used')
        print('[pipeline] InvoiceCleaner ready')
        print('[pipeline] startup complete — all models loaded')

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_dit(self) -> None:
        """
        Load DiT classifier.

        The weights are stored as a raw PyTorch state dict (not a full HF
        checkpoint) because the training notebook saved only the best-epoch
        weights to keep the artefact small. We reconstruct the architecture
        from the HF Hub base model (must be in the local cache) and apply
        the fine-tuned state dict on top.
        """
        if not _DIT_STATE_DICT.exists():
            raise FileNotFoundError(
                f'DiT state dict not found: {_DIT_STATE_DICT}\n'
                'Run notebook 10b to train and save the model first.'
            )

        print('[pipeline] loading DiT classifier ...')
        self.dit_processor = AutoImageProcessor.from_pretrained(
            _DIT_BASE_MODEL,
            local_files_only=True,
        )

        self.dit_model = AutoModelForImageClassification.from_pretrained(
            _DIT_BASE_MODEL,
            num_labels=len(_DIT_LABELS),
            id2label=_DIT_ID2LABEL,
            label2id=_DIT_LABEL2ID,
            ignore_mismatched_sizes=True,  # classifier head has 5 outputs, base has 16
            local_files_only=True,
        )

        state = torch.load(str(_DIT_STATE_DICT), map_location=self.device)
        self.dit_model.load_state_dict(state)
        self.dit_model.to(self.device)
        self.dit_model.eval()
        print('[pipeline] DiT loaded OK')

    def _load_layoutlmv3(self) -> None:
        """
        Load LayoutLMv3 field extractor from the full HF checkpoint directory.

        use_fast=True is required — use_fast=False causes a deadlock on
        macOS Python 3.9+ due to a known issue in the slow tokenizer's
        multiprocessing behaviour.
        """
        if not _LLM_CKPT.exists():
            raise FileNotFoundError(
                f'LayoutLMv3 checkpoint not found: {_LLM_CKPT}\n'
                'Run notebook 12 to train and save the model first.'
            )

        print('[pipeline] loading LayoutLMv3 ...')
        self.layoutlm_processor = LayoutLMv3Processor.from_pretrained(
            str(_LLM_CKPT),
            apply_ocr=False,
            use_fast=True,      # must be True — see docstring
            local_files_only=True,
        )

        self.layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained(
            str(_LLM_CKPT),
            local_files_only=True,
        )
        self.layoutlm_model.to(self.device)
        self.layoutlm_model.eval()

        # Load NER label maps from the dataset directory so they match
        # exactly what the model was trained with.
        with open(_ID2LABEL_PATH) as f:
            self._ner_id2label: Dict[int, str] = {
                int(k): v for k, v in json.load(f).items()
            }

        print('[pipeline] LayoutLMv3 loaded OK')

    # ── Public API ─────────────────────────────────────────────────────────

    def predict(self, image: Image.Image, filename: str) -> dict:
        """
        Run the full classification + optional extraction pipeline on one image.

        Parameters
        ----------
        image : PIL.Image (RGB)
            The document image. For PDFs, pass the first page rendered as RGB.
        filename : str
            Original filename — included in the response for the frontend.

        Returns
        -------
        dict matching the API response schema:
            predicted_class, confidence, all_probabilities,
            is_invoice, fields, processing_mode, filename, error
        """
        image = image.convert('RGB')

        predicted_class, confidence, all_probs = self._classify(image)

        result: dict = {
            'predicted_class':   predicted_class,
            'confidence':        confidence,
            'all_probabilities': all_probs,
            'is_invoice':        predicted_class == 'invoice',
            'fields':            None,
            'processing_mode':   'layoutlmv3',
            'filename':          filename,
            'error':             None,
        }

        if predicted_class == 'invoice':
            try:
                # Primary production path:
                # one-stage + two-stage in parallel, then field-wise routing.
                cleaned = self.hybrid_invoice_extractor.extract(image)
                result['fields'] = cleaned
                result['processing_mode'] = 'hybrid_parallel_two_stage'
            except Exception as exc:
                # Safety fallback: legacy one-stage extraction keeps service alive.
                words, boxes = self._ocr(image)
                raw_fields = self._extract_fields_layoutlmv3(image, words, boxes)
                cleaned = self.cleaner.clean(raw_fields, ocr_words=words)
                result['fields'] = cleaned
                result['processing_mode'] = 'layoutlmv3_fallback'
                result['error'] = (
                    'Hybrid extraction failed; used fallback one-stage path. '
                    f'Cause: {exc.__class__.__name__}: {exc}'
                )

        return result

    # ── DiT inference ──────────────────────────────────────────────────────

    def _classify(
        self, image: Image.Image
    ) -> Tuple[str, float, Dict[str, float]]:
        """Run DiT classifier. Returns (label, confidence, all_probs_dict)."""
        enc = self.dit_processor(images=image, return_tensors='pt')
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out   = self.dit_model(**enc)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]

        pred_idx   = int(np.argmax(probs))
        pred_label = _DIT_ID2LABEL[pred_idx]
        confidence = float(probs[pred_idx])
        all_probs  = {_DIT_ID2LABEL[i]: float(probs[i]) for i in range(len(_DIT_LABELS))}

        return pred_label, confidence, all_probs

    # ── OCR ────────────────────────────────────────────────────────────────

    def _ocr(
        self, image: Image.Image
    ) -> Tuple[List[str], List[List[int]]]:
        """
        Run Tesseract OCR on *image* and return (words, normalised_bboxes).

        Bounding boxes are normalised to [0, 1000] as required by LayoutLMv3.
        Words with Tesseract confidence < _OCR_CONF_THRESHOLD are discarded
        (matches notebook 13 exactly — threshold is 10).

        If OCR finds nothing (blank page, pure image), returns a single
        dummy word so the LayoutLMv3 encoder does not receive an empty input.
        """
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words: List[str]       = []
        boxes: List[List[int]] = []
        w, h = image.size

        for i, text in enumerate(data['text']):
            text = str(text).strip()
            if not text or int(data['conf'][i]) < _OCR_CONF_THRESHOLD:
                continue

            left, top     = data['left'][i],  data['top'][i]
            width, height = data['width'][i], data['height'][i]

            x0 = int(max(0, min(1000, round(left            / w * 1000))))
            y0 = int(max(0, min(1000, round(top             / h * 1000))))
            x1 = int(max(0, min(1000, round((left + width)  / w * 1000))))
            y1 = int(max(0, min(1000, round((top  + height) / h * 1000))))

            # Ensure non-degenerate box
            if x1 <= x0:
                x1 = min(1000, x0 + 1)
            if y1 <= y0:
                y1 = min(1000, y0 + 1)

            words.append(text)
            boxes.append([x0, y0, x1, y1])

        return (words or ['empty']), (boxes or [[0, 0, 1, 1]])

    # ── LayoutLMv3 inference ───────────────────────────────────────────────

    def _extract_fields_layoutlmv3(
        self,
        image: Image.Image,
        words: List[str],
        boxes: List[List[int]],
    ) -> Dict[str, str]:
        """
        Run LayoutLMv3 token classifier and assemble BIO predictions into
        field strings (keys are uppercase, e.g. 'INVOICE_NUMBER').

        This is the exact get_raw_predictions() logic from notebook 13 —
        first-subword-per-word mapping, BIO state machine.
        Returns raw (uncleaned) field strings; cleaning happens in predict().
        """
        encoding = self.layoutlm_processor(
            image, words, boxes=boxes,
            truncation=True,
            padding='max_length',
            max_length=_MAX_LENGTH,
            return_tensors='pt',
        )

        with torch.no_grad():
            outputs = self.layoutlm_model(
                **{k: v.to(self.device) for k, v in encoding.items()}
            )

        token_preds = outputs.logits.argmax(-1).squeeze(0).cpu().tolist()
        word_ids    = encoding.word_ids(batch_index=0)

        # Map subword tokens → word level (take first subword per word)
        word_preds: Dict[int, int] = {}
        for ti, wi in enumerate(word_ids):
            if wi is not None and wi not in word_preds:
                word_preds[wi] = token_preds[ti]

        aligned_words    = [words[i]      for i in sorted(word_preds)]
        aligned_pred_ids = [word_preds[i] for i in sorted(word_preds)]

        # BIO state machine → field strings
        fields:         Dict[str, str] = {}
        current_field:  str | None     = None
        current_tokens: List[str]      = []

        for label_id, word in zip(aligned_pred_ids, aligned_words):
            label = self._ner_id2label[label_id]

            if label == 'O':
                if current_field:
                    text = ' '.join(current_tokens).strip()
                    if text:
                        fields[current_field] = text
                    current_field, current_tokens = None, []

            elif label.startswith('B-'):
                if current_field:
                    text = ' '.join(current_tokens).strip()
                    if text:
                        fields[current_field] = text
                current_field  = label[2:]
                current_tokens = [word]

            elif label.startswith('I-'):
                fn = label[2:]
                if current_field == fn:
                    current_tokens.append(word)
                elif current_field is None and fn in fields:
                    fields[fn] += ' ' + word
                elif current_field is None:
                    current_field, current_tokens = fn, [word]
                else:
                    # Label switch without B-tag — flush current, start new
                    text = ' '.join(current_tokens).strip()
                    if text:
                        fields[current_field] = text
                    current_field, current_tokens = fn, [word]

        # Flush final field
        if current_field:
            text = ' '.join(current_tokens).strip()
            if text:
                fields[current_field] = text

        return fields
