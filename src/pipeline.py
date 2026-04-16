"""
DocumentPipeline — end-to-end document classification + invoice field extraction.

Stage 1: DiT image classifier (discriminative encoder-only vision model).
Stage 2: LayoutLMv3 token classifier for invoice field extraction (runs only
         when Stage 1 predicts 'invoice').

OCR: pytesseract is used on arbitrary input images to produce the words/bboxes
     required by LayoutLMv3.  (FATURA training data used pre-computed HF-format
     annotations, so no OCR was needed during training.)

All models are discriminative only — no generative AI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pytesseract
import torch
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)

# Tesseract binary (Homebrew on macOS; override via env or subclass if needed)
_TESSERACT_CMD = "/opt/homebrew/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

# LayoutLMv3 input constraints
_MAX_LENGTH = 512
_INVOICE_LABEL = "invoice"


class DocumentPipeline:
    """
    Load both models once and run them on demand.

    Parameters
    ----------
    dit_checkpoint  : path to the fine-tuned DiT model directory
    lmv3_checkpoint : path to the fine-tuned LayoutLMv3 checkpoint directory
    label2id_path   : path to label2id.json produced by Notebook 11
    id2label_path   : path to id2label.json produced by Notebook 11
    device          : 'cuda', 'mps', or 'cpu'  (auto-detected if None)
    """

    def __init__(
        self,
        dit_checkpoint: str,
        lmv3_checkpoint: str,
        label2id_path: str,
        id2label_path: str,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # ── Stage 1: DiT classifier ──────────────────────────────────────────
        self._dit_extractor = AutoFeatureExtractor.from_pretrained(dit_checkpoint)
        self._dit_model = AutoModelForImageClassification.from_pretrained(
            dit_checkpoint
        ).to(self.device)
        self._dit_model.eval()
        self.doc_labels: List[str] = list(self._dit_model.config.id2label.values())

        # ── Stage 2: LayoutLMv3 token classifier ────────────────────────────
        with open(label2id_path) as f:
            self._lmv3_label2id: Dict[str, int] = json.load(f)
        with open(id2label_path) as f:
            self._lmv3_id2label: Dict[int, str] = {
                int(k): v for k, v in json.load(f).items()
            }

        self._lmv3_processor = LayoutLMv3Processor.from_pretrained(
            lmv3_checkpoint, apply_ocr=False
        )
        self._lmv3_model = LayoutLMv3ForTokenClassification.from_pretrained(
            lmv3_checkpoint,
            id2label=self._lmv3_id2label,
            label2id=self._lmv3_label2id,
        ).to(self.device)
        self._lmv3_model.eval()

    # ── Public API ───────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> Dict:
        """
        Run the full pipeline on one image.

        Returns
        -------
        {
          'doc_class':  str,          — predicted document class
          'class_conf': float,        — classifier softmax confidence
          'fields':     dict | None,  — extracted fields (None if not invoice)
          'error':      str | None,
        }
        """
        result: Dict = {
            "doc_class":  None,
            "class_conf": None,
            "fields":     None,
            "error":      None,
        }

        path = Path(image_path)
        if not path.exists():
            result["error"] = f"File not found: {image_path}"
            return result

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            result["error"] = f"Cannot open image: {exc}"
            return result

        # Stage 1 — classification
        doc_class, conf = self._classify(image)
        result["doc_class"]  = doc_class
        result["class_conf"] = conf

        # Stage 2 — field extraction (invoice only)
        if doc_class == _INVOICE_LABEL:
            try:
                words, bboxes = self._ocr(image)
                if not words:
                    result["error"] = "No text detected by OCR"
                else:
                    result["fields"] = self._extract_fields(image, words, bboxes)
            except Exception as exc:
                result["error"] = f"Field extraction error: {exc}"

        return result

    def predict_batch(self, image_paths: Sequence[str]) -> List[Dict]:
        """Run predict() on each path and return a list of result dicts."""
        return [self.predict(p) for p in image_paths]

    # ── Private helpers ──────────────────────────────────────────────────────

    def _classify(self, image: Image.Image):
        """Run DiT classifier and return (predicted_class, confidence)."""
        encoding = self._dit_extractor(images=image, return_tensors="pt")
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self._dit_model(**encoding).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_id   = int(probs.argmax())
        pred_class = self._dit_model.config.id2label[pred_id]
        confidence = float(probs[pred_id])
        return pred_class, confidence

    def _ocr(self, image: Image.Image):
        """
        Run Tesseract on the image and return (words, bboxes).

        bboxes are normalised to [0, 1000] for LayoutLMv3.
        """
        img_w, img_h = image.size
        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT, lang="eng"
        )

        words, bboxes = [], []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            if not word:
                continue
            conf = int(data["conf"][i])
            if conf < 0:   # Tesseract returns -1 for block/paragraph markers
                continue

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            # Normalise to [0, 1000]
            x0 = max(0, min(int(x / img_w * 1000), 1000))
            y0 = max(0, min(int(y / img_h * 1000), 1000))
            x1 = max(0, min(int((x + w) / img_w * 1000), 1000))
            y1 = max(0, min(int((y + h) / img_h * 1000), 1000))
            bbox = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

            words.append(word)
            bboxes.append(bbox)

        return words, bboxes

    def _extract_fields(
        self, image: Image.Image, words: List[str], bboxes: List[List[int]]
    ) -> Dict[str, str]:
        """
        Run LayoutLMv3 token classifier and return extracted field values.

        Fields with no detected tokens are omitted from the returned dict.
        """
        encoding = self._lmv3_processor(
            image,
            words,
            boxes=bboxes,
            truncation=True,
            padding="max_length",
            max_length=_MAX_LENGTH,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self._lmv3_model(**encoding).logits
        pred_ids = logits.argmax(-1).squeeze(0).cpu().tolist()

        # LayoutLMv3 processor assigns one token per word for the first
        # sub-word; continuation sub-words get label -100 (we skip those).
        # We walk the word list and pick the prediction for the first
        # sub-token of each word (positions 1..N, skipping CLS/SEP/padding).
        fields: Dict[str, str] = {}
        current_field: Optional[str] = None
        current_tokens: List[str] = []

        # word_ids() maps each token position back to the original word index
        # (None for special tokens).
        word_ids = encoding.get("token_type_ids")  # not what we want
        # Use processor's tokenizer to get word_ids
        tokenizer_output = self._lmv3_processor.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=_MAX_LENGTH,
        )
        token_word_ids = tokenizer_output.word_ids()  # list[int | None]

        seen_word_idx: Optional[int] = None
        for token_pos, word_idx in enumerate(token_word_ids):
            if word_idx is None or word_idx == seen_word_idx:
                # Special token or continuation sub-word — skip
                continue
            seen_word_idx = word_idx

            if word_idx >= len(words):
                break

            label_id = pred_ids[token_pos]
            label    = self._lmv3_id2label.get(label_id, "O")
            word     = words[word_idx]

            if label.startswith("B-"):
                if current_field:
                    fields[current_field] = " ".join(current_tokens)
                current_field  = label[2:]
                current_tokens = [word]
            elif label.startswith("I-") and current_field == label[2:]:
                current_tokens.append(word)
            else:
                if current_field:
                    fields[current_field] = " ".join(current_tokens)
                current_field, current_tokens = None, []

        if current_field:
            fields[current_field] = " ".join(current_tokens)

        return fields
