import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC_DIR))

from extraction_improvements import clean_invoice_number


class CleanInvoiceNumberTests(unittest.TestCase):
    def test_prefers_invoice_label_over_po_and_address_number(self):
        doc_context = {
            'raw_text': (
                'Invoice # US-001 '\
                'P.O.# 2312/2019 '\
                'Address: 1912 Harvest Lane'
            ),
            'labelled_fields': {
                'Invoice #': 'US-001',
                'P.O.#': '2312/2019',
            },
            'ocr_candidates': ['1912', 'US-001', '2312/2019'],
        }
        self.assertEqual(clean_invoice_number('1912', doc_context), 'US-001')

    def test_rejects_label_only_invoice_word(self):
        doc_context = {
            'raw_text': 'INVOICE DATE: 2024-03-10',
            'labelled_fields': {},
            'ocr_candidates': [],
        }
        self.assertEqual(clean_invoice_number('INVOICE', doc_context), '—')

    def test_rejects_address_fragment_bare_digits(self):
        doc_context = {
            'raw_text': 'Bill To: ACME Corp, 1912 Harvest Lane',
            'labelled_fields': {},
            'ocr_candidates': ['1912'],
        }
        self.assertEqual(clean_invoice_number('1912', doc_context), '—')

    def test_keeps_structured_labelled_invoice_number(self):
        doc_context = {
            'raw_text': 'Invoice No: INV-12345-1',
            'labelled_fields': {'Invoice No': 'INV-12345-1'},
            'ocr_candidates': ['INV-12345-1'],
        }
        self.assertEqual(clean_invoice_number('INV-12345-1', doc_context), 'INV-12345-1')


if __name__ == '__main__':
    unittest.main()
