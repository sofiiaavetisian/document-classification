"""
zones.py

Spatial zone detection for invoice documents.

Invoices follow a fairly predictable layout: the seller is near the top,
the buyer block is introduced by a "bill to" style anchor, metadata like
invoice number and date sit in the upper right, and totals appear near
the bottom right. This module uses OCR bounding box coordinates to assign
lines and words to these zones, which helps the extractor pick the right
candidate when multiple options exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Zone names used throughout the extraction pipeline.
# ---------------------------------------------------------------------------

ZONE_SELLER   = "seller"
ZONE_BUYER    = "buyer"
ZONE_METADATA = "metadata"
ZONE_TOTALS   = "totals"
ZONE_BODY     = "body"
ZONE_UNKNOWN  = "unknown"


@dataclass
class PageZones:
    """
    Holds the line indices that belong to each detected zone for one document.
    Line indices refer to the list of OCR lines passed into detect_zones().
    """
    seller:   List[int] = field(default_factory=list)
    buyer:    List[int] = field(default_factory=list)
    metadata: List[int] = field(default_factory=list)
    totals:   List[int] = field(default_factory=list)
    body:     List[int] = field(default_factory=list)

    def zone_for_line(self, line_index: int) -> str:
        """Return the zone name that contains this line index."""
        if line_index in self.seller:
            return ZONE_SELLER
        if line_index in self.buyer:
            return ZONE_BUYER
        if line_index in self.metadata:
            return ZONE_METADATA
        if line_index in self.totals:
            return ZONE_TOTALS
        if line_index in self.body:
            return ZONE_BODY
        return ZONE_UNKNOWN


def _norm_x(x: float, image_width: int) -> float:
    """Normalize an x coordinate to [0, 1] relative to page width."""
    if image_width <= 0:
        return 0.0
    return max(0.0, min(1.0, x / image_width))


def _norm_y(y: float, image_height: int) -> float:
    """Normalize a y coordinate to [0, 1] relative to page height."""
    if image_height <= 0:
        return 0.0
    return max(0.0, min(1.0, y / image_height))


def detect_zones(
    lines: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    buyer_anchors: Optional[List[str]] = None,
    metadata_anchors: Optional[List[str]] = None,
    total_anchors: Optional[List[str]] = None,
) -> PageZones:
    """
    Assign each OCR line to a spatial zone based on its position on the page
    and any anchor keywords it contains.

    Parameters
    ----------
    lines : list of dicts
        Each dict should have at least 'text', 'top', 'left', 'height', 'width'.
        This is the format produced by src.ocr_engine.load_ocr_lines().
    image_width : int
        Width of the source image in pixels.
    image_height : int
        Height of the source image in pixels.
    buyer_anchors : list of str, optional
        Lowercase strings that signal the start of the buyer block.
    metadata_anchors : list of str, optional
        Lowercase strings that signal the metadata block.
    total_anchors : list of str, optional
        Lowercase strings that signal the totals block.

    Returns
    -------
    PageZones
        Object with lists of line indices for each zone.
    """
    from invoice_rules import RECIPIENT_ANCHORS, INVOICE_NUMBER_ANCHORS, TOTAL_ANCHORS

    if buyer_anchors is None:
        buyer_anchors = RECIPIENT_ANCHORS
    if metadata_anchors is None:
        metadata_anchors = INVOICE_NUMBER_ANCHORS + ["date", "due date", "invoice"]
    if total_anchors is None:
        total_anchors = TOTAL_ANCHORS

    zones = PageZones()

    if not lines or image_width <= 0 or image_height <= 0:
        return zones

    # Find the vertical position of key anchor lines so we can use them
    # as boundaries between zones.
    buyer_anchor_y: Optional[float] = None
    totals_anchor_y: Optional[float] = None

    for line in lines:
        text_lower = str(line.get("text", "")).lower()
        top = float(line.get("top", 0))
        norm_y = _norm_y(top, image_height)

        for anchor in buyer_anchors:
            if anchor in text_lower:
                if buyer_anchor_y is None or norm_y < buyer_anchor_y:
                    buyer_anchor_y = norm_y
                break

        for anchor in total_anchors:
            if anchor in text_lower:
                if totals_anchor_y is None or norm_y < totals_anchor_y:
                    totals_anchor_y = norm_y
                break

    # Fall back to fixed thresholds if no anchors were found.
    # These approximate where zones typically appear on a standard invoice.
    if buyer_anchor_y is None:
        buyer_anchor_y = 0.30
    if totals_anchor_y is None:
        totals_anchor_y = 0.70

    for i, line in enumerate(lines):
        text = str(line.get("text", ""))
        text_lower = text.lower()
        top = float(line.get("top", 0))
        left = float(line.get("left", 0))
        norm_y = _norm_y(top, image_height)
        norm_x = _norm_x(left, image_width)

        # Check if this line contains any zone-specific anchor keywords.
        is_buyer_anchor    = any(a in text_lower for a in buyer_anchors)
        is_metadata_anchor = any(a in text_lower for a in metadata_anchors)
        is_total_anchor    = any(a in text_lower for a in total_anchors)

        if is_total_anchor or norm_y >= totals_anchor_y:
            zones.totals.append(i)

        elif is_buyer_anchor or (norm_y >= buyer_anchor_y and norm_x < 0.55):
            zones.buyer.append(i)

        elif is_metadata_anchor or (norm_y < buyer_anchor_y and norm_x >= 0.45):
            # Metadata tends to sit in the upper-right portion of the page.
            zones.metadata.append(i)

        elif norm_y < buyer_anchor_y and norm_x < 0.45:
            # Upper-left is where the seller/issuer information usually appears.
            zones.seller.append(i)

        else:
            zones.body.append(i)

    return zones


def get_lines_in_zone(lines: List[Dict[str, Any]], zone_indices: List[int]) -> List[Dict[str, Any]]:
    """Return the subset of lines that belong to a given zone."""
    return [lines[i] for i in zone_indices if i < len(lines)]


def lines_to_text(lines: List[Dict[str, Any]]) -> str:
    """Concatenate the text of a list of OCR line dicts into one string."""
    return " ".join(str(line.get("text", "")) for line in lines).strip()


def get_top_lines(lines: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    """
    Return the top n lines sorted by vertical position (top coordinate).
    Useful for finding the issuer name, which is typically one of the first
    prominent lines on the page.
    """
    sorted_lines = sorted(lines, key=lambda l: float(l.get("top", 0)))
    return sorted_lines[:n]


def zone_summary(zones: PageZones) -> Dict[str, int]:
    """Return a simple count of lines per zone, useful for debugging."""
    return {
        ZONE_SELLER:   len(zones.seller),
        ZONE_BUYER:    len(zones.buyer),
        ZONE_METADATA: len(zones.metadata),
        ZONE_TOTALS:   len(zones.totals),
        ZONE_BODY:     len(zones.body),
    }