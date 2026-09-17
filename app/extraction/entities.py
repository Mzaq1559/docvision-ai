"""
Lightweight, regex-based information extraction from OCR text.

This is deliberately simple pattern matching, not NLP/NER -- it is fast,
has zero extra dependencies, and runs comfortably on CPU. Results are
automated guesses and are surfaced to the user as such; nothing here should
be treated as verified or authoritative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Phone numbers: fairly permissive, matches sequences of 7+ digits with
# optional country code, spaces, dashes, or parentheses.
PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}(?:[\s.-]?\d{2,4})?"
)

URL_RE = re.compile(r"(?:https?://|www\.)[^\s,]+")

DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})\b",
    re.IGNORECASE,
)

# CNIC-like: 5 digits - 7 digits - 1 digit (Pakistani CNIC format), also
# accepts a bare 13-digit run since OCR often drops dashes.
CNIC_RE = re.compile(r"\b\d{5}-\d{7}-\d{1}\b|\b\d{13}\b")

INVOICE_RE = re.compile(
    r"\b(?:INV|INVOICE|BILL|RECEIPT|REF|ORDER)[\s#:\-]*[A-Z0-9\-]{3,}\b", re.IGNORECASE
)

AMOUNT_RE = re.compile(
    r"(?:Rs\.?|PKR|USD|\$|€|£)\s?[\d,]+(?:\.\d{1,2})?|"
    r"\b[\d,]+(?:\.\d{2})\b"
)


@dataclass
class ExtractionResult:
    emails: List[str] = field(default_factory=list)
    phone_numbers: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    amounts: List[str] = field(default_factory=list)
    identifiers: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "emails": self.emails,
            "phone_numbers": self.phone_numbers,
            "dates": self.dates,
            "urls": self.urls,
            "amounts": self.amounts,
            "identifiers": self.identifiers,
        }

    @property
    def total_fields(self) -> int:
        return sum(len(v) for v in self.as_dict().values())


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _looks_like_date_or_email(candidate: str) -> bool:
    return bool(DATE_RE.search(candidate)) or "@" in candidate


def extract_entities(text: str) -> ExtractionResult:
    """Extract common document fields from OCR text using regex patterns.

    Returns an ``ExtractionResult`` -- never raises on malformed/empty text.
    """
    if not text:
        return ExtractionResult()

    emails = _dedupe_preserve_order(EMAIL_RE.findall(text))
    urls = _dedupe_preserve_order(URL_RE.findall(text))
    dates = _dedupe_preserve_order(DATE_RE.findall(text))
    identifiers = _dedupe_preserve_order(
        CNIC_RE.findall(text) + INVOICE_RE.findall(text)
    )
    amounts = _dedupe_preserve_order(AMOUNT_RE.findall(text))

    # Phone numbers: filter out matches that are really dates or parts of
    # emails/URLs to reduce false positives from this permissive pattern.
    raw_phones = PHONE_RE.findall(text)
    phone_numbers = _dedupe_preserve_order(
        [p for p in raw_phones if len(re.sub(r"\D", "", p)) >= 7 and not _looks_like_date_or_email(p)]
    )

    return ExtractionResult(
        emails=emails,
        phone_numbers=phone_numbers,
        dates=dates,
        urls=urls,
        amounts=amounts,
        identifiers=identifiers,
    )
