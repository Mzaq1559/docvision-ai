"""
OCR engine interface.

Wrapping pytesseract behind this small interface means the rest of the app
(Streamlit UI, extraction module) never imports pytesseract directly -- if
we ever swap in a different OCR backend, only this file changes.

Lazy initialization: the Tesseract binary is only probed for on first use,
not at import time, so importing this module never fails or slows down
app startup even in environments where Tesseract isn't installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from app.ocr.preprocessing import prepare_for_ocr


@dataclass
class OcrLine:
    text: str
    confidence: Optional[float] = None


@dataclass
class OcrResult:
    success: bool
    text: str = ""
    lines: List[OcrLine] = field(default_factory=list)
    mean_confidence: Optional[float] = None
    error: Optional[str] = None

    @property
    def character_count(self) -> int:
        return len(self.text)


class OCREngine:
    """Thin, lazily-initialized wrapper around pytesseract."""

    def __init__(self, language: str = "eng") -> None:
        self.language = language
        self._pytesseract = None
        self._available: Optional[bool] = None
        self._unavailable_reason: str = ""

    def _ensure_backend(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import pytesseract  # imported lazily on purpose

            pytesseract.get_tesseract_version()
            self._pytesseract = pytesseract
            self._available = True
        except Exception as exc:  # pytesseract raises various error types
            self._available = False
            self._unavailable_reason = (
                "The Tesseract OCR engine is not installed or not on PATH "
                f"in this environment ({exc.__class__.__name__})."
            )
        return self._available

    @property
    def is_available(self) -> bool:
        return self._ensure_backend()

    def unavailable_reason(self) -> str:
        self._ensure_backend()
        return self._unavailable_reason

    def extract(self, image: np.ndarray) -> OcrResult:
        """Run OCR on ``image`` and return structured text + confidence.

        Never raises: any failure is captured in ``OcrResult.error`` so the
        Streamlit UI can show a friendly message instead of crashing.
        """
        if not self._ensure_backend():
            return OcrResult(success=False, error=self._unavailable_reason)

        try:
            prepared = prepare_for_ocr(image)
            text = self._pytesseract.image_to_string(prepared, lang=self.language)

            lines: List[OcrLine] = []
            try:
                data = self._pytesseract.image_to_data(
                    prepared, lang=self.language, output_type=self._pytesseract.Output.DICT
                )
                confidences = []
                current_line_words: List[str] = []
                current_line_confs: List[float] = []
                last_line_num = None

                for i, word in enumerate(data.get("text", [])):
                    conf_raw = data.get("conf", ["-1"])[i]
                    try:
                        conf = float(conf_raw)
                    except (TypeError, ValueError):
                        conf = -1.0
                    line_num = data.get("line_num", [0])[i]

                    if word.strip() == "":
                        continue
                    if last_line_num is not None and line_num != last_line_num and current_line_words:
                        lines.append(
                            OcrLine(
                                text=" ".join(current_line_words),
                                confidence=(
                                    sum(current_line_confs) / len(current_line_confs)
                                    if current_line_confs
                                    else None
                                ),
                            )
                        )
                        current_line_words, current_line_confs = [], []

                    current_line_words.append(word)
                    if conf >= 0:
                        current_line_confs.append(conf)
                        confidences.append(conf)
                    last_line_num = line_num

                if current_line_words:
                    lines.append(
                        OcrLine(
                            text=" ".join(current_line_words),
                            confidence=(
                                sum(current_line_confs) / len(current_line_confs)
                                if current_line_confs
                                else None
                            ),
                        )
                    )

                mean_confidence = sum(confidences) / len(confidences) if confidences else None
            except Exception:
                # Confidence extraction is a bonus; the app still works
                # without it, using the plain image_to_string result.
                mean_confidence = None

            return OcrResult(
                success=True,
                text=text.strip(),
                lines=lines,
                mean_confidence=mean_confidence,
            )
        except Exception as exc:
            return OcrResult(success=False, error=f"OCR processing failed: {exc}")
