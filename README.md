## Testing Notes

All CV, OCR-interface, and extraction logic is covered by `pytest` unit
tests in `tests/`, using synthetic in-memory images — no GPU and no
external network calls required. OCR *execution* itself (via Tesseract)
was verified manually in the development environment where the Tesseract
binary was available; the OCR engine additionally has explicit unit-level
handling (and is designed to be tested) for the "Tesseract not installed"
case, so the app degrades gracefully wherever OCR is unavailable.

---

## Contributors

- **Zulqarnain ([@Mzaq1559](https://github.com/Mzaq1559))** — project owner
- **Claude (Anthropic)** — pair-programmed the full pipeline, test suite,
  and documentation via the GitHub MCP integration
