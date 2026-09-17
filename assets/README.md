# Assets

This folder intentionally does not ship large binary sample images to keep
the repository lightweight and avoid licensing questions around real
document photos.

Instead, **Demo Mode** in the Streamlit app (sidebar → "Demo Mode")
generates three synthetic sample images on the fly, in-memory:

- **Clean document** — a flat, well-lit page.
- **Perspective-distorted** — the same page photographed at an angle.
- **Low-quality** — the same page with an uneven lighting/shadow gradient
  and sensor noise added.

This lets anyone try the full pipeline (detection → perspective correction →
enhancement → OCR → extraction) immediately, with zero setup and no
external downloads.

If you want to test against real photos, just use the regular file
uploader with your own JPG/PNG/WEBP images.
