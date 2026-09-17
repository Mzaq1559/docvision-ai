"""
DocVision AI - Streamlit entry point.

Run locally with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import streamlit as st

# Allow running via `streamlit run app/streamlit_app.py` from the repo root
# as well as from within the app/ directory, by making sure the project
# root is importable.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.config.settings import load_settings  # noqa: E402
from app.cv.detector import detect_document, draw_detection  # noqa: E402
from app.cv.enhancement import enhance_document  # noqa: E402
from app.cv.perspective import four_point_transform  # noqa: E402
from app.extraction.entities import extract_entities  # noqa: E402
from app.ocr.engine import OCREngine  # noqa: E402
from app.utils.image import (  # noqa: E402
    bgr_to_rgb,
    decode_image_bytes,
    encode_image_to_bytes,
    resize_max_width,
)
from app.utils.metrics import ProcessingStats, Timer, document_area_ratio  # noqa: E402
from app.utils.validation import (  # noqa: E402
    validate_file_extension,
    validate_file_size,
    validate_image_array,
)

st.set_page_config(page_title="DocVision AI", page_icon="\U0001F4C4", layout="wide")


@st.cache_resource(show_spinner=False)
def get_ocr_engine(language: str) -> OCREngine:
    """Cached so the (lazy) Tesseract backend is only probed once per
    language per session, not on every rerun.
    """
    return OCREngine(language=language)


def _init_session_state() -> None:
    st.session_state.setdefault("uploaded_bytes", None)
    st.session_state.setdefault("uploaded_name", None)


def _reset() -> None:
    st.session_state["uploaded_bytes"] = None
    st.session_state["uploaded_name"] = None


def _load_demo_image(name: str) -> np.ndarray:
    """Generate a small synthetic demo image so the app is fully usable
    without needing to bundle or fetch external sample files.
    """
    import cv2

    canvas = np.full((700, 550, 3), 235, dtype=np.uint8)

    if name == "clean":
        doc = np.full((700, 550, 3), 250, dtype=np.uint8)
        cv2.rectangle(doc, (40, 40), (510, 660), (255, 255, 255), -1)
        cv2.rectangle(doc, (40, 40), (510, 660), (200, 200, 200), 2)
        for i, line_y in enumerate(range(90, 620, 40)):
            length = 420 if i % 5 != 4 else 260
            cv2.line(doc, (70, line_y), (70 + length, line_y), (60, 60, 60), 2)
        return doc

    if name == "rotated":
        doc = np.full((700, 550, 3), 250, dtype=np.uint8)
        cv2.rectangle(doc, (60, 60), (490, 640), (255, 255, 255), -1)
        for i, line_y in enumerate(range(110, 600, 40)):
            length = 380 if i % 5 != 4 else 220
            cv2.line(doc, (90, line_y), (90 + length, line_y), (60, 60, 60), 2)
        matrix = cv2.getRotationMatrix2D((275, 350), 12, 1.0)
        rotated = cv2.warpAffine(doc, matrix, (550, 700), borderValue=(235, 235, 235))
        return rotated

    # "low_quality": clean doc + noise + shadow gradient
    doc = np.full((700, 550, 3), 245, dtype=np.uint8)
    cv2.rectangle(doc, (50, 50), (500, 650), (255, 255, 255), -1)
    for i, line_y in enumerate(range(100, 610, 42)):
        length = 400 if i % 4 != 3 else 240
        cv2.line(doc, (80, line_y), (80 + length, line_y), (70, 70, 70), 2)
    gradient = np.tile(np.linspace(0, 90, 550), (700, 1)).astype(np.uint8)
    gradient = cv2.merge([gradient, gradient, gradient])
    doc = cv2.subtract(doc, gradient)
    noise = np.random.default_rng(42).normal(0, 10, doc.shape).astype(np.int16)
    doc = np.clip(doc.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return doc


def _render_header(app_settings) -> None:
    st.title(app_settings.title)
    st.caption(app_settings.subtitle)
    st.write(
        "Upload a photo of a document and DocVision AI will detect its "
        "boundaries, correct perspective, enhance readability, run OCR, "
        "and pull out common fields like emails, dates, and amounts -- "
        "entirely with classic, CPU-friendly computer vision."
    )


def _render_sidebar_controls(settings):
    with st.sidebar:
        st.header("Processing Controls")
        with st.expander("Detection & Enhancement", expanded=True):
            sensitivity = st.slider(
                "Detection sensitivity", 0, 100, settings.processing.detection_sensitivity
            )
            max_width = st.slider(
                "Max processing width (px)", 400, 2000, settings.processing.max_width, step=100
            )
            strength = st.slider(
                "Enhancement strength", 0, 100, settings.processing.enhancement_strength
            )
            threshold_method = st.selectbox(
                "Thresholding method",
                options=["adaptive_gaussian", "adaptive_mean", "otsu", "none"],
                index=["adaptive_gaussian", "adaptive_mean", "otsu", "none"].index(
                    settings.processing.threshold_method
                ),
            )
        with st.expander("OCR", expanded=True):
            ocr_enabled = st.checkbox("Enable OCR", value=settings.ocr.enabled)

        st.divider()
        st.subheader("Demo Mode")
        demo_choice = st.selectbox(
            "Try a sample image",
            options=["-- none --", "Clean document", "Perspective-distorted", "Low-quality"],
        )
        if st.button("Load demo image", use_container_width=True):
            demo_map = {
                "Clean document": "clean",
                "Perspective-distorted": "rotated",
                "Low-quality": "low_quality",
            }
            key = demo_map.get(demo_choice)
            if key:
                demo_img = _load_demo_image(key)
                st.session_state["uploaded_bytes"] = encode_image_to_bytes(demo_img, ".png")
                st.session_state["uploaded_name"] = f"demo_{key}.png"
            else:
                st.warning("Choose a demo image from the dropdown first.")

        if st.button("Reset", use_container_width=True):
            _reset()

    return {
        "sensitivity": sensitivity,
        "max_width": max_width,
        "strength": strength,
        "threshold_method": threshold_method,
        "ocr_enabled": ocr_enabled,
    }


def _run_pipeline(image_bgr: np.ndarray, controls: dict, settings):
    stats = ProcessingStats(original_resolution=(image_bgr.shape[1], image_bgr.shape[0]))
    timer = Timer()

    with timer:
        resized, _scale = resize_max_width(image_bgr, controls["max_width"])
        stats.processed_resolution = (resized.shape[1], resized.shape[0])

        detection = detect_document(
            resized,
            sensitivity=controls["sensitivity"],
            min_area_ratio=settings.detection.min_area_ratio,
            approx_epsilon_ratio=settings.detection.approx_epsilon_ratio,
        )
        detected_overlay = draw_detection(resized, detection)

        warped = four_point_transform(resized, detection.corners)

        enhanced = enhance_document(
            warped,
            strength=controls["strength"],
            threshold_method=controls["threshold_method"],
        )

        stats.document_area_ratio = document_area_ratio(resized.shape, detection.contour)

        ocr_result = None
        extraction = None
        if controls["ocr_enabled"]:
            engine = get_ocr_engine(settings.ocr.language)
            ocr_result = engine.extract(enhanced)
            if ocr_result.success:
                extraction = extract_entities(ocr_result.text)
                stats.ocr_character_count = ocr_result.character_count
                stats.fields_detected = extraction.total_fields

    stats.processing_time_seconds = timer.elapsed_seconds

    return {
        "detection": detection,
        "detected_overlay": detected_overlay,
        "warped": warped,
        "enhanced": enhanced,
        "ocr_result": ocr_result,
        "extraction": extraction,
        "stats": stats,
    }


def _render_results(original_bgr: np.ndarray, results: dict) -> None:
    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(bgr_to_rgb(original_bgr), use_container_width=True)
        st.markdown("**Perspective Corrected**")
        st.image(bgr_to_rgb(results["warped"]), use_container_width=True)
    with col2:
        st.markdown("**Detected Document**")
        caption = None
        if results["detection"].confidence == "fallback":
            caption = "No confident boundary found -- using the full image as a fallback."
        elif results["detection"].confidence == "low":
            caption = "Boundary detected with lower confidence (approximate rectangle)."
        st.image(bgr_to_rgb(results["detected_overlay"]), use_container_width=True, caption=caption)
        st.markdown("**Enhanced**")
        st.image(results["enhanced"], use_container_width=True, clamp=True, channels="GRAY")

    enhanced_bytes = encode_image_to_bytes(results["enhanced"], ".png")
    st.download_button(
        "Download Processed Image",
        data=enhanced_bytes,
        file_name="docvision_processed.png",
        mime="image/png",
        use_container_width=True,
    )

    st.subheader("OCR Text")
    ocr_result = results["ocr_result"]
    if ocr_result is None:
        st.info("OCR is disabled. Enable it in Processing Controls to extract text.")
    elif not ocr_result.success:
        st.warning(f"OCR is unavailable: {ocr_result.error}")
    elif not ocr_result.text.strip():
        st.info("OCR ran successfully but no readable text was found in this image.")
    else:
        st.text_area("Extracted text", value=ocr_result.text, height=220)
        if ocr_result.mean_confidence is not None:
            st.caption(f"Mean OCR confidence: {ocr_result.mean_confidence:.1f}%")
        st.download_button(
            "Download TXT",
            data=ocr_result.text.encode("utf-8"),
            file_name="docvision_text.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.subheader("Extracted Information")
    st.caption(
        "Automated extraction based on simple pattern matching. Always "
        "verify against the original document before relying on these values."
    )
    extraction = results["extraction"]
    if extraction is None:
        st.info("Run OCR to see extracted fields.")
    else:
        field_labels = [
            ("emails", "Emails"),
            ("phone_numbers", "Phone Numbers"),
            ("dates", "Dates"),
            ("urls", "URLs"),
            ("amounts", "Amounts"),
            ("identifiers", "Identifiers"),
        ]
        data = extraction.as_dict()
        cols = st.columns(3)
        for idx, (key, label) in enumerate(field_labels):
            with cols[idx % 3]:
                st.markdown(f"**{label}**")
                values = data.get(key, [])
                if values:
                    for value in values:
                        st.code(value, language=None)
                else:
                    st.caption("No information of this type detected.")

    st.subheader("Processing Analytics")
    stats = results["stats"]
    display = stats.as_display_dict()
    metric_cols = st.columns(3)
    metric_items = list(display.items())
    for idx, (label, value) in enumerate(metric_items):
        with metric_cols[idx % 3]:
            st.metric(label, value)


def main() -> None:
    settings = load_settings()
    _init_session_state()
    _render_header(settings.app)
    controls = _render_sidebar_controls(settings)

    uploaded_file = st.file_uploader(
        "Upload a document image", type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_file is not None:
        st.session_state["uploaded_bytes"] = uploaded_file.getvalue()
        st.session_state["uploaded_name"] = uploaded_file.name

    raw_bytes = st.session_state.get("uploaded_bytes")
    filename = st.session_state.get("uploaded_name") or ""

    if raw_bytes is None:
        st.info("Upload an image or load a demo image from the sidebar to get started.")
        return

    ext_check = validate_file_extension(filename or "upload.png")
    size_check = validate_file_size(len(raw_bytes))
    if not ext_check.is_valid:
        st.error(ext_check.message)
        return
    if not size_check.is_valid:
        st.error(size_check.message)
        return

    image_bgr = decode_image_bytes(raw_bytes)
    image_check = validate_image_array(image_bgr)
    if not image_check.is_valid:
        st.error(image_check.message)
        return

    try:
        with st.spinner("Processing document..."):
            results = _run_pipeline(image_bgr, controls, settings)
    except Exception as exc:  # last-resort safety net -- never show a traceback
        st.error(
            "Something went wrong while processing this image. Please try "
            "a different photo or adjust the processing controls."
        )
        st.caption(f"Internal detail: {exc.__class__.__name__}")
        return

    _render_results(image_bgr, results)


if __name__ == "__main__":
    main()
