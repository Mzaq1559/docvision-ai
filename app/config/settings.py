"""
Configuration loading for DocVision AI.

Reads configs/default.yaml (falling back to hard-coded defaults if the file
cannot be found, e.g. in certain packaging/deployment scenarios) and exposes
a simple, typed-ish ``Settings`` object plus a Streamlit-friendly loader.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict

import yaml

_DEFAULTS: Dict[str, Any] = {
    "processing": {
        "max_width": 1000,
        "detection_sensitivity": 50,
        "enhancement_strength": 50,
        "threshold_method": "adaptive_gaussian",
    },
    "ocr": {
        "enabled": True,
        "language": "eng",
        "min_confidence": 0,
    },
    "detection": {
        "min_area_ratio": 0.15,
        "approx_epsilon_ratio": 0.02,
    },
    "app": {
        "title": "DocVision AI",
        "subtitle": "Computer Vision Document Scanner & Intelligence",
    },
}

_CONFIG_PATH_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "default.yaml"),
    os.path.join(os.getcwd(), "configs", "default.yaml"),
]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_raw_config() -> Dict[str, Any]:
    for path in _CONFIG_PATH_CANDIDATES:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh) or {}
                return _deep_merge(_DEFAULTS, loaded)
            except (OSError, yaml.YAMLError):
                # Fall through to defaults; never crash the app over config I/O.
                break
    return dict(_DEFAULTS)


@dataclass(frozen=True)
class ProcessingSettings:
    max_width: int = 1000
    detection_sensitivity: int = 50
    enhancement_strength: int = 50
    threshold_method: str = "adaptive_gaussian"


@dataclass(frozen=True)
class OcrSettings:
    enabled: bool = True
    language: str = "eng"
    min_confidence: int = 0


@dataclass(frozen=True)
class DetectionSettings:
    min_area_ratio: float = 0.15
    approx_epsilon_ratio: float = 0.02


@dataclass(frozen=True)
class AppSettings:
    title: str = "DocVision AI"
    subtitle: str = "Computer Vision Document Scanner & Intelligence"


@dataclass(frozen=True)
class Settings:
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    ocr: OcrSettings = field(default_factory=OcrSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    app: AppSettings = field(default_factory=AppSettings)

    def with_overrides(self, **kwargs: Any) -> "Settings":
        """Return a copy of these settings with top-level section overrides.

        Example: settings.with_overrides(processing=replace(settings.processing, max_width=800))
        """
        return replace(self, **kwargs)


def load_settings() -> Settings:
    """Load settings from configs/default.yaml, merged over built-in defaults."""
    raw = _load_raw_config()
    return Settings(
        processing=ProcessingSettings(**raw.get("processing", {})),
        ocr=OcrSettings(**raw.get("ocr", {})),
        detection=DetectionSettings(**raw.get("detection", {})),
        app=AppSettings(**raw.get("app", {})),
    )
