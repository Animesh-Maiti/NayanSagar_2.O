"""Shared application configuration for the NayanSagar side-scan sonar hazard detection platform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Central configuration constants for the web and processing layers."""

    title: str = "NayanSagar AI - Side-Scan Sonar Hazard Detection"
    tile_size: int = 640
    overlap: float = 0.15
    max_upload_size_mb: int = 2048
    weights_path: str = "ai_engine/weights/best.pt"
    sample_image_path: str = "app/static/samples/demo.png"
    sample_xtf_path: str = "data/sample.xtf"


APP_CONFIG = AppConfig()


def get_config() -> AppConfig:
    """Return the application default configuration instance."""
    return APP_CONFIG
