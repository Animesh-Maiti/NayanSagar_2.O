"""Run Phase 1 XTF ingestion, memory-bounded preprocessing, and tile extraction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

try:
    from .parse_xtf import parse_xtf
    from .preprocess import preprocess_sonar
    from .tile_generator import generate_tiles
except ImportError:
    from parse_xtf import parse_xtf
    from preprocess import preprocess_sonar
    from tile_generator import generate_tiles

LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT = Path("data/raw_xtf/DATA0000117.H-PU.xtf")
DEFAULT_OUTPUT = Path("data/processed_tiles")


def _safe_median_altitude(telemetry: dict[str, np.ndarray]) -> float:
    """Return a positive finite altitude median, or a conservative pixel fallback."""
    altitude = np.asarray(telemetry.get("SensorAltitude", []), dtype=np.float64)
    valid = altitude[np.isfinite(altitude) & (altitude > 0)]
    if valid.size:
        return float(np.median(valid))
    LOGGER.warning("SensorAltitude has no positive finite samples; using altitude_px=1.0")
    return 1.0


def _downsample_cross_track(waterfall: np.ndarray, altitude_px: float, max_samples: int) -> tuple[np.ndarray, float, int]:
    """Bound cross-track memory while preserving the altitude-to-sample scale."""
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    factor = max(1, int(np.ceil(waterfall.shape[1] / max_samples)))
    if factor == 1:
        return waterfall, altitude_px, factor
    LOGGER.warning("Downsampling cross-track samples by %sx for memory safety", factor)
    return waterfall[:, ::factor], altitude_px / factor, factor


def run_pipeline(
    input_path: str | Path = DEFAULT_INPUT,
    output_root: str | Path = DEFAULT_OUTPUT,
    chunk_rows: int = 256,
    max_samples: int = 8192,
) -> tuple[list[dict[str, object]], float]:
    """Execute Phase 1 and return tile metadata plus the selected altitude."""
    waterfall, telemetry = parse_xtf(input_path)
    altitude_px = _safe_median_altitude(telemetry)
    waterfall, altitude_px, _ = _downsample_cross_track(waterfall, altitude_px, max_samples)
    LOGGER.info("Raw waterfall shape=%s; altitude_px=%.3f", waterfall.shape, altitude_px)

    ground = preprocess_sonar(waterfall, altitude_px, chunk_rows=chunk_rows)
    LOGGER.info("Ground-range matrix shape=%s", ground.shape)

    output_root = Path(output_root)
    for category in ("background", "targets"):
        category_dir = output_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for tile_path in category_dir.glob("*.png"):
            tile_path.unlink()
    records = generate_tiles(ground, output_root, telemetry=telemetry, altitude=altitude_px)
    return records, altitude_px


def main() -> None:
    """Run the end-to-end pipeline from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=8192)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    records, altitude = run_pipeline(args.input, args.output_root, args.chunk_rows, args.max_samples)
    background = sum(record["category"] == "background" for record in records)
    targets = sum(record["category"] == "targets" for record in records)
    print(f"Background tiles generated: {background}")
    print(f"Target candidate tiles generated: {targets}")
    print(f"Altitude used: {altitude:.3f} pixels")
    print(f"Metadata: {Path(args.output_root) / 'tiles_metadata.json'}")


if __name__ == "__main__":
    main()