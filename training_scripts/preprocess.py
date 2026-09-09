"""Physics-aware preprocessing for side-scan sonar waterfalls."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def lee_enhanced_filter(image: np.ndarray, win_size: int = 5, k: float = 1.0, cu: float = 0.523) -> np.ndarray:
    """Apply a Lee-enhanced local-statistics filter to Rayleigh-speckled sonar data."""
    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional array")
    if win_size < 3 or win_size % 2 == 0 or k <= 0 or cu <= 0:
        raise ValueError("win_size must be odd and >= 3; k and cu must be positive")
    source = np.asarray(image, dtype=np.float32)
    local_mean = cv2.blur(source, (win_size, win_size), borderType=cv2.BORDER_REFLECT)
    local_sq_mean = cv2.blur(source * source, (win_size, win_size), borderType=cv2.BORDER_REFLECT)
    local_variance = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
    noise_variance = (local_mean * cu) ** 2
    coefficient = np.sqrt(local_variance) / (local_mean + np.finfo(np.float32).eps)
    weight = np.clip((coefficient - cu) / (k * coefficient + np.finfo(np.float32).eps), 0.0, 1.0)
    weight[local_variance <= noise_variance] = 0.0
    return local_mean + weight * (source - local_mean)


def _project_side(side: np.ndarray, altitude_px: float) -> np.ndarray:
    """Project one side's slant samples onto a regular ground-range coordinate."""
    if side.shape[1] == 0:
        return side.copy()
    slant = np.arange(side.shape[1], dtype=np.float32)
    valid = slant >= altitude_px
    if not np.any(valid):
        return np.empty((side.shape[0], 0), dtype=np.float32)
    ground = np.sqrt(np.maximum(slant[valid] ** 2 - altitude_px ** 2, 0.0))
    grid = np.arange(np.ceil(ground[-1]) + 1, dtype=np.float32)
    projected = np.empty((side.shape[0], grid.size), dtype=np.float32)
    for row_index, row in enumerate(side[:, valid]):
        projected[row_index] = np.interp(grid, ground, row).astype(np.float32)
    return projected


def slant_to_ground_range(waterfall_array: np.ndarray, altitude_px: float) -> np.ndarray:
    """Remove the nadir water-column region and project port/starboard to ground range."""
    if waterfall_array.ndim != 2:
        raise ValueError("waterfall_array must be two-dimensional")
    if not np.isfinite(altitude_px) or altitude_px < 0:
        raise ValueError("altitude_px must be a finite non-negative value")
    midpoint = waterfall_array.shape[1] // 2
    port = np.asarray(waterfall_array[:, :midpoint], dtype=np.float32)
    starboard = np.asarray(waterfall_array[:, midpoint:], dtype=np.float32)
    projected_port = _project_side(port, float(altitude_px))
    projected_starboard = _project_side(starboard, float(altitude_px))
    return np.concatenate((np.fliplr(projected_port), projected_starboard), axis=1)


def preprocess_sonar(
    waterfall_array: np.ndarray,
    altitude_px: float,
    chunk_rows: int | None = None,
) -> np.ndarray:
    """Despeckle a waterfall and apply slant correction, optionally in row chunks."""
    source = np.asarray(waterfall_array)
    if chunk_rows is None:
        return slant_to_ground_range(lee_enhanced_filter(source), altitude_px)
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")

    # Keep a filter-radius halo so chunk boundaries do not create visible seams.
    halo = 2
    chunks: list[np.ndarray] = []
    for start in range(0, source.shape[0], chunk_rows):
        stop = min(start + chunk_rows, source.shape[0])
        padded_start = max(0, start - halo)
        padded_stop = min(source.shape[0], stop + halo)
        filtered = lee_enhanced_filter(source[padded_start:padded_stop])
        core_start = start - padded_start
        core_stop = core_start + (stop - start)
        chunks.append(slant_to_ground_range(filtered[core_start:core_stop], altitude_px))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)


def main() -> None:
    """Run preprocessing on a NumPy waterfall file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input .npy waterfall")
    parser.add_argument("--altitude-px", required=True, type=float, help="Sensor altitude in sample pixels")
    parser.add_argument("--output", required=True, type=Path, help="Output .npy file")
    args = parser.parse_args()
    result = preprocess_sonar(np.load(args.input), args.altitude_px)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result)


if __name__ == "__main__":
    main()
