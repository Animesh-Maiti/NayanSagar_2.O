"""Extract and classify fixed-size sonar tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

TILE_SIZE = 640
STRIDE = 512


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def _windows(length: int, size: int, stride: int) -> list[int]:
    """Return window starts, including the final edge-aligned window."""
    if length < size:
        return []
    starts = list(range(0, length - size + 1, stride))
    final = length - size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _is_target(tile: np.ndarray, variance_threshold: float, contrast_threshold: float) -> bool:
    """Classify highlights/shadows using robust range and local variance cues."""
    variance = float(np.var(tile))
    contrast = float(np.percentile(tile, 95) - np.percentile(tile, 5))
    local_variance = cv2.Laplacian(tile, cv2.CV_32F).var()
    return variance >= variance_threshold or contrast >= contrast_threshold or local_variance >= variance_threshold


def generate_tiles(
    source: np.ndarray | str | Path,
    output_root: str | Path = Path("data/processed_tiles"),
    variance_threshold: float | None = None,
    contrast_threshold: float = 40.0,
    telemetry: dict[str, np.ndarray] | None = None,
    altitude: float | None = None,
) -> list[dict[str, Any]]:
    """Tile an array or image directory and write classified PNGs plus metadata."""
    output_root = Path(output_root)
    background_dir = output_root / "background"
    targets_dir = output_root / "targets"
    background_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        inputs = sorted(path for path in Path(source).iterdir() if path.suffix.lower() in {".png", ".tif", ".tiff"})
        matrices = [(path.name, _read_image(path)) for path in inputs]
    elif isinstance(source, (str, Path)):
        path = Path(source)
        matrices = [(path.name, _read_image(path))]
    else:
        matrices = [("array", np.asarray(source))]
    metadata: list[dict[str, Any]] = []
    for source_name, matrix in matrices:
        if matrix.ndim != 2:
            raise ValueError(f"Source {source_name} must be single-channel, got shape {matrix.shape}")
        matrix = np.nan_to_num(matrix.astype(np.float32), nan=0.0)
        threshold = variance_threshold if variance_threshold is not None else max(float(np.var(matrix)) * 1.5, 1.0)
        for top in _windows(matrix.shape[0], TILE_SIZE, STRIDE):
            for left in _windows(matrix.shape[1], TILE_SIZE, STRIDE):
                tile = matrix[top : top + TILE_SIZE, left : left + TILE_SIZE]
                target = _is_target(tile, threshold, contrast_threshold)
                category = "targets" if target else "background"
                stem = Path(source_name).stem
                filename = f"{stem}_y{top}_x{left}.png"
                destination = output_root / category / filename
                normalized = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if not cv2.imwrite(str(destination), normalized):
                    raise OSError(f"Unable to write tile: {destination}")
                record: dict[str, Any] = {
                    "tile_filename": str(Path(category) / filename),
                    "source": source_name,
                    "source_ping_index": top,
                    "source_ping_end": top + TILE_SIZE - 1,
                    "bbox": {"x_min": left, "y_min": top, "x_max": left + TILE_SIZE, "y_max": top + TILE_SIZE},
                    "category": category,
                }
                if telemetry is not None:
                    latitudes = np.asarray(telemetry.get("SensorYcoord", []), dtype=np.float64)[top : top + TILE_SIZE]
                    longitudes = np.asarray(telemetry.get("SensorXcoord", []), dtype=np.float64)[top : top + TILE_SIZE]
                    record["lat_lon_bounds"] = {
                        "lat_min": float(np.nanmin(latitudes)) if np.isfinite(latitudes).any() else None,
                        "lat_max": float(np.nanmax(latitudes)) if np.isfinite(latitudes).any() else None,
                        "lon_min": float(np.nanmin(longitudes)) if np.isfinite(longitudes).any() else None,
                        "lon_max": float(np.nanmax(longitudes)) if np.isfinite(longitudes).any() else None,
                    }
                record["altitude"] = float(altitude) if altitude is not None else None
                metadata.append(record)
    metadata_path = output_root / "tiles_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    """Run tile extraction from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Preprocessed image or image directory")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed_tiles"))
    parser.add_argument("--variance-threshold", type=float, default=None)
    parser.add_argument("--contrast-threshold", type=float, default=40.0)
    args = parser.parse_args()
    records = generate_tiles(args.input, args.output_root, args.variance_threshold, args.contrast_threshold)
    print(f"Generated {len(records)} tiles; metadata: {args.output_root / 'tiles_metadata.json'}")


if __name__ == "__main__":
    main()
