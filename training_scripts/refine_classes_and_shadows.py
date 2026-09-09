"""Recover fine-grained target lineage and add conservative acoustic-shadow polygons."""

from __future__ import annotations

import argparse
import logging
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
CLASS_NAMES = {
    0: "shipwreck",
    1: "aircraft_wreck",
    2: "crab_pot_trap",
    3: "ghost_net",
    4: "debris_highlight",
    5: "acoustic_shadow",
}
IMAGE_SIZE = 640
PRIMARY_CLASSES = set(range(5))


def backup_labels(labels_root: Path, backup_root: Path) -> None:
    """Create and verify a complete label backup before any label is modified."""
    if not labels_root.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {labels_root}")
    if backup_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {backup_root}")
    shutil.copytree(labels_root, backup_root)
    original = sorted(path.relative_to(labels_root) for path in labels_root.rglob("*.txt"))
    copied = sorted(path.relative_to(backup_root) for path in backup_root.rglob("*.txt"))
    if original != copied:
        shutil.rmtree(backup_root, ignore_errors=True)
        raise IOError("Label backup verification failed; no labels were modified")
    LOGGER.info("Verified label backup: %d files at %s", len(copied), backup_root)


def _image_for_label(label_path: Path, images_root: Path) -> Path:
    """Resolve a label's paired image by split and stem."""
    split = label_path.parent.name
    image_path = images_root / split / f"{label_path.stem}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing paired image for {label_path}: {image_path}")
    return image_path


def _parse_polygon(line: str) -> tuple[int, np.ndarray] | None:
    """Parse one YOLO segmentation row into a class and normalized Nx2 polygon."""
    values = line.split()
    if len(values) < 7 or len(values[1:]) % 2:
        return None
    try:
        class_id = int(values[0])
        points = np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
    except ValueError:
        return None
    if len(points) < 3 or np.any(~np.isfinite(points)):
        return None
    return class_id, np.clip(points, 0.0, 1.0)


def _polygon_line(class_id: int, points: np.ndarray) -> str:
    """Serialize a normalized polygon as a YOLO segmentation row."""
    clipped = np.clip(points, 0.0, 1.0)
    return str(class_id) + " " + " ".join(f"{float(value):.6f}" for value in clipped.reshape(-1))


def _source_seabed_class(label_path: Path, source_root: Path) -> str:
    """Read the original Seabed YOLO class name for a unified label stem."""
    source_stem = label_path.stem.split("SeabedObjects-KLSG_zip_", 1)[-1]
    candidates = list(source_root.rglob(f"{source_stem}.txt"))
    if not candidates:
        return "other"
    names = {0: "aircraft", 1: "fish", 2: "other", 3: "shipwreck"}
    first = candidates[0].read_text(encoding="utf-8", errors="replace").split()
    if not first:
        return "other"
    try:
        return names.get(int(first[0]), "other")
    except ValueError:
        return first[0].lower()


def _lineage_class(label_path: Path, polygon: np.ndarray, source_root: Path) -> int:
    """Recover a fine-grained primary class from filename lineage and geometry."""
    name = label_path.stem
    if name.startswith("AI4Shipwrecks_"):
        return 0
    if name.startswith(("SeabedObjects_", "SeabedObjects-KLSG_zip_")):
        source_class = _source_seabed_class(label_path, source_root)
        if "aircraft" in source_class or "plane" in source_class:
            return 1
        if "wreck" in source_class or "ship" in source_class:
            return 0
        return 4
    if name.startswith(("GhostVision_", "GhostVision_DatasetAndModels_")):
        pixels = polygon * IMAGE_SIZE
        x, y, width, height = cv2.boundingRect(np.round(pixels).astype(np.int32))
        ratio = max(width / max(height, 1), height / max(width, 1))
        contour = np.round(pixels).astype(np.int32).reshape(-1, 1, 2)
        area = max(cv2.contourArea(contour), 1.0)
        hull_area = max(cv2.contourArea(cv2.convexHull(contour)), area)
        perimeter_ratio = (cv2.arcLength(contour, True) ** 2) / area
        if ratio >= 2.0 or hull_area / area >= 1.35 or perimeter_ratio >= 35.0:
            return 3
        return 2
    return 4


def _shadow_polygon(image: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Find one conservative dark contiguous region on the target's outer side."""
    pixels = np.round(target * IMAGE_SIZE).astype(np.int32)
    x, y, width, height = cv2.boundingRect(pixels.reshape(-1, 1, 2))
    center_x = x + width / 2.0
    direction = -1 if center_x < IMAGE_SIZE / 2 else 1
    pad_y = max(4, int(height * 0.25))
    y0, y1 = max(0, y - pad_y), min(IMAGE_SIZE, y + height + pad_y)
    search_width = max(24, min(3 * width, 220))
    if direction < 0:
        sx0, sx1 = max(0, x - search_width), x
    else:
        sx0, sx1 = x + width, min(IMAGE_SIZE, x + width + search_width)
    if sx1 <= sx0 or y1 <= y0:
        return None
    window = image[y0:y1, sx0:sx1]
    if window.size == 0:
        return None
    background_strip = image[y0:y1, max(0, sx0 - 16):sx0] if direction < 0 else image[y0:y1, sx1:min(IMAGE_SIZE, sx1 + 16)]
    if background_strip.size == 0:
        background_strip = image[y0:y1, :]
    local_background = float(np.median(background_strip))
    if local_background <= 1:
        return None
    threshold, _ = cv2.threshold(window, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (window <= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_area = max(float(cv2.contourArea(pixels.reshape(-1, 1, 2))), 1.0)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not 0.25 * target_area <= area <= 3.0 * target_area:
            continue
        contour[:, 0, 0] += sx0
        contour[:, 0, 1] += y0
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, -1)
        mean_intensity = float(image[mask.astype(bool)].mean()) if mask.any() else 255.0
        if mean_intensity >= 0.35 * local_background:
            continue
        edge_distance = abs(float(cv2.boundingRect(contour)[0] - (x if direction < 0 else x + width)))
        candidates.append((edge_distance, contour))
    if not candidates:
        return None
    contour = min(candidates, key=lambda item: item[0])[1]
    points = contour.reshape(-1, 2).astype(np.float32) / IMAGE_SIZE
    return points if len(points) >= 3 else None


def refine_dataset(dataset_root: Path, source_root: Path, backup_root: Path) -> Counter[int]:
    """Back up labels, refine all labels, and append validated acoustic shadows."""
    backup_labels(dataset_root / "labels", backup_root)
    counts: Counter[int] = Counter()
    for split in ("train", "val"):
        label_dir = dataset_root / "labels" / split
        image_dir = dataset_root / "images" / split
        for label_path in sorted(label_dir.glob("*.txt")):
            image = cv2.imread(str(_image_for_label(label_path, dataset_root / "images")), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"Expected 640x640 grayscale image for {label_path}")
            parsed = [_parse_polygon(line) for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines()]
            polygons = [item for item in parsed if item is not None]
            refined: list[tuple[int, np.ndarray]] = []
            for old_class, polygon in polygons:
                new_class = old_class if old_class == 5 else _lineage_class(label_path, polygon, source_root)
                refined.append((new_class, polygon))
            shadows: list[np.ndarray] = []
            for class_id, polygon in refined:
                if class_id in PRIMARY_CLASSES:
                    shadow = _shadow_polygon(image, polygon)
                    if shadow is not None:
                        shadows.append(shadow)
            refined.extend((5, shadow) for shadow in shadows)
            label_path.write_text("\n".join(_polygon_line(class_id, polygon) for class_id, polygon in refined) + ("\n" if refined else ""), encoding="utf-8")
            for class_id, _ in refined:
                counts[class_id] += 1
    return counts


def _write_yaml(path: Path) -> None:
    """Write the six-class taxonomy configuration."""
    lines = ["path: ../data/unified_dataset", "train: images/train", "val: images/val", "names:"]
    lines.extend(f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run safe lineage refinement and acoustic-shadow synthesis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/unified_dataset"))
    parser.add_argument("--source-root", type=Path, default=Path("data/benchmark_images/unzipped/SeabedObjects-KLSG_zip"))
    parser.add_argument("--backup-root", type=Path, default=Path("data/unified_dataset/labels_backup"))
    parser.add_argument("--yaml", type=Path, default=Path("sonar_data.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    counts = refine_dataset(args.dataset_root, args.source_root, args.backup_root)
    _write_yaml(args.yaml)
    print("Refined annotation totals")
    for class_id, name in CLASS_NAMES.items():
        print(f"{class_id}: {name:<20} {counts[class_id]:>8}")
    print(f"Label backup: {args.backup_root}")
    print(f"Configuration: {args.yaml}")


if __name__ == "__main__":
    main()
