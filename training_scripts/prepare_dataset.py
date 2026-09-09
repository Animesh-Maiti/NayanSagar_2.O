"""Extract benchmark archives and build a unified 640x640 YOLO-seg dataset."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
ANNOTATION_EXTENSIONS = {".json", ".xml", ".txt"}
_JSONL_CACHE: dict[Path, dict[str, dict[str, Any]]] = {}
CLASS_NAMES = {
    0: "ghost_net",
    1: "structural_hazard",
    2: "debris_highlight",
    3: "acoustic_shadow",
}

# Dataset-specific source labels mapped to the project taxonomy.
LABEL_MAP = {
    "crab-pot": 1,
    "crab_pot": 1,
    "shipwreck": 1,
    "aircraft": 1,
    "fish": 2,
    "other": 2,
    "debris": 2,
    "ghost-net": 0,
    "ghost_net": 0,
    "shadow": 3,
    "acoustic-shadow": 3,
}


@dataclass(frozen=True)
class Annotation:
    """One normalized polygon annotation."""

    class_id: int
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Sample:
    """An image and its source annotation, both within staging."""

    image_path: Path
    annotation_path: Path | None
    dataset: str


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract an archive while preventing path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zfile:
        root = destination.resolve()
        for member in zfile.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        zfile.extractall(destination)


def extract_archives(benchmark_root: Path, staging_root: Path, clean: bool = False) -> list[Path]:
    """Extract all benchmark ZIPs into separate staging folders."""
    if clean and staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    for archive in sorted(benchmark_root.glob("*.zip")):
        destination = staging_root / archive.stem
        marker = destination / ".extracted"
        if not marker.exists():
            LOGGER.info("Extracting %s", archive.name)
            _safe_extract(archive, destination)
            marker.write_text("extracted\n", encoding="utf-8")
        else:
            LOGGER.info("Using existing staging: %s", destination)
        for nested_archive in destination.glob("*.zip"):
            nested_destination = destination / nested_archive.stem
            nested_marker = nested_destination / ".extracted"
            if not nested_marker.exists():
                LOGGER.info("Extracting nested archive %s", nested_archive.name)
                _safe_extract(nested_archive, nested_destination)
                nested_marker.write_text("extracted\n", encoding="utf-8")
        extracted.append(destination)
    return extracted


def _find_annotation(image_path: Path) -> Path | None:
    """Find the annotation sharing an image stem, including YOLO split layouts."""
    for suffix in (".txt", ".json", ".xml"):
        candidate = image_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
        candidate = image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}{suffix}"
        if candidate.is_file():
            return candidate
        candidate = image_path.parent.parent / "labels" / f"{image_path.stem}{suffix}"
        if candidate.is_file():
            return candidate
    for parent in (image_path.parent, *image_path.parents):
        metadata_path = parent / "metadata.jsonl"
        if metadata_path.is_file():
            return metadata_path
    if "AI4Shipwrecks" in image_path.parts and image_path.parent.name == "images":
        candidate = image_path.parent.parent / "labels" / image_path.name
        if candidate.is_file():
            return candidate
    return None


def inspect_dataset(root: Path) -> tuple[list[Sample], Counter[str]]:
    """Discover images, paired labels, and annotation formats under a staging root."""
    samples: list[Sample] = []
    formats: Counter[str] = Counter()
    for image_path in sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
        if "AI4Shipwrecks" in image_path.parts and "extras" in image_path.parts:
            continue
        annotation_path = _find_annotation(image_path)
        if annotation_path is None:
            continue
        suffix = annotation_path.suffix.lower()
        format_name = {".json": "COCO JSON", ".xml": "Pascal VOC XML", ".txt": "YOLO TXT", ".jsonl": "JSONL bounding boxes"}.get(suffix, "Raster mask")
        formats[format_name] += 1
        samples.append(Sample(image_path, annotation_path, root.name))
    LOGGER.info("%s: %d labeled images; formats=%s", root.name, len(samples), dict(formats))
    return samples, formats


def _class_id(label: Any, dataset: str) -> int | None:
    """Map a source class name or integer to the unified class taxonomy."""
    if isinstance(label, (int, float)) or (isinstance(label, str) and label.strip().isdigit()):
        source_id = int(label)
        if dataset.startswith("SeabedObjects"):
            return {0: 1, 1: 2, 2: 2, 3: 1}.get(source_id)
        if dataset.startswith("GhostVision"):
            return 1
        return None
    normalized = str(label).strip().lower().replace(" ", "-")
    return LABEL_MAP.get(normalized)


def _clip_polygon(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...] | None:
    """Clip normalized polygon points and reject degenerate shapes."""
    clipped = tuple((min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))) for x, y in points)
    return clipped if len(clipped) >= 3 and len(set(clipped)) >= 3 else None


def _yolo_annotations(path: Path, width: int, height: int, dataset: str) -> list[Annotation]:
    annotations: list[Annotation] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        class_id = _class_id(values[0], dataset)
        if class_id is None:
            continue
        numbers = [float(value) for value in values[1:]]
        if len(numbers) >= 6 and len(numbers) % 2 == 0:
            points = [(numbers[index], numbers[index + 1]) for index in range(0, len(numbers), 2)]
        elif len(numbers) == 4:
            center_x, center_y, box_width, box_height = numbers
            points = [(center_x - box_width / 2, center_y - box_height / 2), (center_x + box_width / 2, center_y - box_height / 2), (center_x + box_width / 2, center_y + box_height / 2), (center_x - box_width / 2, center_y + box_height / 2)]
        else:
            continue
        polygon = _clip_polygon(points)
        if polygon:
            annotations.append(Annotation(class_id, polygon))
    return annotations


def _voc_annotations(path: Path, width: int, height: int, dataset: str) -> list[Annotation]:
    root = ET.parse(path).getroot()
    annotations: list[Annotation] = []
    for obj in root.findall(".//object"):
        class_id = _class_id(obj.findtext("name", ""), dataset)
        box = obj.find("bndbox")
        if class_id is None or box is None:
            continue
        xmin, ymin = float(box.findtext("xmin", "0")) / width, float(box.findtext("ymin", "0")) / height
        xmax, ymax = float(box.findtext("xmax", "0")) / width, float(box.findtext("ymax", "0")) / height
        polygon = _clip_polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
        if polygon:
            annotations.append(Annotation(class_id, polygon))
    return annotations


def _coco_annotations(path: Path, image_path: Path, width: int, height: int, dataset: str) -> list[Annotation]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in payload.get("images", [])}
    image_record = next((item for item in images.values() if Path(item.get("file_name", "")).name == image_path.name), None)
    if image_record is None:
        return []
    categories = {item["id"]: item.get("name", item["id"]) for item in payload.get("categories", [])}
    annotations: list[Annotation] = []
    for item in payload.get("annotations", []):
        if item.get("image_id") != image_record["id"]:
            continue
        class_id = _class_id(categories.get(item.get("category_id"), ""), dataset)
        segmentation = item.get("segmentation", [])
        if class_id is None:
            continue
        if segmentation and isinstance(segmentation[0], list):
            segmentation = segmentation[0]
        if len(segmentation) >= 6:
            points = [(segmentation[index] / width, segmentation[index + 1] / height) for index in range(0, len(segmentation), 2)]
        else:
            x, y, box_width, box_height = item.get("bbox", [0, 0, 0, 0])
            points = [(x / width, y / height), ((x + box_width) / width, y / height), ((x + box_width) / width, (y + box_height) / height), (x / width, (y + box_height) / height)]
        polygon = _clip_polygon(points)
        if polygon:
            annotations.append(Annotation(class_id, polygon))
    return annotations


def _jsonl_annotations(path: Path, image_path: Path, width: int, height: int, dataset: str) -> list[Annotation]:
    """Convert object bounding boxes stored in one-record-per-line metadata."""
    records = _JSONL_CACHE.get(path)
    if records is None:
        records = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("file_name"):
                records[Path(record["file_name"]).name] = record
        _JSONL_CACHE[path] = records
    record = records.get(image_path.name, {})
    objects = record.get("objects", {})
    annotations: list[Annotation] = []
    for box, label in zip(objects.get("bbox", []), objects.get("category", [])):
        class_id = _class_id(label, dataset)
        if class_id is None or len(box) != 4:
            continue
        x, y, box_width, box_height = (float(value) for value in box)
        polygon = _clip_polygon([(x / width, y / height), ((x + box_width) / width, y / height), ((x + box_width) / width, (y + box_height) / height), (x / width, (y + box_height) / height)])
        if polygon:
            annotations.append(Annotation(class_id, polygon))
    return annotations


def convert_annotations(sample: Sample, image: np.ndarray) -> list[Annotation]:
    """Read one annotation file and return unified normalized polygons."""
    height, width = image.shape[:2]
    if sample.annotation_path is None:
        return []
    suffix = sample.annotation_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        mask = cv2.imread(str(sample.annotation_path), cv2.IMREAD_UNCHANGED)
        return _mask_annotations(mask) if mask is not None else []
    if suffix == ".txt":
        return _yolo_annotations(sample.annotation_path, width, height, sample.dataset)
    if suffix == ".xml":
        return _voc_annotations(sample.annotation_path, width, height, sample.dataset)
    if suffix == ".jsonl":
        return _jsonl_annotations(sample.annotation_path, sample.image_path, width, height, sample.dataset)
    return _coco_annotations(sample.annotation_path, sample.image_path, width, height, sample.dataset)


def _standardize_image(image: np.ndarray) -> np.ndarray:
    """Convert any source image to a 640x640 uint8 grayscale image."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    values = image.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size:
        low, high = np.percentile(finite, (1, 99))
        values = np.zeros_like(values) if high <= low else np.clip((values - low) * 255.0 / (high - low), 0, 255)
    return values.astype(np.uint8)


def _mask_annotations(mask: np.ndarray) -> list[Annotation]:
    """Convert nonzero connected mask regions into normalized class-1 polygons."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary.shape[:2]
    annotations: list[Annotation] = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue
        points = [(float(point[0][0]) / width, float(point[0][1]) / height) for point in contour]
        polygon = _clip_polygon(points)
        if polygon:
            annotations.append(Annotation(1, polygon))
    return annotations


def _write_yaml(path: Path) -> None:
    """Write the requested YOLO dataset configuration."""
    lines = ["path: ../data/unified_dataset", "train: images/train", "val: images/val", "names:"]
    lines.extend(f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _existing_summary(output_root: Path) -> dict[str, Counter[int]]:
    """Count annotations already present in an append-only unified dataset."""
    summary = {"train": Counter(), "val": Counter()}
    for split in summary:
        for label_path in (output_root / "labels" / split).glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                values = line.split()
                if values and values[0].isdigit() and int(values[0]) in CLASS_NAMES:
                    summary[split][int(values[0])] += 1
    return summary


def prepare_dataset(
    benchmark_root: str | Path = Path("data/benchmark_images"),
    output_root: str | Path = Path("data/unified_dataset"),
    staging_root: str | Path = Path("data/benchmark_images/unzipped"),
    seed: int = 42,
    clean: bool = False,
) -> dict[str, Counter[int]]:
    """Extract, convert, split, and configure the unified dataset."""
    benchmark_root, output_root, staging_root = Path(benchmark_root), Path(output_root), Path(staging_root)
    staged_roots = extract_archives(benchmark_root, staging_root, clean=clean)
    samples: list[Sample] = []
    for root in staged_roots:
        discovered, _ = inspect_dataset(root)
        samples.extend(discovered)
    append_only = output_root.exists() and not clean
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    if append_only:
        samples = [sample for sample in samples if sample.dataset == "AI4Shipwrecks"]
    random.Random(seed).shuffle(samples)
    split_index = int(len(samples) * 0.8)
    summary = _existing_summary(output_root) if append_only else {"train": Counter(), "val": Counter()}
    for index, sample in enumerate(samples):
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            LOGGER.warning("Skipping unreadable image: %s", sample.image_path)
            continue
        annotations = convert_annotations(sample, image)
        standardized = _standardize_image(image)
        split = "train" if index < split_index else "val"
        output_stem = f"{sample.dataset}_{sample.image_path.stem}".replace(" ", "_")
        image_destination = output_root / "images" / split / f"{output_stem}.png"
        label_destination = output_root / "labels" / split / f"{output_stem}.txt"
        if image_destination.exists() and append_only:
            LOGGER.warning("Preserving existing output: %s", image_destination)
            continue
        cv2.imwrite(str(image_destination), standardized)
        label_destination.write_text("\n".join(f"{annotation.class_id} " + " ".join(f"{value:.6f}" for point in annotation.points for value in point) for annotation in annotations) + ("\n" if annotations else ""), encoding="utf-8")
        for annotation in annotations:
            summary[split][annotation.class_id] += 1
    _write_yaml(Path("sonar_data.yaml"))
    LOGGER.info("Converted %d labeled samples", len(samples))
    return summary


def main() -> None:
    """Run dataset preparation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=Path("data/benchmark_images"))
    parser.add_argument("--output-root", type=Path, default=Path("data/unified_dataset"))
    parser.add_argument("--staging-root", type=Path, default=Path("data/benchmark_images/unzipped"))
    parser.add_argument("--clean", action="store_true", help="Remove existing staging and unified output first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = prepare_dataset(args.benchmark_root, args.output_root, args.staging_root, clean=args.clean)
    print("\nClass summary (converted annotations)")
    print(f"{'Class':<22} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-" * 50)
    for class_id, name in CLASS_NAMES.items():
        train_count, val_count = summary["train"][class_id], summary["val"][class_id]
        print(f"{class_id}: {name:<18} {train_count:>8} {val_count:>8} {train_count + val_count:>8}")
    print(f"\nImages and labels: {args.output_root}")
    print("Configuration: sonar_data.yaml")


if __name__ == "__main__":
    main()
