"""Rebuild a balanced six-class sonar segmentation dataset from five sources."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import stat
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CLASS_NAMES = {0: "shipwreck", 1: "aircraft_wreck", 2: "crab_pot_trap", 3: "ghost_net", 4: "debris_highlight", 5: "acoustic_shadow"}


@dataclass
class Record:
    image: Path
    annotations: list[tuple[int, np.ndarray]]
    dataset: str
    group: str


def safe_extract_nested(root: Path) -> None:
    """Extract nested ZIP payloads safely into sibling directories."""
    for archive in list(root.rglob("*.zip")):
        destination = archive.with_suffix("")
        marker = destination / ".extracted"
        if marker.exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            base = destination.resolve()
            for member in source.infolist():
                target = (destination / member.filename).resolve()
                if target != base and base not in target.parents:
                    raise ValueError(f"Unsafe nested archive member: {member.filename}")
            source.extractall(destination)
        marker.write_text("ok\n", encoding="utf-8")


def parse_yolo(path: Path, source_names: list[str], dataset: str) -> list[tuple[int, np.ndarray]]:
    """Read YOLO boxes or polygons and remap source class names."""
    mapping = {"shipwreck": 0, "airplane": 1, "cylinder": 4, "manta": 4, "victim": 4}
    debris = {"net": 3, "rope": 3, "chain": 3, "can": 4, "bottle": 4, "tire": 4, "propeller": 4, "mine": 4, "drink-carton": 4, "hook": 4, "valve": 4, "shampoo-bottle": 4, "standing-bottle": 4}
    result: list[tuple[int, np.ndarray]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        try:
            source_id = int(values[0])
            numbers = [float(value) for value in values[1:]]
        except ValueError:
            continue
        source_name = source_names[source_id].lower() if 0 <= source_id < len(source_names) else str(source_id)
        if dataset == "GhostVision":
            class_id = 2
        elif dataset == "SeabedObjects_KLSG":
            class_id = {0: 1, 1: 2, 2: 4, 3: 0}.get(source_id)
        else:
            class_id = mapping.get(source_name, debris.get(source_name))
        if class_id is None:
            continue
        if len(numbers) == 4:
            cx, cy, width, height = numbers
            points = np.asarray([[cx - width / 2, cy - height / 2], [cx + width / 2, cy - height / 2], [cx + width / 2, cy + height / 2], [cx - width / 2, cy + height / 2]], dtype=np.float32)
        elif len(numbers) >= 6 and len(numbers) % 2 == 0:
            points = np.asarray(numbers, dtype=np.float32).reshape(-1, 2)
        else:
            continue
        result.append((class_id, np.clip(points, 0.0, 1.0)))
    return result


def mask_annotations(path: Path) -> list[tuple[int, np.ndarray]]:
    """Convert nonzero AI4Shipwrecks mask regions to class-0 polygons."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return []
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary.shape
    result = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue
        points = np.asarray([(point[0][0] / width, point[0][1] / height) for point in contour], dtype=np.float32)
        if len(points) >= 3:
            result.append((0, np.clip(points, 0.0, 1.0)))
    return result


def jsonl_index(root: Path) -> dict[str, dict[str, Any]]:
    """Index GhostVision JSONL metadata by image basename."""
    records: dict[str, dict[str, Any]] = {}
    for path in root.rglob("metadata.jsonl"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("file_name"):
                records[Path(record["file_name"]).name] = record
    return records


def jsonl_annotations(record: dict[str, Any], width: int, height: int) -> list[tuple[int, np.ndarray]]:
    """Convert GhostVision Crab-Pot metadata boxes to polygons."""
    objects = record.get("objects", {})
    result = []
    for box in objects.get("bbox", []):
        if len(box) != 4:
            continue
        x, y, w, h = (float(value) for value in box)
        points = np.asarray([(x / width, y / height), ((x + w) / width, y / height), ((x + w) / width, (y + h) / height), (x / width, (y + h) / height)], dtype=np.float32)
        result.append((2, np.clip(points, 0.0, 1.0)))
    return result


def discover(root: Path, dataset: str) -> list[Record]:
    """Discover paired images and annotations in a staged dataset."""
    safe_extract_nested(root)
    names = {"dataset_sss_wrecks_aircraft": ["cylinder", "manta", "airplane", "shipwreck", "victim"], "fls_marine_debris": ["mine", "can", "bottle", "drink-carton", "chain", "propeller", "tire", "hook", "valve", "shampoo-bottle", "standing-bottle"], "SeabedObjects_KLSG": ["aircraft", "fish", "other", "shipwreck"]}.get(dataset, [])
    metadata = jsonl_index(root) if dataset == "GhostVision" else {}
    records: list[Record] = []
    for image in sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
        if "labels" in image.parts or "extras" in image.parts:
            continue
        annotations: list[tuple[int, np.ndarray]] = []
        if dataset == "AI4Shipwrecks":
            if image.parent.name == "images":
                mask = image.parent.parent / "labels" / image.name
                annotations = mask_annotations(mask) if mask.is_file() else []
        elif dataset == "GhostVision":
            probe = metadata.get(image.name)
            raw = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            annotations = jsonl_annotations(probe, raw.shape[1], raw.shape[0]) if probe and raw is not None else []
        else:
            candidates = [
                image.with_suffix(".txt"),
                image.parent.parent / "labels" / image.name.replace(image.suffix, ".txt"),
                image.parent.parent / "labels" / f"{image.stem}.txt",
                image.parent.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt",
            ]
            label = next((candidate for candidate in candidates if candidate.is_file()), None)
            annotations = parse_yolo(label, names, dataset) if label else []
        relative_parts = image.parts
        split_name = next((part for part in ("train", "valid", "test") if part in relative_parts), "all")
        stem_tokens = image.stem.split("_")
        group = f"{split_name}_{'_'.join(stem_tokens[:3])}" if len(stem_tokens) >= 3 else f"{split_name}_{image.stem}"
        records.append(Record(image, annotations, dataset, group))
    LOGGER.info("%s: discovered %d images", dataset, len(records))
    return records


def standardize(image: np.ndarray) -> np.ndarray:
    """Convert source imagery to robust 640x640 uint8 grayscale."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA).astype(np.float32)
    finite = resized[np.isfinite(resized)]
    if finite.size:
        low, high = np.percentile(finite, (1, 99))
        resized = np.zeros_like(resized) if high <= low else np.clip((resized - low) * 255.0 / (high - low), 0, 255)
    return resized.astype(np.uint8)


def select_records(records: list[Record], seed: int = 42) -> list[Record]:
    """Apply rare-class retention, crab-only cap, and 7% negative selection."""
    rng = random.Random(seed)
    rare = [record for record in records if any(class_id in {0, 1, 3} for class_id, _ in record.annotations)]
    crab_only = [record for record in records if record.annotations and all(class_id == 2 for class_id, _ in record.annotations)]
    mixed = [record for record in records if record.annotations and record not in rare and record not in crab_only]
    negatives = [record for record in records if not record.annotations]
    rng.shuffle(crab_only)
    rng.shuffle(negatives)
    selected = rare + mixed + crab_only[:450]
    negative_count = min(len(negatives), max(1, round(len(selected) * 0.07 / 0.93)))
    selected.extend(negatives[:negative_count])
    return selected


def stratified_group_split(records: list[Record], seed: int = 42) -> tuple[list[Record], list[Record]]:
    """Assign whole source groups to an approximately 85/15 stratified split."""
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in records:
        groups[(record.dataset, record.group)].append(record)
    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    total = len(records)
    target_val = max(1, round(total * 0.15))
    total_classes = Counter(class_id for record in records for class_id, _ in record.annotations)
    target_classes = {class_id: max(1, round(count * 0.15)) for class_id, count in total_classes.items()}
    group_counts = {
        key: Counter(class_id for record in items for class_id, _ in record.annotations)
        for key, items in group_items
    }
    val_keys: set[tuple[str, str]] = set()
    remaining = list(group_items)
    val_classes = Counter()
    val_count = 0

    # Reserve a whole source group for each populated class, choosing the group
    # that gets closest to that class's 15% target without splitting provenance.
    for class_id in sorted(target_classes, key=lambda item: total_classes[item]):
        candidates = [(key, items) for key, items in remaining if group_counts[key][class_id] > 0]
        if not candidates:
            continue
        selected_key, selected_items = min(
            candidates,
            key=lambda pair: (
                abs(target_classes[class_id] - (val_classes[class_id] + group_counts[pair[0]][class_id])),
                len(pair[1]),
            ),
        )
        val_keys.add(selected_key)
        remaining.remove((selected_key, selected_items))
        val_count += len(selected_items)
        val_classes.update(group_counts[selected_key])

    while remaining and val_count < target_val:
        def score(pair: tuple[tuple[str, str], list[Record]]) -> float:
            key, items = pair
            candidate_classes = group_counts[key]
            before = sum(abs(target_classes[class_id] - val_classes[class_id]) / target_classes[class_id] for class_id in target_classes)
            after = sum(abs(target_classes[class_id] - (val_classes[class_id] + candidate_classes[class_id])) / target_classes[class_id] for class_id in target_classes)
            size_penalty = abs((val_count + len(items)) - target_val) / max(target_val, 1)
            return (before - after) - size_penalty
        selected_key, selected_items = max(remaining, key=score)
        val_keys.add(selected_key)
        val_count += len(selected_items)
        val_classes.update(group_counts[selected_key])
        remaining.remove((selected_key, selected_items))
    train = [item for key, items in group_items if key not in val_keys for item in items]
    val = [item for key, items in group_items if key in val_keys for item in items]
    return train, val


def write_dataset(records: list[Record], output_root: Path, source_root: Path) -> Counter[int]:
    """Write balanced images and labels with deterministic split assignment."""
    if output_root.exists():
        def clear_readonly(function: Any, path: str, _: Any) -> None:
            Path(path).chmod(stat.S_IWRITE)
            function(path)
        shutil.rmtree(output_root, onerror=clear_readonly)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True)
        (output_root / "labels" / split).mkdir(parents=True)
    train, val = stratified_group_split(records)
    counts = Counter()
    for split, split_records in (("train", train), ("val", val)):
        for index, record in enumerate(split_records):
            image = cv2.imread(str(record.image), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            stem = f"{record.dataset}_{record.image.stem}_{index}"
            cv2.imwrite(str(output_root / "images" / split / f"{stem}.png"), standardize(image))
            lines = []
            for class_id, points in record.annotations:
                lines.append(str(class_id) + " " + " ".join(f"{float(value):.6f}" for value in np.clip(points, 0, 1).reshape(-1)))
                counts[(split, class_id)] += 1
            (output_root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return counts


def main() -> None:
    """Audit staging, rebuild balanced output, and print class counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, default=Path("data/benchmark_images/unzipped"))
    parser.add_argument("--output-root", type=Path, default=Path("data/balanced_unified_v2"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    datasets = [("AI4Shipwrecks", "AI4Shipwrecks"), ("GhostVision", "GhostVision"), ("SeabedObjects_KLSG", "SeabedObjects_KLSG"), ("fls_marine_debris", "fls_marine_debris"), ("sss_wrecks_aircraft", "dataset_sss_wrecks_aircraft")]
    all_records: list[Record] = []
    for folder, dataset in datasets:
        all_records.extend(discover(args.staging_root / folder, dataset))
    selected = select_records(all_records)
    counts = write_dataset(selected, args.output_root, args.staging_root)
    print(f"Selected records: {len(selected)}")
    print(f"Train records: {sum(1 for record in selected if record in stratified_group_split(selected)[0])}")
    print(f"Output: {args.output_root}")
    print(f"{'Class':<22} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-" * 50)
    for class_id, name in CLASS_NAMES.items():
        train_count, val_count = counts[("train", class_id)], counts[("val", class_id)]
        print(f"{class_id}: {name:<18} {train_count:>8} {val_count:>8} {train_count + val_count:>8}")


if __name__ == "__main__":
    main()
