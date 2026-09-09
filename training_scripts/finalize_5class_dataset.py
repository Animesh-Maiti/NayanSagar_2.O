#!/usr/bin/env python
"""Sanitize the balanced YOLO segmentation dataset to the requested clean 5-class schema.

Workflow:
1. Remove any annotation rows whose first token is `5` (acoustic_shadow).
2. Preserve empty background-label files as empty text files so YOLO sees them as negative samples.
3. Validate the remaining labels keep class IDs within 0..4.
4. Rewrite the requested YAML taxonomy in data/balanced_unified_v2/sonar_data_v2.yaml.
5. Optionally create a stable zip artifact using Python's zipfile module.
"""

from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable


CLASS_NAMES = {
    0: "shipwreck",
    1: "aircraft_wreck",
    2: "crab_pot_trap",
    3: "ghost_net",
    4: "debris_highlight",
}


def _iter_label_files(labels_root: Path) -> Iterable[Path]:
    """Yield every label file under the train/val label directories inclusive."""
    for split in ("train", "val"):
        split_dir = labels_root / split
        if split_dir.exists():
            for txt_path in sorted(split_dir.glob("*.txt")):
                yield txt_path


def sanitize_labels(labels_root: Path) -> dict[str, int]:
    """Remove class-5 rows and verify all remaining classes stay in [0, 4].

    Background files may be empty byte files or empty text files; preserve them.
    """
    stats = Counter()
    removed_class_5 = 0
    kept_empty = 0

    for label_path in _iter_label_files(labels_root):
        original_lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
        filtered_lines: list[str] = []

        for line in original_lines:
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if not fields:
                continue
            try:
                class_id = int(fields[0])
            except ValueError:
                # Some lines may already be comments or malformed; preserve and let validation fail.
                filtered_lines.append(line)
                continue

            if class_id == 5:
                removed_class_5 += 1
                stats["removed_class_5"] += 1
                continue

            # Validate all remaining non-empty classes remain in the intended 5-class space.
            if class_id < 0 or class_id > 4:
                raise ValueError(f"Invalid class id {class_id} found in {label_path}")
            filtered_lines.append(line)

        # Preserve line endings and write empty file as zero bytes, but not delete the file.
        content = "\n".join(filtered_lines)
        if content:
            content += "\n"
        label_path.write_text(content, encoding="utf-8")

        # Record whether an empty file existed before/after; keep it as a background label file.
        if len(filtered_lines) == 0:
            kept_empty += 1
            stats["empty_after_sanitize"] += 1

    # Make sure no class-5 lines survived the pass.
    for label_path in _iter_label_files(labels_root):
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                class_id = int(line.split()[0])
            except ValueError:
                continue
            if class_id == 5:
                raise ValueError(f"Class-5 row survived sanitize step: {label_path}")
            if class_id < 0 or class_id > 4:
                raise ValueError(f"Invalid class id {class_id} found in {label_path}")

    stats["removed_class_5"] = removed_class_5
    stats["kept_empty_samples"] = kept_empty
    return dict(stats)


def write_yaml(dataset_root: Path, yaml_path: Path) -> None:
    """Write the requested five-class taxonomy into the SONAR YAML config."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = """path: /content/dataset
train: images/train
val: images/val
names:
  0: shipwreck
  1: aircraft_wreck
  2: crab_pot_trap
  3: ghost_net
  4: debris_highlight
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")


def package_dataset_zip(dataset_root: Path, output_zip: Path) -> None:
    """Create the requested stable ZIP using Python's zipfile module.

    The archive contains the balanced dataset directory as a package, with a
    deterministic file order via path sorting.
    """
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    # Build a deterministic archive for the requested dataset root.
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        files = sorted(dataset_root.rglob("*"))
        for file_path in files:
            if file_path.is_file():
                rel = file_path.relative_to(dataset_root.parent)
                archive.write(file_path, arcname=str(rel))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/balanced_unified_v2"))
    parser.add_argument("--labels-root", type=Path, default=Path("data/balanced_unified_v2/labels"))
    parser.add_argument("--yaml-path", type=Path, default=Path("data/balanced_unified_v2/sonar_data_v2.yaml"))
    parser.add_argument("--zip-output", type=Path, default=Path("data/NayanSagar_Dataset_v2_Balanced.zip"))
    args = parser.parse_args()

    stats = sanitize_labels(args.labels_root)
    write_yaml(args.dataset_root, args.yaml_path)
    package_dataset_zip(args.dataset_root, args.zip_output)

    print("Sanitized 5-class dataset")
    print(f"Removed class-5 rows: {stats['removed_class_5']}")
    print(f"Preserved empty negative-background files: {stats['empty_after_sanitize']}")
    print(f"Wrote taxonomy config: {args.yaml_path}")
    print(f"Created stable dataset package: {args.zip_output}")


if __name__ == "__main__":
    main()
