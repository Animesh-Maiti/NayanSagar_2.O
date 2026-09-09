#!/usr/bin/env python
"""Deterministic acoustic-shadow extraction utilities.

The extractor combines:
1. adaptive intensity thresholding on the local sonar image,
2. a conservative trackline/ray-casting window to sample the side of the target
   that is most likely to contain an acoustic shadow.

The code is intentionally dependency-light and designed for a workspace that
contains grayscale sonar images and YOLO-style polygon labels.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Iterable, Sequence


class AcousticShadowExtractor:
    """Deterministically locate dark contiguous shadow evidence from sonar imagery."""

    def __init__(self, image_size: int = 640):
        self.image_size = image_size

    def adaptive_threshold(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        """Return an Otsu-style threshold and mapping of dark pixels relative to the image background.

        The threshold is the weighted median of two cues:
            - Otsu threshold,
            - the image percentile of the lower 25% intensity band.
        """
        if image.ndim != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        otsu_value, otsu_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low = np.percentile(image, 25)
        combined = float(np.clip(0.65 * otsu_value + 0.35 * low, 0, 255))
        dark = (image < combined).astype(np.uint8)
        return combined, dark

    def trackline_ray_cast(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray | None:
        """Approximate a trackline ray from an object polygon to enforce a deterministic side direction.

        It selects a single best dark contour contiguous inside the target-adjacent region.
        """
        if contour.shape[0] < 3:
            return None

        pixels = np.round(contour * self.image_size).astype(np.int32)
        x, y, width, height = cv2.boundingRect(pixels.reshape(-1, 1, 2))
        center_x = x + width / 2.0
        direction = -1 if center_x < self.image_size / 2 else 1
        pad_y = max(4, int(height * 0.25))
        y0, y1 = max(0, y - pad_y), min(self.image_size, y + height + pad_y)
        search_width = max(24, min(3 * width, 220))

        if direction < 0:
            sx0, sx1 = max(0, x - search_width), x
        else:
            sx0, sx1 = x + width, min(self.image_size, x + width + search_width)

        if sx1 <= sx0 or y1 <= y0:
            return None

        window = image[y0:y1, sx0:sx1]
        if window.size == 0:
            return None

        # Determine a local median background strip from the opposite side of the object.
        background_strip = image[y0:y1, max(0, sx0 - 16):sx0] if direction < 0 else image[y0:y1, sx1:min(self.image_size, sx1 + 16)]
        if background_strip.size == 0:
            background_strip = image[y0:y1, :]
        local_background = float(np.median(background_strip))
        if local_background <= 1:
            return None

        threshold, _ = cv2.threshold(window, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = (window <= threshold).astype(np.uint8)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        target_area = max(float(cv2.contourArea(pixels.reshape(-1, 1, 2))), 1.0)
        candidates: list[tuple[float, np.ndarray]] = []
        for contour_item in contours:
            area = float(cv2.contourArea(contour_item))
            if not 0.25 * target_area <= area <= 3.0 * target_area:
                continue

            contour_item_shifted = contour_item.copy()
            contour_item_shifted[:, 0, 0] += sx0
            contour_item_shifted[:, 0, 1] += y0
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour_item_shifted], -1, 1, -1)
            mean_intensity = float(image[mask.astype(bool)].mean()) if mask.any() else 255.0
            if mean_intensity >= 0.35 * local_background:
                continue

            edge_distance = abs(float(cv2.boundingRect(contour_item_shifted)[0] - (x if direction < 0 else x + width)))
            candidates.append((edge_distance, contour_item_shifted))

        if not candidates:
            return None

        contour_best = min(candidates, key=lambda item: item[0])[1]
        points = contour_best.reshape(-1, 2).astype(np.float32) / self.image_size
        return points if len(points) >= 3 else None

    def extract_from_polygons(self, image: np.ndarray, polygons: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Return deterministic shadow contours from a list of object polygons."""
        shadows: list[np.ndarray] = []
        for polygon in polygons:
            shadow = self.trackline_ray_cast(image, polygon)
            if shadow is not None:
                shadows.append(shadow)
        return shadows


def extract_acoustic_shadows(image_path: str | Path, polygon_paths: Sequence[str | Path]) -> list[np.ndarray]:
    """Convenience wrapper that reads an image and a set of polygon files for shadow extraction."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    polygons: list[np.ndarray] = []
    for polygon_path in polygon_paths:
        contents = Path(polygon_path).read_text(encoding="utf-8", errors="replace").splitlines()
        for line in contents:
            fields = line.strip().split()
            if not fields:
                continue
            try:
                class_id = int(fields[0])
            except ValueError:
                continue
            # only keep object lines for the 5-class clean map; class 5 should already be stripped.
            if class_id < 0 or class_id > 4:
                continue
            numbers = [float(value) for value in fields[1:]]
            if len(numbers) >= 6 and len(numbers) % 2 == 0:
                points = np.asarray(numbers, dtype=np.float32).reshape(-1, 2)
                polygons.append(np.clip(points, 0.0, 1.0))

    extractor = AcousticShadowExtractor(image_size=image.shape[0])
    return extractor.extract_from_polygons(image, polygons)
