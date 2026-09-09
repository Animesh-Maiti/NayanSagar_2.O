"""Tile extraction and coordinate remapping helpers for sonar waterfall matrices."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def slice_waterfall(waterfall_matrix, tile_size=640, overlap_pct=0.15):
    """Slide a rectangular tile across a 2D waterfall matrix and yield tiles with global offsets.

    Yields
    ------
    tuple[np.ndarray, tuple[int, int, int, int]]
        (tile_image, (y1, x1, y2, x2)) where coordinates are bounded against source matrix.
    """
    matrix = np.asarray(waterfall_matrix)
    if matrix.ndim != 2:
        raise ValueError('waterfall_matrix must be a 2D numpy array')
    if tile_size <= 0:
        raise ValueError('tile_size must be a positive integer')
    if not 0 <= overlap_pct < 1:
        raise ValueError('overlap_pct must be in the range [0, 1)')

    stride = max(1, int(round(tile_size * (1 - overlap_pct))))
    h, w = matrix.shape
    y = 0
    while y < h:
        x = 0
        while x < w:
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            tile = matrix[y:y2, x:x2]
            if tile.size == 0:
                break
            yield tile, (y, x, y2, x2)
            if x2 >= w:
                break
            x += stride
        if y2 >= h:
            break
        y += stride


def remap_to_global_coords(tile_detections, global_offset):
    """Shift detection coordinates from a tile coordinate frame into a global source frame.

    tile_detections may be a list of dicts with bounding boxes or contours.
    global_offset is (y_offset, x_offset), typically the tile's source origin.
    """
    if not isinstance(tile_detections, list):
        raise TypeError('tile_detections must be a list of detection dictionaries')
    if len(global_offset) != 2:
        raise ValueError('global_offset must be a two-value tuple (y_offset, x_offset)')

    y_offset, x_offset = int(global_offset[0]), int(global_offset[1])
    remapped = []
    for detection in tile_detections:
        item = dict(detection)
        bbox = item.get('bbox')
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            item['bbox'] = [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset]
        contour = item.get('contour')
        if contour is not None:
            contour_arr = np.asarray(contour)
            if contour_arr.ndim == 2:
                shifted = np.empty_like(contour_arr)
                shifted[:, 0] = contour_arr[:, 0] + x_offset
                shifted[:, 1] = contour_arr[:, 1] + y_offset
                item['contour'] = shifted.astype(np.int32)
        remapped.append(item)
    return remapped
