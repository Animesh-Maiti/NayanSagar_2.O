"""Tile extraction and coordinate remapping helpers for sonar waterfall matrices."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def slice_waterfall(waterfall_img, tile_size=640, overlap=128):
    """Return a list of overlapping tiles with source offsets as requested by the Streamlit dashboard.

    Parameters
    ----------
    waterfall_img:
        2D grayscale or color matrix.
    tile_size:
        Target square tile size. Defaults to 640.
    overlap:
        Pixel overlap between adjacent windows. Defaults to 100.

    Returns
    -------
    list[tuple[np.ndarray, int, int, str]]
        Each tuple contains (tile, x_offset, y_offset, slice_id).
    """
    matrix = np.asarray(waterfall_img)
    if matrix.ndim != 2:
        raise ValueError('waterfall_img must be a 2D numpy array')
    if tile_size <= 0:
        raise ValueError('tile_size must be a positive integer')
    if overlap < 0:
        raise ValueError('overlap must be a non-negative integer')

    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return []

    stride = max(1, tile_size - overlap)

    def window_starts(length):
        if length <= tile_size:
            return [0]
        starts = list(range(0, length - tile_size + 1, stride))
        final_start = length - tile_size
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    slices = []
    slice_id = 0
    for y in window_starts(rows):
        for x in window_starts(cols):
            y2 = min(y + tile_size, rows)
            x2 = min(x + tile_size, cols)
            tile = matrix[y:y2, x:x2]
            slices.append((tile.copy(), x, y, f'slice_{slice_id:04d}'))
            slice_id += 1

    return slices


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
