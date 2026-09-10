"""Hydrographic acoustic-shadow and relief-verification helpers for AI-only false-alarm rejection."""

from __future__ import annotations

import cv2
import numpy as np


def calculate_target_slant_range(bbox, image_width, active_altitude, active_slant_range):
    """Calculate target slant range from cross-track pixel geometry."""
    try:
        x1, _, x2, _ = [float(value) for value in bbox]
        width = float(image_width)
        altitude = float(active_altitude)
        swath_range = float(active_slant_range)
        if width <= 0 or altitude < 0 or swath_range < 0:
            return 0.0

        center_x = (x1 + x2) / 2.0
        dx_px = abs(center_x - (width / 2.0))
        meters_per_pixel = swath_range / (width / 2.0) if width else 0.0
        ground_dist_m = dx_px * meters_per_pixel
        return round(float((altitude**2 + ground_dist_m**2) ** 0.5), 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def detect_acoustic_shadow(tile_gray, bbox, search_direction='auto', shadow_intensity_thresh=30):
    """Detect a downstream acoustic shadow corridor and compute its pixel length.

    Parameters
    ----------
    tile_gray:
        2D uint8 or float grayscale tile image containing the highlight and downstream shadow.
    bbox:
        Proposed detection bounding box [x1, y1, x2, y2] measured in image pixel coordinates.
    search_direction:
        'auto' is supported; left/right vector search is inferred from image center/nadir.
    shadow_intensity_thresh:
        Threshold below which a pixel is considered an acoustic shadow, default 30.

    Returns
    -------
    dict
        {
            'has_shadow': bool,
            'shadow_length_px': float,
            'shadow_bbox': [sx1, sy1, sx2, sy2],
        }
    """
    if tile_gray is None:
        raise ValueError('tile_gray must be a valid 2D image matrix')
    image = np.asarray(tile_gray)
    if image.ndim != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    else:
        x1, y1, x2, y2 = 0, 0, image.shape[1] - 1, image.shape[0] - 1

    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
    img_h, img_w = image.shape[:2]
    if x2 <= x1 or y2 <= y1:
        return {'has_shadow': False, 'shadow_length_px': 0.0, 'shadow_bbox': [0, 0, 0, 0]}

    # Determine propagation axis from center of tile; side-scan sonar records are fundamentally left/right from nadir.
    center_x = img_w // 2
    # Search side downstream of the highlight bbox away from the center line.
    if search_direction == 'auto':
        direction = 'right' if x1 >= center_x else 'left'
    elif search_direction in ('left', 'right'):
        direction = search_direction
    else:
        direction = 'right'

    y_start = max(0, y1)
    y_end = min(img_h, y2)
    # Estimate a local seabed background around the target and derive a relative
    # void threshold. This remains useful after CLAHE and across gain changes.
    band = image[max(0, y1 - max(4, (y2 - y1) // 2)):min(img_h, y2 + max(4, (y2 - y1) // 2) + 1)]
    background_samples = np.concatenate(
        [band[:, :max(1, x1)], band[:, min(img_w, x2 + 1):]], axis=1
    ) if x1 > 0 or x2 + 1 < img_w else band
    background = float(np.percentile(background_samples, 60)) if background_samples.size else float(shadow_intensity_thresh)
    dynamic_threshold = min(float(shadow_intensity_thresh), max(8.0, background * 0.55))

    search_start = x2 if direction == 'right' else x1 - 1
    search_stop = img_w if direction == 'right' else -1
    search_step = 1 if direction == 'right' else -1
    row_hits = []
    for yy in range(y_start, y_end + 1):
        run = []
        for xx in range(search_start, search_stop, search_step):
            if float(image[yy, xx]) < dynamic_threshold:
                run.append(xx)
            elif run:
                break
        if len(run) >= 2:
            row_hits.append((yy, run))

    if not row_hits:
        return {'has_shadow': False, 'shadow_length_px': 0.0, 'shadow_bbox': [0, 0, 0, 0]}

    shadow_xs = [x for _, run in row_hits for x in run]
    shadow_ys = [y for y, run in row_hits for _ in run]
    edge = x2 if direction == 'right' else x1
    shadow_len_px = max(abs(x - edge) + 1 for x in shadow_xs)
    return {
        'has_shadow': True,
        'shadow_length_px': float(shadow_len_px),
        'shadow_bbox': [int(min(shadow_xs)), int(min(shadow_ys)), int(max(shadow_xs)), int(max(shadow_ys))],
    }


def verify_hazard_relief(shadow_len_px, towfish_altitude_m=12.0, slant_range_m=50.0, meters_per_pixel=0.1):
    """Convert a detected shadow corridor into an estimated relief height and gate it physically.

    H = (L_shadow_m * H_towfish) / R_slant
    If the estimated height is less than 0.25 m, reject as flat clutter instead of a hazard.
    """
    try:
        L_shadow_m = float(shadow_len_px) * float(meters_per_pixel)
        H_towfish = float(towfish_altitude_m)
        R_slant = float(slant_range_m)
        if R_slant <= 0:
            return {'is_valid_hazard': False, 'estimated_height_m': 0.0}
        H = (L_shadow_m * H_towfish) / R_slant
        if H < 0.25:
            return {'is_valid_hazard': False, 'estimated_height_m': round(float(max(0.0, H)), 2)}
        return {'is_valid_hazard': True, 'estimated_height_m': round(float(H), 2)}
    except Exception:
        return {'is_valid_hazard': False, 'estimated_height_m': 0.0}
