"""Hydrographic acoustic-shadow and relief-verification helpers for AI-only false-alarm rejection."""

from __future__ import annotations

import cv2
import numpy as np


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

    # Use a corridor along the object footprint away from nadir. Find contiguous low-intensity region.
    corridor_x1 = max(0, x1)
    corridor_x2 = min(img_w, x2)
    y_start = max(0, y1)
    y_end = min(img_h, y2)

    if direction == 'right':
        search_xs = range(corridor_x2, img_w)
    else:
        search_xs = range(corridor_x1 - 1, -1, -1)

    # Downstream corridor is in the same row band as bbox; sample a vertical band around highlight.
    mask = np.zeros_like(image, dtype=np.uint8)
    for row in range(y_start, min(y_end + 1, img_h)):
        for col in range(corridor_x1, min(corridor_x2 + 1, img_w)):
            if image[row, col] < shadow_intensity_thresh:
                mask[row, col] = 255

    # Expand to search corridor downstream away from the object center.
    shadow_pixels = []
    for yy in range(y_start, min(y_end + 1, img_h)):
        for xx in range(corridor_x1, min(corridor_x2 + 1, img_w)):
            if image[yy, xx] < shadow_intensity_thresh:
                shadow_pixels.append((yy, xx))

    # Use the object mask contiguity if there are low-intensity pixels below threshold. Estimate width as max contiguous length.
    shadow_len_px = 0.0
    shadow_x1 = img_w
    shadow_y1 = img_h
    shadow_x2 = 0
    shadow_y2 = 0

    for yy in range(y_start, min(y_end + 1, img_h)):
        reaching = False
        x_lo = None
        for xx in range(corridor_x1, min(corridor_x2 + 1, img_w)):
            # Search from bbox edge to downstream side.
            if direction == 'right':
                corridor_col = xx
            else:
                corridor_col = -1 - xx
            if image[yy, min(img_w - 1, max(0, corridor_col))] < shadow_intensity_thresh:
                reaching = True
                if x_lo is None:
                    x_lo = xx
                shadow_len_px = max(shadow_len_px, abs(xx - x1))
                shadow_x1 = min(shadow_x1, xx)
                shadow_y1 = min(shadow_y1, yy)
                shadow_x2 = max(shadow_x2, xx)
                shadow_y2 = max(shadow_y2, yy)
        if reaching:
            pass
    if shadow_len_px <= 0:
        # fallback from low intensity contiguous pixels in a downstream object-aligned search band.
        if shadow_pixels:
            ys, xs = zip(*shadow_pixels)
            shadow_len_px = float(max(xs) - min(xs))
            shadow_x1 = min(xs)
            shadow_y1 = min(ys)
            shadow_x2 = max(xs)
            shadow_y2 = max(ys)

    has_shadow = shadow_len_px >= 1.0
    return {
        'has_shadow': bool(has_shadow),
        'shadow_length_px': float(shadow_len_px),
        'shadow_bbox': [int(shadow_x1), int(shadow_y1), int(shadow_x2), int(shadow_y2)],
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
