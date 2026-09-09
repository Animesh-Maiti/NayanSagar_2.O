"""Hydrographic geometry helpers for converting sonar shadow lengths to hazard height estimates."""

from __future__ import annotations


def estimate_hazard_height(shadow_length_m, towfish_altitude_m, slant_range_m):
    """Estimate target height using the acoustic shadow geometry relation H_t = (L_s * H_a) / R_s.

    Parameters
    ----------
    shadow_length_m:
        Acoustic shadow length in meters.
    towfish_altitude_m:
        Towfish altitude above sea floor or local datum in meters.
    slant_range_m:
        Slant range in meters, as recorded by sonar packet telemetry.

    Returns
    -------
    float
        Sanity-bounded height estimate in meters, clipped to [0, 50].
    """
    try:
        shadow_length = float(shadow_length_m)
        altitude = float(towfish_altitude_m)
        slant_range = float(slant_range_m)
    except (TypeError, ValueError):
        return 0.0

    if slant_range <= 0:
        return 0.0
    if shadow_length < 0:
        shadow_length = abs(shadow_length)
    if altitude < 0:
        altitude = abs(altitude)

    height = (shadow_length * altitude) / slant_range
    if height != height or height in (float('inf'), -float('inf')):
        return 0.0
    # Hydrographic acoustic output bounds.
    if height < 0:
        height = 0.0
    if height > 50:
        height = 50.0
    return float(height)
