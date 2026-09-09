"""Confidence filtering and label remapping utilities for sonar hazard detections."""

from __future__ import annotations

import copy
from typing import Any

DEFAULT_CONF_THRESHOLDS = {
    'aircraft_wreck': 0.35,
    'ghost_net': 0.30,
    'debris_highlight': 0.20,
    'shipwreck': 0.18,
    'crab_pot_trap': 0.15,
}

CLASS_REMAP = {
    'crab_pot_trap': 'debris_highlight',
    'shipwreck': 'shipwreck',
    'aircraft_wreck': 'aircraft_wreck',
    'ghost_net': 'ghost_net',
    'debris_highlight': 'debris_highlight',
}


def _class_name(detection: dict[str, Any]) -> str:
    """Extract a canonical class name from varying detection dict layouts."""
    if isinstance(detection, dict):
        for key in ('class_name', 'label', 'class', 'hazard_class', 'name'):
            if key in detection:
                value = str(detection[key])
                if value:
                    return value.lower().replace(' ', '_')
        for key in ('class_id', 'id'):
            if key in detection:
                value = detection[key]
                if isinstance(value, int):
                    return {
                        0: 'shipwreck',
                        1: 'aircraft_wreck',
                        2: 'crab_pot_trap',
                        3: 'ghost_net',
                        4: 'debris_highlight',
                    }.get(value, str(value))
    return 'unknown'


def _confidence(detection: dict[str, Any]) -> float:
    """Return a float confidence value from a detection dict."""
    if not isinstance(detection, dict):
        return 0.0
    for key in ('confidence', 'conf', 'score', 'probability'):
        if key in detection:
            try:
                return float(detection[key])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def filter_and_remap(detections, custom_thresholds=None, remap_enabled=True):
    """Filter weak detections and optionally remap ambiguous sonar clutter labels.

    Parameters
    ----------
    detections:
        Iterable of detection dictionaries, or a single detection dictionary.
    custom_thresholds:
        Optional dictionary of class labels to custom confidence thresholds.
    remap_enabled:
        When True apply CLASS_REMAP to labels such as crab_pot_trap -> debris_highlight.

    Returns
    -------
    list[dict]
        Filtered detections in the same structure as the input, with remapped labels.
    """
    if detections is None:
        return []
    if isinstance(detections, dict):
        detections = [detections]

    thresholds = dict(DEFAULT_CONF_THRESHOLDS)
    if custom_thresholds:
        thresholds.update({str(k).lower().replace(' ', '_'): float(v) for k, v in custom_thresholds.items()})

    filtered: list[dict[str, Any]] = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        class_name = _class_name(item)
        conf = _confidence(item)
        threshold = thresholds.get(class_name, 0.0)
        if conf < threshold:
            continue

        out = copy.deepcopy(item)
        if remap_enabled and class_name in CLASS_REMAP:
            target = CLASS_REMAP[class_name]
            for key in ('class_name', 'label', 'class', 'hazard_class', 'name'):
                if key in out:
                    out[key] = target
                    break
            if 'class_id' in out:
                class_id_labels = {
                    0: 'shipwreck',
                    1: 'aircraft_wreck',
                    2: 'debris_highlight',
                    3: 'ghost_net',
                    4: 'debris_highlight',
                }
                out['class_id'] = {value: key for key, value in class_id_labels.items()}.get(target, out.get('class_id'))

        filtered.append(out)

    return filtered
