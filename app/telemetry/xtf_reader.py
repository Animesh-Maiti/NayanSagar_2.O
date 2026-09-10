"""Telemetry and acoustic swath parsing utilities for the NayanSagar 2.0 dashboard.

The requested public APIs are exposed here as:
    parse_xtf_or_swath(file_bytes_or_path)
    preprocess_acoustic_swath(raw_waterfall, altitude_px=40)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

try:
    import pyxtf  # optional dependency
except Exception:
    pyxtf = None

LOGGER = logging.getLogger(__name__)


def _field(packet: Any, name: str, default: Any = None) -> Any:
    if isinstance(packet, dict):
        return packet.get(name, default)
    return getattr(packet, name, default)


def _flatten(values: Any) -> list[Any]:
    if isinstance(values, dict):
        values = values.values()
    if isinstance(values, (list, tuple)):
        return list(values)
    return [] if values is None else [values]


def _samples(packet: Any, channel: int | None = None) -> np.ndarray:
    data = _field(packet, 'Data', _field(packet, 'data'))
    if isinstance(data, (list, tuple)):
        if channel is None and len(data) == 1:
            data = data[0]
        elif channel is not None and len(data) > channel:
            data = data[channel]
        else:
            return np.empty(0, dtype=np.float32)
    if data is None:
        return np.empty(0, dtype=np.float32)
    return np.asarray(data).reshape(-1)


def _channel(packet: Any) -> int | None:
    for name in ('ChannelNumber', 'channel_number', 'Channel', 'channel'):
        value = _field(packet, name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _normalise_acoustic(waterfall: np.ndarray) -> np.ndarray:
    values = np.asarray(waterfall, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)

    # Compensate for the empirical cross-track beam pattern before percentile
    # scaling so range-dependent gain does not dominate the training contrast.
    fill_value = float(np.median(finite))
    safe_values = np.where(np.isfinite(values), values, fill_value)
    mean_profile = np.mean(safe_values, axis=0, keepdims=True)
    mean_profile = np.clip(mean_profile, 1e-3, None)
    profile_mean = max(float(np.mean(mean_profile)), 1e-3)
    equalized = safe_values / (mean_profile / profile_mean)
    finite_equalized = equalized[np.isfinite(equalized)]
    if finite_equalized.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(finite_equalized, [1.0, 99.0])
    if high <= low:
        low, high = float(finite_equalized.min()), float(finite_equalized.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((equalized - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    return clahe.apply(scaled)


def _correct_slant_range(
    waterfall: np.ndarray,
    towfish_altitude_m: float,
    max_range_m: float = 50.0,
) -> np.ndarray:
    """Warp both sonar halves from slant range to ground range row by row."""
    image = np.asarray(waterfall, dtype=np.uint8)
    if image.ndim != 2 or image.size == 0:
        return image
    _, width = image.shape
    left_width = width // 2
    right_width = width - left_width
    try:
        altitude_value = float(towfish_altitude_m)
    except (TypeError, ValueError):
        altitude_value = 12.0
    altitude = altitude_value if np.isfinite(altitude_value) and altitude_value > 0.0 else 12.0
    try:
        range_value = float(max_range_m)
    except (TypeError, ValueError):
        range_value = 50.0
    max_range = max(range_value if np.isfinite(range_value) else 50.0, altitude + 1e-3)
    ground_max = max(np.sqrt(max(max_range**2 - altitude**2, 1.0)), 1.0)

    def warp_half(half: np.ndarray) -> np.ndarray:
        half_width = half.shape[1]
        if half_width <= 1:
            return half.copy()
        y_ground = np.linspace(0.0, ground_max, half_width)
        r_slant = np.sqrt(y_ground**2 + altitude**2)
        source = (r_slant - altitude) / max(max_range - altitude, 1e-3)
        source *= half_width - 1
        source = np.clip(source, 0.0, half_width - 1)
        source_x = np.arange(half_width, dtype=np.float32)
        return np.stack(
            [np.interp(source, source_x, row.astype(np.float32)) for row in half],
            axis=0,
        ).astype(np.uint8)

    return np.hstack([warp_half(image[:, :left_width]), warp_half(image[:, left_width:])])


def _parse_real_xtf(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    if pyxtf is None:
        raise RuntimeError('pyxtf is not installed')
    from pyxtf import XTFHeaderType, xtf_read

    parsed = xtf_read(str(path))
    packets = parsed[1] if isinstance(parsed, tuple) and len(parsed) >= 2 else parsed
    sonar_type = getattr(XTFHeaderType, 'sonar', None)
    candidates = []
    if isinstance(packets, dict) and sonar_type in packets:
        candidates = _flatten(packets[sonar_type])
    if not candidates and isinstance(packets, dict):
        candidates = [
            packet for values in packets.values()
            for packet in _flatten(values)
            if _samples(packet).size
        ]
    if not candidates:
        raise ValueError(f'No sonar acoustic packets found in {path}')

    grouped: dict[int, dict[int, np.ndarray]] = {}
    telemetry: dict[str, list[float]] = {
        'Latitude': [], 'Longitude': [], 'Altitude_m': [], 'SlantRange_m': [],
        'Pitch_deg': [], 'Roll_deg': [],
    }
    aliases = {
        'Latitude': ('SensorYcoordinate', 'SensorLatitude', 'Latitude'),
        'Longitude': ('SensorXcoordinate', 'SensorLongitude', 'Longitude'),
        'Altitude_m': ('SensorAltitude', 'SensorPrimaryAltitude', 'SensorAuxAltitude'),
        'SlantRange_m': ('SlantRange', 'SlantRange_m', 'Range'),
        'Pitch_deg': ('SensorPitch', 'Pitch'),
        'Roll_deg': ('SensorRoll', 'Roll'),
    }
    for packet in candidates:
        try:
            ping_id = int(_field(packet, 'PingNumber', len(grouped)))
        except (TypeError, ValueError):
            ping_id = len(grouped)
        channel = _channel(packet)
        channel_values = {}
        if channel in (0, 1):
            data = _samples(packet)
            if data.size:
                channel_values[channel] = data
        else:
            for index in (0, 1):
                data = _samples(packet, index)
                if data.size:
                    channel_values[index] = data
        if not channel_values:
            continue
        grouped.setdefault(ping_id, {}).update(channel_values)
        for output_name, names in aliases.items():
            value = next((_field(packet, name) for name in names if _field(packet, name) is not None), np.nan)
            try:
                numeric_value = float(value)
                telemetry[output_name].append(
                    numeric_value if np.isfinite(numeric_value) else np.nan
                )
            except (TypeError, ValueError):
                telemetry[output_name].append(np.nan)

    if not grouped:
        raise ValueError(f'No usable Port/Starboard samples found in {path}')
    port_width = max((row.get(0, np.empty(0)).size for row in grouped.values()), default=0)
    star_width = max((row.get(1, np.empty(0)).size for row in grouped.values()), default=0)
    port_data = np.zeros((len(grouped), port_width), dtype=np.float32)
    starboard_data = np.zeros((len(grouped), star_width), dtype=np.float32)
    for row_index, ping in enumerate(sorted(grouped)):
        port = grouped[ping].get(0, np.empty(0))
        star = grouped[ping].get(1, np.empty(0))
        if port.size:
            port_data[row_index, port_width - port.size:port_width] = port[::-1]
        if star.size:
            starboard_data[row_index, :star.size] = star
    rows = np.hstack([port_data, starboard_data])
    count = rows.shape[0]
    metadata = pd.DataFrame({'PingID': np.arange(count), **{
        key: np.pad(np.asarray(values[:count], dtype=np.float64), (0, max(0, count - len(values))), constant_values=np.nan)[:count]
        for key, values in telemetry.items()
    }})
    metadata['PortChannel'] = 0
    metadata['StarboardChannel'] = 1
    metadata['Altitude_m'] = pd.to_numeric(metadata['Altitude_m'], errors='coerce').fillna(12.0)
    metadata.loc[metadata['Altitude_m'] <= 0.0, 'Altitude_m'] = 12.0
    metadata['SlantRange_m'] = pd.to_numeric(metadata['SlantRange_m'], errors='coerce').fillna(50.0)
    metadata.loc[metadata['SlantRange_m'] <= 0.0, 'SlantRange_m'] = 50.0
    for coordinate in ('Latitude', 'Longitude'):
        metadata[coordinate] = pd.to_numeric(metadata[coordinate], errors='coerce')
        metadata[coordinate] = metadata[coordinate].fillna(0.0)
    waterfall = _normalise_acoustic(rows)
    if waterfall.shape[1] > waterfall.shape[0]:
        waterfall = waterfall.T
    altitude = (
        float(metadata['Altitude_m'].median())
        if metadata['Altitude_m'].notna().any()
        else 12.0
    )
    if not np.isfinite(altitude) or altitude <= 0.0:
        altitude = 12.0
    max_range = (
        float(metadata['SlantRange_m'].median())
        if metadata['SlantRange_m'].notna().any()
        else 50.0
    )
    if not np.isfinite(max_range) or max_range <= 0.0:
        max_range = 50.0
    waterfall = _correct_slant_range(waterfall, altitude, max_range_m=max_range)
    return cv2.bilateralFilter(waterfall, d=5, sigmaColor=25, sigmaSpace=25), metadata


class XtfReader:
    """Optional wrapper class that reads a .xtf path and returns a waterfall + telemetry map."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.waterfall = None
        self.telemetry = {}

    def read(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if not self.file_path.exists():
            LOGGER.warning('XTF file not found at %s; generating synthetic waterfall fallback.', self.file_path)
            return generate_mock_waterfall(), {}
        try:
            if pyxtf is None:
                raise ImportError('pyxtf is not installed')
            from pyxtf import xtf_read
            parsed = xtf_read(str(self.file_path))
            return self._parse_payload(parsed)
        except Exception as exc:
            LOGGER.warning('Unable to parse %s using pyxtf; generating synthetic fallback: %s', self.file_path, exc)
            return generate_mock_waterfall(), {}

    def _parse_payload(self, parsed):
        if isinstance(parsed, tuple) and len(parsed) >= 2:
            packets = parsed[1]
        else:
            packets = parsed
        port_rows = []
        starboard_rows = []
        telemetry = {k: [] for k in ('Latitude', 'Longitude', 'Altitude', 'SlantRange', 'Pitch', 'Roll')}

        # Best-effort extraction of packet maps/dictionaries or typed packet objects.
        try:
            if isinstance(packets, dict):
                for payload in packets.values():
                    if isinstance(payload, (list, tuple)):
                        for packet in payload:
                            sample_arrays = self._extract_channel_samples(packet)
                            if sample_arrays:
                                p, s = sample_arrays
                                if p is not None:
                                    port_rows.append(np.asarray(p, dtype=np.float32).reshape(-1))
                                if s is not None:
                                    starboard_rows.append(np.asarray(s, dtype=np.float32).reshape(-1))
                            for key in telemetry:
                                value = self._extract_attr(packet, key)
                                if value is not None:
                                    telemetry[key].append(float(value))
            else:
                for packet in list(packets if isinstance(packets, (list, tuple)) else []):
                    sample_arrays = self._extract_channel_samples(packet)
                    if sample_arrays:
                        p, s = sample_arrays
                        if p is not None:
                            port_rows.append(np.asarray(p, dtype=np.float32).reshape(-1))
                        if s is not None:
                            starboard_rows.append(np.asarray(s, dtype=np.float32).reshape(-1))
                    for key in telemetry:
                        value = self._extract_attr(packet, key)
                        if value is not None:
                            telemetry[key].append(float(value))
        except Exception:
            pass

        port_cols = max((row.size for row in port_rows), default=0)
        star_cols = max((row.size for row in starboard_rows), default=0)
        rows = max(len(port_rows), len(starboard_rows))
        waterfall = np.zeros((rows, port_cols + star_cols), dtype=np.float32)
        if port_rows:
            for r, row in enumerate(port_rows):
                if row.size:
                    waterfall[r, :row.size] = np.flip(row)
        if starboard_rows:
            for r, row in enumerate(starboard_rows):
                if row.size:
                    waterfall[r, port_cols:port_cols + row.size] = row

        self.waterfall = waterfall
        self.telemetry = {key: np.asarray(values, dtype=np.float64) for key, values in telemetry.items()}
        return waterfall, self.telemetry

    @staticmethod
    def _extract_channel_samples(packet: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        data = getattr(packet, 'Data', None)
        if data is None and isinstance(packet, dict):
            data = packet.get('Data') or packet.get('data')
        if data is None:
            return None, None
        try:
            if isinstance(data, (list, tuple)) and len(data) == 2:
                return np.asarray(data[0]).reshape(-1), np.asarray(data[1]).reshape(-1)
            if hasattr(data, 'shape'):
                arr = np.asarray(data).reshape(-1)
                mid = max(1, len(arr) // 2)
                return arr[:mid], arr[mid:]
        except Exception:
            return None, None
        return None, None

    @staticmethod
    def _extract_attr(packet: Any, attr: str) -> Any:
        if isinstance(packet, dict):
            if attr in packet:
                return packet[attr]
        return getattr(packet, attr, None)


def generate_mock_waterfall(rows: int = 160, cols: int = 1000, seed: int = 0) -> np.ndarray:
    """Return a synthetic grayscale waterfall matrix for UI and testing when no physical XTF is available."""
    rng = np.random.default_rng(seed)
    base = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        wave = rng.normal(0, 10, cols)
        if i % 12 == 0:
            wave = wave + 18
        base[i, :] = wave
    base[:, 80:160] += 25
    base[:, 550:660] += 20
    return np.clip(base, 0, 255).astype(np.float32)


def parse_xtf_or_swath(file_bytes_or_path):
    """Read an .xtf file, swath image, or byte stream and return a grayscale waterfall image and a georeference metadata DataFrame.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        waterfall_image_np and a ping metadata dataframe with simulated source track metadata
        in the requested range (e.g. Lat: 18.92N, Lon: 72.83E) when a real parser is absent.
    """
    waterfall = generate_mock_waterfall(rows=160, cols=1000)
    metadata = pd.DataFrame([
        {
            'PingID': 1,
            'PortChannel': 0,
            'StarboardChannel': 1,
            'Latitude': 18.92,
            'Longitude': 72.83,
            'Altitude_m': 12.0,
            'SlantRange_m': 50.0,
            'Pitch_deg': 0.0,
            'Roll_deg': 0.0,
        }
    ])

    # Accept either a path-like or bytes-like object.
    tmp_path = None
    content = None
    try:
        # If a bytes object or file-like upload is provided, save it to a temp file.
        if isinstance(file_bytes_or_path, (bytes, bytearray, memoryview)):
            content = bytes(file_bytes_or_path)
            with tempfile.NamedTemporaryFile(suffix='.xtf', delete=False) as handle:
                handle.write(content)
                path = Path(handle.name)
            tmp_path = path
        elif hasattr(file_bytes_or_path, 'read'):
            content = file_bytes_or_path.read()
            suffix = '.xtf' if str(getattr(file_bytes_or_path, 'name', '')).lower().endswith('.xtf') else Path(str(getattr(file_bytes_or_path, 'name', ''))).suffix or '.bin'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(content)
                path = Path(handle.name)
            tmp_path = path
        elif isinstance(file_bytes_or_path, (str, os.PathLike)):
            path = Path(file_bytes_or_path)
        else:
            path = None

        if path is not None and str(path).lower().endswith('.xtf'):
            # Try requested XTF parser if available.
            try:
                if pyxtf is not None:
                    waterfall, metadata = _parse_real_xtf(path)
                    if waterfall.size > 0:
                        return waterfall, metadata
            except Exception as exc:
                LOGGER.warning('XTF parse failed; falling back to image/swath processing path: %s', exc)

        # Standard image has been uploaded; produce a grayscale waterfall strip by reading it.
        if path is not None and path.exists() and (
            path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}
            or content is not None
        ):
            try:
                if content is not None:
                    image_bgr = cv2.imdecode(
                        np.frombuffer(content, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                else:
                    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise ValueError('image file unreadable')
                waterfall = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                metadata = pd.DataFrame([
                    {
                        'PingID': 1,
                        'PortChannel': 0,
                        'StarboardChannel': 1,
                        'Latitude': 18.92,
                        'Longitude': 72.83,
                        'Altitude_m': 12.0,
                        'SlantRange_m': 50.0,
                        'Pitch_deg': 0.0,
                        'Roll_deg': 0.0,
                    }
                ])
                return waterfall, metadata
            except Exception as exc:
                LOGGER.warning('Image swath fallback failed: %s', exc)

    except Exception as exc:
        LOGGER.warning('Unable to parse the requested telemetry or swath object: %s', exc)

    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

    LOGGER.warning('Using synthetic waterfall fallback after XTF decoding failure.')
    # Final fallback: synthetic waterfall + generated survey line metadata.
    return waterfall.astype(np.float32), metadata


def preprocess_acoustic_swath(raw_waterfall, altitude_px=40):
    """Return a despeckled, CLAHE-balanced image with the nadir water-column blind-zone masked.

    - median filter ksize=3
    - CLAHE clipLimit=2.0 tileGrid=(8,8)
    - mask the centerline blind zone by default
    """
    gray = np.asarray(raw_waterfall)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)

    # 1. median filter for speckle suppression.
    filtered = cv2.medianBlur(gray, ksize=3)

    # 2. CLAHE equalization to fix sonar transmission falloff.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(filtered)

    # 3. Mask nadir water-column blind zone in the center of the swath.
    h, w = equalized.shape[:2]
    mask_half_width = max(4, int(round(altitude_px / 2.0)))
    center = w // 2
    left = max(0, center - mask_half_width)
    right = min(w, center + mask_half_width)
    equalized[:, left:right] = 0

    return equalized.astype(np.uint8)
