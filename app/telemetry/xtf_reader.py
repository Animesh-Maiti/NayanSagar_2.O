"""Complete wrapper around optional pyxtf dependency for XTF parsing and synthetic telemetry fallbacks."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyxtf
except Exception:
    pyxtf = None

LOGGER = logging.getLogger(__name__)


class XtfReader:
    """Wrapper around pyxtf that offers a normalization-friendly parsing workflow for XTF telemetry."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.waterfall = None
        self.telemetry = {}

    def read(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return a unified waterfall image and navigation telemetry arrays from an XTF file or mock generator."""
        if not self.file_path.exists():
            LOGGER.warning('XTF file not found at %s; generating a synthetic waterfall fallback.', self.file_path)
            return generate_mock_waterfall(), {}

        try:
            if pyxtf is None:
                raise ImportError('pyxtf is not installed')
            from pyxtf import xtf_read
            parsed = xtf_read(str(self.file_path))
            return self._parse_payload(parsed)
        except Exception as exc:
            LOGGER.warning('Unable to parse %s using pyxtf; falling back to synthetic waterfall: %s', self.file_path, exc)
            return generate_mock_waterfall(), {}

    def _parse_payload(self, parsed: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Generic parser for pyxtf payload structures using robust extraction semantics."""
        # Support tuple of (header, packet_map), or mapping directly.
        if isinstance(parsed, tuple) and len(parsed) >= 2:
            packets = parsed[1]
        else:
            packets = parsed

        port_rows: list[np.ndarray] = []
        starboard_rows: list[np.ndarray] = []
        telemetry: dict[str, list[float]] = {k: [] for k in (
            'SensorLatitude', 'SensorLongitude', 'SensorAltitude', 'SlantRange'
        )}

        # Prefer packet dictionary walkers keyed by packet type.
        if isinstance(packets, dict):
            for key, payload in packets.items():
                if isinstance(payload, (list, tuple)):
                    for packet in payload:
                        sample_arrays = self._extract_channel_samples(packet)
                        if sample_arrays:
                            port_sample, star_sample = sample_arrays
                            if port_sample is not None:
                                port_rows.append(port_sample.astype(np.float32))
                            if star_sample is not None:
                                starboard_rows.append(star_sample.astype(np.float32))
                        for name in telemetry:
                            value = self._extract_attr(packet, name)
                            if value is not None:
                                telemetry[name].append(float(value))
        else:
            # Backend compatibility fallback.
            try:
                for packet in list(packets):
                    sample_arrays = self._extract_channel_samples(packet)
                    if sample_arrays:
                        port_sample, star_sample = sample_arrays
                        if port_sample is not None:
                            port_rows.append(port_sample.astype(np.float32))
                        if star_sample is not None:
                            starboard_rows.append(star_sample.astype(np.float32))
                    for name in telemetry:
                        value = self._extract_attr(packet, name)
                        if value is not None:
                            telemetry[name].append(float(value))
            except Exception:
                pass

        # Normalize sample lengths using interpolation to one common width.
        max_port = max((row.size for row in port_rows), default=1)
        max_star = max((row.size for row in starboard_rows), default=1)
        common_port = max_port
        common_star = max_star
        normalized_port = [self._resample_linear(row, common_port) for row in port_rows]
        normalized_star = [self._resample_linear(row, common_star) for row in starboard_rows]

        merged_rows = []
        for _ in range(max(len(normalized_port), len(normalized_star))):
            # Stack two swaths centrally: Port (flipped) left side, starboard right side.
            pass

        waterfall = self._stitch_port_starboard(normalized_port, normalized_star)
        telemetry_arrays = {key: np.asarray(values, dtype=np.float64) for key, values in telemetry.items()}
        self.waterfall = waterfall
        self.telemetry = telemetry_arrays
        return waterfall, telemetry_arrays

    @staticmethod
    def _extract_channel_samples(packet: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Extract Port (Chan 0) and Starboard (Chan 1) arrays from a packet-like object or dict."""
        data = getattr(packet, 'Data', None)
        if data is None and isinstance(packet, dict):
            data = packet.get('Data') or packet.get('data')
        if data is None:
            return None, None
        try:
            if isinstance(data, (list, tuple)) and len(data) == 2:
                port = np.asarray(data[0]).reshape(-1)
                starboard = np.asarray(data[1]).reshape(-1)
                return port, starboard
            if hasattr(data, 'shape'):
                arr = np.asarray(data).reshape(-1)
                # naive split of a 1D vector into two halves
                middle = max(1, len(arr) // 2)
                return arr[:middle], arr[middle:]
        except Exception:
            return None, None
        return None, None

    @staticmethod
    def _extract_attr(packet: Any, name: str) -> Any:
        """Robust field extraction from native objects or dict-like packet maps."""
        if isinstance(packet, dict):
            if name in packet:
                return packet[name]
        if hasattr(packet, name):
            return getattr(packet, name)
        return None

    @staticmethod
    def _resample_linear(array: np.ndarray, target_len: int) -> np.ndarray:
        """Normalize a 1D ping line to a requested length by linear interpolation."""
        src = np.asarray(array, dtype=np.float32)
        if src.size == 0:
            return np.zeros(target_len, dtype=np.float32)
        if src.size == target_len:
            return src.astype(np.float32)
        x_old = np.linspace(0, 1, src.size)
        x_new = np.linspace(0, 1, target_len)
        try:
            values = np.interp(x_new, x_old, src)
        except Exception:
            values = np.zeros(target_len, dtype=np.float32)
        return values.astype(np.float32)

    @staticmethod
    def _stitch_port_starboard(port_rows: list[np.ndarray], starboard_rows: list[np.ndarray]) -> np.ndarray:
        """Create a unified grayscale waterfall image from port/flipped and starboard arrays."""
        rows = max(len(port_rows), len(starboard_rows))
        if rows == 0:
            return np.zeros((0, 0), dtype=np.float32)
        port_cols = max((row.size for row in port_rows), default=0)
        star_cols = max((row.size for row in starboard_rows), default=0)
        width = port_cols + star_cols
        out = np.zeros((rows, width), dtype=np.float32)
        for i in range(rows):
            port = port_rows[i] if i < len(port_rows) else np.zeros(0, dtype=np.float32)
            star = starboard_rows[i] if i < len(starboard_rows) else np.zeros(0, dtype=np.float32)
            if port.size:
                port = np.flip(port)
                out[i, :port.size] = port
            if star.size:
                out[i, port_cols : port_cols + star.size] = star
        return out


def generate_mock_waterfall(rows: int = 160, cols: int = 1000, seed: int = 0) -> np.ndarray:
    """Return a synthetic grayscale waterfall matrix for UI and testing when no physical XTF is available."""
    rng = np.random.default_rng(seed)
    base = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        wave = rng.normal(0, 10, cols)
        if i % 12 == 0:
            wave = wave + 18
        base[i, :] = wave
    # Inject bright swaths for synthetic hazards.
    base[:, 80:160] += 25
    base[:, 550:660] += 20
    return np.clip(base, 0, 255).astype(np.float32)
