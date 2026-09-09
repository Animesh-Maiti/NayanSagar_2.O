"""Ingest XTF side-scan sonar packets into a stitched waterfall and telemetry."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
LFS_POINTER_MARKER = b"version https://git-lfs.github.com/spec/v1"


def _field(packet: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an object-like or mapping-like packet."""
    if isinstance(packet, dict):
        return packet.get(name, default)
    return getattr(packet, name, default)


def _as_samples(packet: Any) -> np.ndarray:
    """Return a sonar packet's acoustic samples as a one-dimensional array."""
    data = _field(packet, "Data")
    if data is None:
        data = _field(packet, "data")
    if data is None:
        return np.empty(0, dtype=np.float32)
    return np.asarray(data).reshape(-1)


def _flatten_packets(packet_values: Any) -> list[Any]:
    """Flatten a pyxtf packet bucket while leaving individual packets intact."""
    if isinstance(packet_values, dict):
        packet_values = packet_values.values()
    if isinstance(packet_values, (list, tuple)):
        return [packet for packet in packet_values]
    return [packet_values]


def _channel_number(packet: Any) -> int | None:
    """Extract a zero-based channel number from a sonar packet."""
    for name in ("ChannelNumber", "channel_number", "Channel", "channel"):
        value = _field(packet, name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _channel_samples(packet: Any, channel: int) -> np.ndarray:
    """Read one channel from either a channel packet or a multi-channel ping."""
    data = _field(packet, "Data", _field(packet, "data"))
    if isinstance(data, (list, tuple)) and len(data) > channel:
        return np.asarray(data[channel]).reshape(-1)
    if channel == _channel_number(packet):
        return _as_samples(packet)
    return np.empty(0, dtype=np.float32)


def _packet_candidates(packets: Any, header_types: tuple[Any, ...]) -> list[Any]:
    """Select packets from keyed pyxtf output, including data-bearing fallbacks."""
    if not isinstance(packets, dict):
        return _flatten_packets(packets)

    LOGGER.info("Available XTF packet types: %s", [str(key) for key in packets.keys()])
    candidates: list[Any] = []
    for header_type in header_types:
        if header_type is not None and header_type in packets:
            candidates.extend(_flatten_packets(packets[header_type]))
    candidates = [packet for packet in candidates if _as_samples(packet).size]
    if candidates:
        return candidates

    for packet_values in packets.values():
        for packet in _flatten_packets(packet_values):
            if _as_samples(packet).size:
                candidates.append(packet)
    return candidates


def _header_summary(file_header: Any) -> str:
    """Return useful header fields without depending on one pyxtf version."""
    fields = {
        "Channels count": ("NumberOfSonarChannels", "NumberOfChannels", "channels"),
        "Sonar Name": ("SonarName", "sonar_name"),
        "System Type": ("SystemType", "system_type"),
    }
    values = []
    for label, names in fields.items():
        value = next((getattr(file_header, name) for name in names if hasattr(file_header, name)), "unknown")
        values.append(f"{label}={value}")
    return ", ".join(values)


def _pad_and_stitch(port_rows: list[np.ndarray], starboard_rows: list[np.ndarray]) -> np.ndarray:
    """Pad variable-length channel rows and concatenate port/starboard at nadir."""
    ping_count = max(len(port_rows), len(starboard_rows))
    if ping_count == 0:
        return np.empty((0, 0), dtype=np.float32)
    port_width = max((row.size for row in port_rows), default=0)
    starboard_width = max((row.size for row in starboard_rows), default=0)
    dtype = np.result_type(*(row.dtype for row in port_rows + starboard_rows), np.float32)
    waterfall = np.zeros((ping_count, port_width + starboard_width), dtype=dtype)
    for ping_index in range(ping_count):
        port = port_rows[ping_index] if ping_index < len(port_rows) else np.empty(0, dtype=dtype)
        starboard = (starboard_rows[ping_index] if ping_index < len(starboard_rows) else np.empty(0, dtype=dtype))
        if port.size:
            waterfall[ping_index, port_width - port.size : port_width] = np.fliplr(port[np.newaxis, :])[0]
        if starboard.size:
            waterfall[ping_index, port_width : port_width + starboard.size] = starboard
    return waterfall


def parse_xtf(input_path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Parse an XTF file into a waterfall and ping-aligned navigation telemetry."""
    try:
        from pyxtf import XTFHeaderType, xtf_read
    except ImportError as exc:
        raise RuntimeError("pyxtf is required to ingest XTF files; install requirements.txt") from exc

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"XTF input does not exist: {path}")
    with path.open("rb") as handle:
        prefix = handle.read(1024)
    if path.stat().st_size < 1024 or LFS_POINTER_MARKER in prefix:
        raise ValueError(
            "The file is an unresolved Git LFS pointer (~130 bytes). Please run 'git lfs pull' or download the actual binary payload."
        )

    parsed = xtf_read(str(path))
    if isinstance(parsed, tuple) and len(parsed) >= 2:
        file_header, packets = parsed[0], parsed[1]
    else:
        file_header = None
        packets = parsed
    sonar_type = getattr(XTFHeaderType, "sonar", None)
    bathy_type = getattr(XTFHeaderType, "bathy", None)
    sonar_packets = _packet_candidates(packets, (sonar_type, bathy_type))
    if not sonar_packets:
        packet_types = list(packets.keys()) if isinstance(packets, dict) else []
        LOGGER.error("No valid ping arrays found in %s", path)
        LOGGER.error("File header: %s", _header_summary(file_header))
        LOGGER.error("Available packet types: %s", packet_types)
        raise ValueError(
            f"No valid ping arrays found in {path}. "
            f"Header: {_header_summary(file_header)}; packet types: {packet_types}"
        )

    grouped: dict[int, dict[int, tuple[Any, np.ndarray]]] = {}
    telemetry: dict[str, list[float]] = {name: [] for name in (
        "SensorAltitude", "SensorHeading", "SensorPitch", "SensorRoll", "SensorXcoord", "SensorYcoord"
    )}
    telemetry_aliases = {
        "SensorAltitude": ("SensorAltitude", "SensorPrimaryAltitude", "SensorAuxAltitude"),
        "SensorHeading": ("SensorHeading",),
        "SensorPitch": ("SensorPitch",),
        "SensorRoll": ("SensorRoll",),
        "SensorXcoord": ("SensorXcoord", "SensorXcoordinate"),
        "SensorYcoord": ("SensorYcoord", "SensorYcoordinate"),
    }
    for packet in sonar_packets:
        ping_index = int(_field(packet, "PingNumber", _field(packet, "ping_number", len(grouped))))
        channel = _channel_number(packet)
        channels = (channel,) if channel in (0, 1) else (0, 1)
        records = [(candidate, _channel_samples(packet, candidate)) for candidate in channels]
        records = [(candidate, samples) for candidate, samples in records if samples.size]
        if not records:
            LOGGER.warning("Skipping packet with no usable channel samples: %s", type(packet).__name__)
            continue
        ping = grouped.setdefault(ping_index, {})
        for candidate, samples in records:
            ping[candidate] = (packet, samples)
        if len(ping) == len(records):
            for name in telemetry:
                value = next((_field(packet, alias) for alias in telemetry_aliases[name] if _field(packet, alias) is not None), np.nan)
                try:
                    telemetry[name].append(float(value))
                except (TypeError, ValueError):
                    telemetry[name].append(np.nan)

    ordered_pings = sorted(grouped)
    port_rows = [grouped[index][0][1] if 0 in grouped[index] else np.empty(0, dtype=np.float32) for index in ordered_pings]
    starboard_rows = [grouped[index][1][1] if 1 in grouped[index] else np.empty(0, dtype=np.float32) for index in ordered_pings]
    if any(row.size == 0 for row in port_rows + starboard_rows):
        LOGGER.warning("Some pings are missing channel 0 or 1; missing rows are zero-padded")
    return _pad_and_stitch(port_rows, starboard_rows), {key: np.asarray(values) for key, values in telemetry.items()}


def main() -> None:
    """Run XTF ingestion from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to an XTF file")
    parser.add_argument("--visualize", action="store_true", help="Save a raw waterfall preview beside the input")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    waterfall, telemetry = parse_xtf(args.input)
    LOGGER.info("Parsed waterfall shape=%s and telemetry fields=%s", waterfall.shape, list(telemetry))
    if args.visualize:
        import matplotlib.pyplot as plt
        output_path = args.input.with_suffix(".waterfall.png")
        plt.imsave(output_path, waterfall, cmap="gray", format="png")
        LOGGER.info("Saved preview to %s", output_path)


if __name__ == "__main__":
    main()
