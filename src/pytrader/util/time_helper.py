"""Utility helpers for working with canonical time-unit labels."""

import math
import re
from datetime import timedelta


TIME_UNIT_BASE_SECONDS: dict[str, float] = {
    "s": 1.0,
    "m": 60.0,
    "h": 3_600.0,
    "d": 86_400.0,
    "w": 7 * 86_400.0,
    "M": 30.4375 * 86_400.0,
    "y": 365.25 * 86_400.0,
}


def normalize_time_unit_label(value: str) -> str:
    """Return the canonical lowercase label for a supported time unit."""

    if not isinstance(value, str):
        raise TypeError(f"'time_unit' must be a string value, but found: {type(value)}")

    match = re.fullmatch(r"(\d+)\s*([smhdwMyY])", value.strip())
    if not match:
        raise ValueError(
            "'time_unit' must be specified as '<integer><unit>' (e.g., '1m', '5m', '1h', '1M', '1y')"
        )

    magnitude = int(match.group(1))
    if magnitude <= 0:
        raise ValueError("'time_unit' magnitude must be positive")

    unit_token = match.group(2)
    unit = "M" if unit_token == "M" else unit_token.lower()

    if unit not in TIME_UNIT_BASE_SECONDS:
        raise ValueError(
            f"Unsupported 'time_unit' unit '{unit_token}'. Supported units: {sorted(TIME_UNIT_BASE_SECONDS.keys())}"
        )

    return f"{magnitude}{unit}"


def time_unit_label_to_timedelta(label: str) -> timedelta:
    """Convert a canonical time-unit label into a timedelta."""

    canonical = normalize_time_unit_label(label)
    match = re.fullmatch(r"(\d+)([smhdwMy])", canonical)
    if not match:
        raise ValueError(f"Invalid canonical time unit label: {canonical}")

    value = int(match.group(1))
    unit = match.group(2)
    seconds = value * TIME_UNIT_BASE_SECONDS[unit]
    return timedelta(seconds=seconds)


def timedelta_to_time_unit_label(delta: timedelta) -> str:
    """Map a timedelta to the nearest supported canonical label."""

    if not isinstance(delta, timedelta):
        raise TypeError(f"Expected 'timedelta', but found: {type(delta)}")

    seconds = delta.total_seconds()
    if seconds <= 0:
        raise ValueError("'time_unit' timedelta must be positive")

    for unit, base_seconds in TIME_UNIT_BASE_SECONDS.items():
        ratio = seconds / base_seconds
        magnitude = round(ratio)
        if math.isclose(ratio, magnitude, rel_tol=1e-9, abs_tol=1e-9) and magnitude > 0:
            return f"{magnitude}{unit}"

    raise ValueError("Unable to map the provided 'time_unit' timedelta to a supported canonical label")


__all__ = [
    "TIME_UNIT_BASE_SECONDS",
    "normalize_time_unit_label",
    "time_unit_label_to_timedelta",
    "timedelta_to_time_unit_label",
]
