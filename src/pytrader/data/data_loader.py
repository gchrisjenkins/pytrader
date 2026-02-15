"""Utilities for loading, aligning, and segmenting OHLC(V) market data.

The helpers in this module present a consistent interface for reading CSV
candles from disk, aligning multiple symbols on a shared timeline, and returning
metadata about the contiguous segments that are suitable for downstream
calibration or simulation workflows.
"""

import os
import glob
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OhlcSegment:
    """Metadata describing a contiguous run of aligned candles.

    Attributes:
        start: Timestamp of the first candle in the segment.
        end: Timestamp of the final candle in the segment.
        length: Number of candles spanned by the segment.
        sl: Slice pointing at the rows inside the aligned master frame.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    length: int
    sl: slice  # slice in the aligned DataFrame


@dataclass
class AssembledOhlcData:
    """Container for the assembled price data and accompanying metadata.

    Attributes:
        df: Price frame indexed by UTC timestamps (tz-naive). Columns follow the
            pattern ``open_<symbol>``, ``high_<symbol>``, ``low_<symbol>``,
            ``close_<symbol>`` and optionally ``volume_<symbol>``.
        selected_segments: Ordered list of the segments that were stitched
            together.
        total_segments: Count of all contiguous segments detected during
            alignment.
        eligible_segments: Number of segments that satisfied the selection
            policy.
        avg_segment_length: Average length (in bars) across selected segments.
        longest_segment_length: Length of the longest selected segment.
        total_assembled_length: Total number of rows in ``df`` after assembly.
        time_interval: A timedelta that represents the spacing between candles.
    """

    df: pd.DataFrame
    selected_segments: list[OhlcSegment]
    total_segments: int
    eligible_segments: int
    avg_segment_length: float
    longest_segment_length: int
    total_assembled_length: int
    time_interval: timedelta


def _suffix_to_timedelta(suffix: str) -> timedelta:
    """Convert an interval suffix such as ``"1m"`` into a ``timedelta``."""

    s = suffix.strip().lower()
    if s.endswith("m"):
        return timedelta(minutes=int(s[:-1]))
    if s.endswith("h"):
        return timedelta(hours=int(s[:-1]))
    if s.endswith("d"):
        return timedelta(days=int(s[:-1]))
    raise ValueError(f"Unsupported interval suffix: {suffix!r}")


def _build_aligned_prices(
    frames: dict[str, pd.DataFrame],
    interval_suffix: str
) -> tuple[pd.DataFrame, list[OhlcSegment], timedelta]:
    """Align per-symbol frames on the shared timestamp intersection.

    The routine intersects the available indexes, detects gaps larger than the
    requested interval, and returns a consolidated frame containing prefixed
    OHLC(V) columns per symbol along with the segment metadata.

    Args:
        frames: Mapping of symbol -> OHLC(V) frame (index must be timestamps).
        interval_suffix: Interval string (e.g. ``"1m"``) used to determine the
            gap threshold.

    Returns:
        A tuple ``(aligned, segments, interval)`` where ``aligned`` is the
        combined price frame, ``segments`` is the ordered list of
        :class:`OhlcSegment` objects, and ``interval`` is the candle spacing.

    Raises:
        ValueError: If no overlapping timestamps exist across the inputs.
    """

    time_interval = _suffix_to_timedelta(interval_suffix)

    # intersection of indices
    common_index = None
    for _, df in frames.items():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    if common_index is None or len(common_index) == 0:
        raise ValueError("No overlapping timestamps across assets.")
    common_index = common_index.sort_values()

    # find breakpoints (gaps > interval)
    gaps = (common_index[1:] - common_index[:-1]).to_series()
    boundaries = [0] + list(np.where(gaps > time_interval)[0] + 1) + [len(common_index)]

    segments: list[OhlcSegment] = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1]
        seg_len = b - a
        if seg_len > 0:
            segments.append(OhlcSegment(common_index[a], common_index[b - 1], seg_len, slice(a, b)))

    # assemble columns per symbol; include volume_<sym> only if present in source
    aligned_cols: dict[str, pd.Series] = {}
    for sym, df in frames.items():
        re = df.reindex(common_index)
        for field in ("open", "high", "low", "close"):
            aligned_cols[f"{field}_{sym}"] = pd.to_numeric(re[field], errors="coerce")
        if "volume" in df.columns:
            aligned_cols[f"volume_{sym}"] = pd.to_numeric(re["volume"], errors="coerce")

    aligned = pd.DataFrame(aligned_cols, index=common_index)
    return aligned, segments, time_interval


def _select_segments(
    segments: list[OhlcSegment],
    *,
    max_duration: int | None,
    min_segment_duration: int | None,
    allow_multi_segments: bool = True
) -> list[OhlcSegment]:
    """Choose which contiguous segments should be stitched together.

    Args:
        segments: Candidate segments produced by :func:`_build_aligned_prices`.
        max_duration: Optional cap on the total number of bars to keep across
            the selected segments. ``None`` keeps every eligible bar.
        min_segment_duration: Minimum acceptable segment length. When ``None``
            the function derives a heuristic threshold based on the data.
        allow_multi_segments: If ``False`` the single longest segment is
            returned; otherwise multiple segments may be combined.

    Returns:
        Ordered list of segments that meet the selection policy.
    """

    if not segments:
        return []
    total = sum(s.length for s in segments)
    longest = max(s.length for s in segments)

    if min_segment_duration is None:
        # Heuristic: target ~20% of the available data while clamping to
        # practical bounds so we favour longer, cleaner stretches when possible.
        base = max(400, int(0.2 * (max_duration or total)))
        msd = int(np.clip(base, 200, 5000))
        min_segment_duration = min(msd, longest)

    eligible = [s for s in segments if s.length >= min_segment_duration] or [max(segments, key=lambda s: s.length)]
    if not allow_multi_segments:
        return [max(eligible, key=lambda s: s.length)]

    if max_duration is None:
        return sorted(eligible, key=lambda s: s.start)

    eligible_sorted = sorted(eligible, key=lambda s: s.length, reverse=True)
    picked, acc = [], 0
    for seg in eligible_sorted:
        picked.append(seg)
        acc += seg.length
        if acc >= max_duration:
            break
    return sorted(picked, key=lambda s: s.start)


def _concat_segments(df: pd.DataFrame, picked: list[OhlcSegment]) -> pd.DataFrame:
    """Concatenate selected segments from the aligned frame and drop duplicates."""

    if not picked:
        return df.iloc[0:0]
    parts = [df.iloc[s.sl] for s in picked]
    out = pd.concat(parts, axis=0)
    out = out[~out.index.duplicated(keep="first")]
    return out


def _truncate_to_max_duration(df: pd.DataFrame, picked: list[OhlcSegment], max_duration: int | None) -> pd.DataFrame:
    """Trim the assembled frame so it does not exceed ``max_duration`` rows."""

    if max_duration is None:
        return df
    total = sum(s.length for s in picked)
    if total <= max_duration:
        return df
    excess = total - max_duration
    if excess <= 0:
        return df
    cut = df.index[-excess:]
    return df.drop(index=cut)


def load_historical_market_data(data_dir: str, symbol: str, candle_duration: str) -> pd.DataFrame:
    """Load a symbol's OHLC(V) candles from CSV snapshots on disk.

    The loader expects files matching
    ``<data_dir>/<symbol>/**/*_{symbol}_{candle_duration}.csv`` and requires the
    columns ``timestamp`` (milliseconds), ``open``, ``high``, ``low`` and
    ``close``. If a ``volume`` column is present it is retained as well.

    Args:
        data_dir: Root directory that contains per-symbol sub-folders.
        symbol: Trading symbol whose candles should be loaded (e.g. ``"BTCUSDT"``).
        candle_duration: Interval suffix such as ``"1m"`` or ``"1h"``.

    Returns:
        A Pandas ``DataFrame`` indexed by UTC timestamps (tz-naive) with lower
        cased column names.

    Raises:
        FileNotFoundError: If no CSV files match the expected pattern.
        ValueError: If a CSV file is missing required columns.
    """

    pattern = os.path.join(data_dir, symbol, "**", f"*_{symbol}_{candle_duration}.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No files for {symbol} at interval '{candle_duration}'. Pattern: {pattern}")

    parts = []
    for fp in files:
        df = pd.read_csv(fp)
        required = {"timestamp", "open", "high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{fp} missing required columns {sorted(required)}; found {df.columns.tolist()}"
            )
        cols = ["timestamp", "open", "high", "low", "close"]
        if "volume" in df.columns:
            cols.append("volume")
        parts.append(df[cols].copy())

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    # ms -> UTC -> tz-naive (to match your current behavior)
    dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.drop(columns=["timestamp"])
    df.index = dt
    df = df[~df.index.duplicated(keep="first")]
    df = df.rename(columns={c: c.lower() for c in df.columns})
    return df


def load_and_assemble(
    data_dir: str,
    symbols: list[str],
    *,
    candle_duration: str,                # e.g., "1m"
    start_date: str | None = None,       # ISO date (inclusive)
    end_date: str | None = None,         # ISO date (exclusive)
    max_candles: int | None = None,
    min_segment_length: int | None = None,
    allow_multi_segments: bool = True,
) -> AssembledOhlcData:
    """Orchestrate the full data loading workflow for multiple symbols.

    Args:
        data_dir: Root path that contains the per-symbol CSV directories.
        symbols: Ordered collection of symbols to align and stitch together.
        candle_duration: Interval string such as ``"1m"``.
        start_date: Optional inclusive ISO datetime filter.
        end_date: Optional exclusive ISO datetime filter.
        max_candles: Optional cap on the number of rows kept during alignment
            and assembly.
        min_segment_length: Minimum contiguous length (in bars) a segment must
            have to be considered for assembly.
        allow_multi_segments: When ``False`` only the single best segment is
            retained.

    Returns:
        An :class:`AssembledOhlcData` instance containing the fused price frame
        and metadata about the segments that were used.

    Raises:
        ValueError: If any symbol has no data after applying date filters.
    """

    if not symbols:
        raise ValueError("At least one symbol must be provided for assembly.")

    seen_symbols: set[str] = set()
    ordered_symbols: list[str] = []
    for sym in symbols:
        if sym not in seen_symbols:
            ordered_symbols.append(sym)
            seen_symbols.add(sym)

    frames: dict[str, pd.DataFrame] = {}

    for sym in ordered_symbols:
        df = load_historical_market_data(data_dir, sym, candle_duration)
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date, utc=True).tz_localize(None)]
        if end_date:
            df = df[df.index < pd.to_datetime(end_date, utc=True).tz_localize(None)]
        if df.empty:
            raise ValueError(f"No data for {sym} after applying date filters.")
        frames[sym] = df

    aligned, segments, time_interval = _build_aligned_prices(frames, candle_duration)

    if max_candles is not None:
        aligned = aligned.tail(int(max_candles))

    picked = _select_segments(
        segments,
        max_duration=None,
        min_segment_duration=min_segment_length,
        allow_multi_segments=allow_multi_segments,
    )
    assembled = _concat_segments(aligned, picked)
    if max_candles is not None:
        assembled = assembled.tail(int(max_candles))

    seg_lengths = [s.length for s in picked]
    meta = AssembledOhlcData(
        df=assembled,
        selected_segments=picked,
        total_segments=len(segments),
        eligible_segments=len(picked),
        avg_segment_length=float(np.mean(seg_lengths)) if seg_lengths else 0.0,
        longest_segment_length=max(seg_lengths) if seg_lengths else 0,
        total_assembled_length=len(assembled),
        time_interval=time_interval,
    )

    return meta
