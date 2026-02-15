import glob
import os
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from tqdm import tqdm

DEFAULT_FOLDER = "./market_data"

SOURCE = "binance"
BASE_URL = "https://api.binance.com"
KLINES_EP = "/api/v3/klines"

# Force string intervals that match Binance Spot REST docs (case-sensitive).
# Supported: 1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
DEFAULT_INTERVAL = "1m"

# Binance allows up to 1000 klines per request on /api/v3/klines
MAX_LIMIT = 1000

# Exact allowed intervals (case-sensitive)
_ALLOWED_INTERVALS = {
    "1s", "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _normalize_interval_for_api(x: str) -> str:
    """
    Normalize a user-provided interval to Binance's canonical, case-sensitive form.
    - Accepts strings like "1m", "1M", "1H", "15m", etc.
    - Rejects integers (e.g., 1) and unsupported combos.
    Rules:
      * 'M' (uppercase) means month and MUST remain uppercase.
      * s,m,h,d,w are lowercased.
    """
    if not isinstance(x, str):
        raise ValueError(f"Interval must be a Binance interval string, got {type(x)}")
    s = x.strip()
    if s.isdigit():
        raise ValueError(f"Integer shorthand like '{s}' is not supported. Use '1m', '1h', '1M', etc.")

    m = re.fullmatch(r"(\d+)\s*([smhdwM])", s, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unsupported interval format: {x}")

    num = int(m.group(1))
    unit_in = m.group(2)
    # Canonicalize unit: keep 'M' uppercase for months; others lowercase.
    unit = 'M' if unit_in == 'M' else unit_in.lower()
    canonical = f"{num}{unit}"

    if canonical not in _ALLOWED_INTERVALS:
        raise ValueError(f"Unsupported interval: {x}. Allowed: {sorted(_ALLOWED_INTERVALS)}")
    return canonical


def _interval_to_suffix(x: str) -> str:
    """Use the canonical Binance string (case-sensitive) in filenames."""
    return _normalize_interval_for_api(x)


def _interval_ms(interval: str) -> int:
    """
    Return the interval length in milliseconds for fixed-length units
    (s, m, h, d, w). Month ('1M') is variable; handled separately.
    """
    s = _normalize_interval_for_api(interval)
    unit = s[-1]
    val = int(s[:-1])

    if unit == "s":
        return val * 1_000
    if unit == "m":
        return val * 60_000
    if unit == "h":
        return val * 3_600_000
    if unit == "d":
        return val * 86_400_000
    if unit == "w":
        return val * 7 * 86_400_000
    if unit == "M":
        raise ValueError("Use _next_cursor() for 'M' intervals; month length varies.")
    raise ValueError(f"Unsupported interval: {interval}")


def _next_month_open_ms(open_ms: int) -> int:
    """Given a monthly bar open (ms), compute the next month’s open (ms)."""
    dt = datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc)
    year = dt.year + (1 if dt.month == 12 else 0)
    month = 1 if dt.month == 12 else dt.month + 1
    next_dt = dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
    return int(next_dt.timestamp() * 1000)


def _next_cursor(last_open_ms: int, interval: str) -> int:
    """
    Compute the next request cursor after a page of klines, based on interval.
    For '1M', compute next calendar month open; otherwise add fixed step ms.
    """
    s = _normalize_interval_for_api(interval)
    if s.endswith("M"):
        return _next_month_open_ms(last_open_ms)
    return last_open_ms + _interval_ms(s)


def _parse_date(s_or_dt, *, end_of_day=False) -> datetime:
    """
    Parse a date into a timezone-aware UTC datetime.
    Accepted inputs:
      - datetime (naive or aware): returned as UTC (converted if tz-aware).
      - str in one of:
          * "ddMMyyyy"        (e.g., "01062024" → 2024-06-01 00:00:00 UTC)
          * "yyyy-mm-dd"      (e.g., "2024-06-01")
          * "yyyy/mm/dd"      (e.g., "2024/06/01")
          * ISO-8601 date or datetime (e.g., "2024-06-01T12:34:56Z")
    If end_of_day=True and the input is a date (no time), the result is set to 23:59:59.
    Otherwise, time defaults to 00:00:00 when only a date is provided.
    """
    if s_or_dt is None:
        return None
    if isinstance(s_or_dt, datetime):
        dt = s_or_dt.astimezone(timezone.utc) if s_or_dt.tzinfo else s_or_dt.replace(tzinfo=timezone.utc)
        return dt

    s = str(s_or_dt).strip()
    dt = None

    # ddMMyyyy
    if len(s) == 8 and s.isdigit():
        dt = datetime.strptime(s, "%d%m%Y").replace(tzinfo=timezone.utc)
    else:
        # try common date formats
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                pass
        if dt is None:
            # last resort: ISO-8601 parsing
            try:
                s_iso = s.replace("Z", "+00:00")
                dt = datetime.fromisoformat(s_iso)
                dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                raise ValueError(f"Unrecognized date format: {s_or_dt}")

    if end_of_day:
        bare_date = (len(s) == 8 and s.isdigit()) or s in (
            dt.strftime("%Y-%m-%d"),
            dt.strftime("%Y/%m/%d"),
        )
        if bare_date:
            dt = dt.replace(hour=23, minute=59, second=59, microsecond=0, tzinfo=timezone.utc)
    return dt


def _to_millis(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _from_millis(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _make_weekly_filename(root_dir: str, source: str, symbol: str, interval_suffix: str, file_start: datetime) -> str:
    """
    LAYOUT:
      {root}/{source}/{symbol}/{YYYY}/{MM}/{ddMMyyyy}_{symbol}_{suffix}.csv
    file_start is a *naive UTC* datetime indicating the start date for the file.
    """
    yyyy = file_start.strftime("%Y")
    mm = file_start.strftime("%m")
    date_str = file_start.strftime("%d%m%Y")
    symbol_folder = os.path.join(root_dir, source, symbol, yyyy, mm)
    os.makedirs(symbol_folder, exist_ok=True)
    return os.path.join(symbol_folder, f"{date_str}_{symbol}_{interval_suffix}.csv")


def _existing_latest_timestamp_ms(root_dir: str, source: str, symbol: str, interval_suffix: str) -> int | None:
    """
    Scan existing files for this symbol+suffix and return the latest timestamp (ms),
    or None if no files exist.
    """
    pattern = os.path.join(root_dir, source, symbol, "**", f"*_{symbol}_{interval_suffix}.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        return None
    latest_ts = None
    for fp in files:
        try:
            df = pd.read_csv(fp, usecols=["timestamp"])
            if df.empty:
                continue
            ts = int(df["timestamp"].max())
            latest_ts = ts if latest_ts is None else max(latest_ts, ts)
        except Exception:
            continue
    return latest_ts


def _existing_min_max_timestamp_ms(root_dir: str, source: str, symbol: str, interval_suffix: str) -> tuple[int | None, int | None]:
    """
    Scan all saved CSVs for this symbol+suffix and return (min_ts, max_ts) in ms.
    Returns (None, None) if nothing found.
    """
    pattern = os.path.join(root_dir, source, symbol, "**", f"*_{symbol}_{interval_suffix}.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    have_min = None
    have_max = None
    for fp in files:
        try:
            df = pd.read_csv(fp, usecols=["timestamp"])
            if df.empty:
                continue
            fmin = int(df["timestamp"].min())
            fmax = int(df["timestamp"].max())
            have_min = fmin if have_min is None else min(have_min, fmin)
            have_max = fmax if have_max is None else max(have_max, fmax)
        except Exception:
            continue
    return have_min, have_max


# ---------------------------------------------------------------------
# Progress bar (console)
# ---------------------------------------------------------------------

def _progress_init(total_ms: int) -> dict:
    """
    Initialize progress tracking state.
    Returns a dict with total_ms and last_shown (percentage).
    """
    return {"total_ms": max(total_ms, 1), "last_shown": -1}


def _progress_update(state: dict, current_ms: int, prefix: str = "Progress", width: int = 30):
    """
    Update the console progress bar if percentage has increased by at least 1%.
    """
    total = state["total_ms"]
    pct = int(max(0, min(100, (current_ms * 100) // total)))
    if pct <= state["last_shown"]:
        return
    state["last_shown"] = pct

    filled = int((pct * width) // 100)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix} [{bar}] {pct:3d}%", end="", flush=True)


def _progress_finish(state: dict, prefix: str = "Progress", width: int = 30):
    """
    Force 100% and print a newline.
    """
    _progress_update(state, state["total_ms"], prefix=prefix, width=width)
    print()  # newline


# ---------------------------------------------------------------------
# HTTP & pagination
# ---------------------------------------------------------------------

def _explain_binance_error(resp: requests.Response) -> str:
    """
    Build a helpful, single-line error string for Binance HTTP errors.
    Tries to include Binance's JSON {code,msg} when available.
    """
    reason = resp.reason or ""
    url = resp.url
    status = resp.status_code
    retry_after = resp.headers.get("Retry-After")
    detail = None
    try:
        j = resp.json()
        if isinstance(j, dict):
            code = j.get("code")
            msg = j.get("msg")
            if code is not None or msg is not None:
                detail = f"code={code}, msg={msg}"
    except Exception:
        pass

    extras = []
    if status == 451:
        extras.append("Unavailable For Legal Reasons (possible geo restrictions).")
    if status == 418:
        extras.append("IP banned per Binance (418 I'm a teapot).")
    if retry_after:
        extras.append(f"Retry-After={retry_after}s")

    parts = [f"HTTP {status} {reason} for {url}"]
    if detail:
        parts.append(f"({detail})")
    if extras:
        parts.append(" ".join(extras))
    return " ".join(parts)


def _fetch_klines_page(session: requests.Session, symbol: str, interval: str, start_ms: int, end_ms: int | None, limit: int) -> tuple[list, dict]:
    """
    Fetch a single page of klines. Returns (rows, headers).
    """
    interval_api = _normalize_interval_for_api(interval)
    params = {
        "symbol": symbol,
        "interval": interval_api,  # canonical, case-sensitive
        "startTime": int(start_ms),
        "limit": min(int(limit), MAX_LIMIT),
    }
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    NON_RETRYABLE = {400, 401, 403, 404, 409, 418, 422, 451}
    MAX_ATTEMPTS = 6

    for attempt in range(MAX_ATTEMPTS):
        try:
            r = session.get(BASE_URL + KLINES_EP, params=params, timeout=30)

            # --- Rate limit: retry with backoff or server-provided Retry-After ---
            if r.status_code == 429:
                retry_after_hdr = r.headers.get("Retry-After")
                try:
                    sleep_s = int(retry_after_hdr) if retry_after_hdr is not None else 2 ** attempt
                except ValueError:
                    sleep_s = 2 ** attempt
                print(
                    f"[Rate limit] HTTP 429 from Binance (attempt {attempt + 1}/{MAX_ATTEMPTS}). "
                    f"Sleeping {sleep_s}s before retry…"
                    )
                time.sleep(sleep_s)
                continue

            # --- Non-retryable client errors: fail fast with clear message ---
            if r.status_code in NON_RETRYABLE:
                msg = _explain_binance_error(r)
                raise requests.HTTPError(msg, response=r)

            # --- Other 4xx/5xx: retry with capped backoff; give up after last attempt ---
            if r.status_code >= 400:
                msg = _explain_binance_error(r)
                if attempt < MAX_ATTEMPTS - 1:
                    sleep_s = min(2 ** attempt, 30)
                    print(f"{msg} Retrying in {sleep_s}s (attempt {attempt + 2}/{MAX_ATTEMPTS})…")
                    time.sleep(sleep_s)
                    continue
                else:
                    # Final attempt exhausted — raise with context
                    raise requests.HTTPError(msg, response=r)

            # --- Success ---
            rows = r.json()  # list[list]
            return rows, r.headers

        except (requests.ConnectionError, requests.Timeout) as e:
            # Transient network issues — retry with capped backoff
            if attempt < MAX_ATTEMPTS - 1:
                sleep_s = min(2 ** attempt, 30)
                print(
                    f"[Network] {e.__class__.__name__} on attempt {attempt + 1}/{MAX_ATTEMPTS}. "
                    f"Sleeping {sleep_s}s before retry…"
                    )
                time.sleep(sleep_s)
                continue
            else:
                # Out of retries — bubble up
                raise

    return [], {}


def _fetch_klines_range(symbol: str, interval=DEFAULT_INTERVAL, start_dt: datetime | None = None, end_dt: datetime | None = None, limit: int = MAX_LIMIT) -> pd.DataFrame:
    """
    Paginate Binance klines from start_dt to end_dt (UTC). If end_dt is None, fetch up to "now".
    Returns a DataFrame with columns: timestamp, open, high, low, close, volume, turnover, num_trades.
    """
    interval_api = _normalize_interval_for_api(interval)
    if start_dt is None:
        start_dt = datetime.now(tz=timezone.utc) - timedelta(days=30)
    if end_dt is None:
        end_dt = datetime.now(tz=timezone.utc)

    start_ms = _to_millis(start_dt)
    end_ms = _to_millis(end_dt)

    session = requests.Session()
    out_rows: list[list] = []

    # ----- tqdm progress setup -----
    total_ms = max(1, end_ms - start_ms)
    bar = tqdm(
        total=total_ms,
        unit="ms",
        unit_scale=True,
        desc=f"Fetching {symbol} {_normalize_interval_for_api(interval)}",
        leave=True,
    )
    last_covered = 0

    cursor = start_ms
    while True:
        if cursor > end_ms:
            break
        page, headers = _fetch_klines_page(session, symbol, interval_api, cursor, end_ms, limit)
        if not page:
            break
        out_rows.extend(page)

        # Advance cursor from the last open time according to the interval.
        last_open_ms = int(page[-1][0])  # open_time
        cursor = _next_cursor(last_open_ms, interval_api)

        # Progress update based on time covered
        covered = max(0, min(end_ms - start_ms, cursor - start_ms))
        delta = covered - last_covered
        if delta > 0:
            bar.update(delta)
            last_covered = covered

        # Gentle pacing to be nice (Binance has per-minute weights)
        time.sleep(0.1)

    # Ensure the bar completes and closes cleanly
    if last_covered < total_ms:
        bar.update(total_ms - last_covered)
    bar.close()

    if not out_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "turnover", "num_trades"])

    df = pd.DataFrame(out_rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])

    # Keep previous columns + num_trades
    out = df[["open_time","open","high","low","close","volume","quote_asset_volume","num_trades"]].rename(
        columns={"open_time":"timestamp", "quote_asset_volume":"turnover"}
    )

    # Cast numeric columns (prices/volumes remain float; num_trades as int)
    out["timestamp"] = out["timestamp"].astype("int64")
    out["num_trades"] = out["num_trades"].astype("int64")
    for c in ["open","high","low","close","volume","turnover"]:
        out[c] = out[c].astype(float)

    # De-dup & sort just in case
    out = out.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return out


# ---------------------------------------------------------------------
# Save (month → week sharding) + summary
# ---------------------------------------------------------------------

def _save_weekly_csv(df: pd.DataFrame, symbol: str, interval=DEFAULT_INTERVAL, root_dir=DEFAULT_FOLDER, overwrite=False):
    if df.empty:
        print("No new data to save.")
        return

    # Convert timestamps to timezone-aware UTC and also create naive UTC
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["dt_naive"] = df["datetime"].dt.tz_convert(None)

    # Month period (UTC-naive)
    df["month"] = df["dt_naive"].dt.to_period("M")

    # Sunday-based week starts (W-SAT means weeks ending Sat; start_time = Sunday)
    df["week_start"] = df["dt_naive"].dt.to_period("W-SAT").apply(lambda r: r.start_time)

    suffix = _interval_to_suffix(interval)

    # Process by month
    for month_key, month_df in df.groupby("month"):
        month_start = month_key.start_time   # naive UTC
        month_end_excl = (month_key + 1).start_time

        week_starts = sorted(month_df["week_start"].unique())

        for ws in week_starts:
            file_start = max(ws, month_start)
            file_end_excl = min(ws + timedelta(days=7), month_end_excl)

            week_slice = month_df[month_df["week_start"] == ws]
            mask = (week_slice["dt_naive"] >= file_start) & (week_slice["dt_naive"] < file_end_excl)
            out_df = week_slice.loc[mask].drop(columns=["datetime","dt_naive","month","week_start"])

            if out_df.empty:
                continue

            filename = _make_weekly_filename(root_dir, SOURCE, symbol, suffix, file_start)

            if os.path.exists(filename) and not overwrite:
                existing_df = pd.read_csv(filename)
                out_df = pd.concat([existing_df, out_df], ignore_index=True)\
                           .drop_duplicates(subset="timestamp")\
                           .sort_values("timestamp")

            out_df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")

    # ---------- Summary over ALL saved files ----------
    symbol_root = os.path.join(root_dir, SOURCE, symbol)
    all_files = []
    for root, _, files in os.walk(symbol_root):
        for f in files:
            if f.endswith(f"_{suffix}.csv"):
                all_files.append(os.path.join(root, f))
    if all_files:
        all_dfs = [pd.read_csv(fp) for fp in all_files]
        combined_df = pd.concat(all_dfs, ignore_index=True)\
                        .drop_duplicates(subset="timestamp")\
                        .sort_values("timestamp")
        _summarize_data(combined_df)
    else:
        _summarize_data(df.drop(columns=["datetime", "dt_naive", "month", "week_start"]))


def _summarize_data(df: pd.DataFrame):
    """Print summary of saved data."""
    if df.empty:
        print("No data in file.")
        return
    earliest = datetime.utcfromtimestamp(int(df["timestamp"].min()) / 1000)
    latest = datetime.utcfromtimestamp(int(df["timestamp"].max()) / 1000)
    total_candles = len(df)
    # Approximate coverage in days if interval is 1 minute
    total_days = total_candles / (60 * 24)
    contiguous_days = total_candles // (60 * 24)

    print("\n================ Data Summary ================")
    print(f"Earliest: {earliest}")
    print(f"Latest:   {latest}")
    print(f"Total candles: {total_candles} (~{total_days:.2f} days)")
    print(f"1-day intervals of contiguous candles: {contiguous_days}")
    print("==============================================\n")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def download(
    symbol: str,
    start_date,
    *,
    interval=DEFAULT_INTERVAL,
    root_dir=DEFAULT_FOLDER,
    overwrite=False,
    end_date=None,
    mode: str = "auto",  # "auto" | "backfill" | "forward"
):
    """
    Download BINANCE spot klines for a single symbol.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., "BTCUSDT").
    start_date : str | datetime
        Start of the requested window (UTC). Accepted formats:
          - "ddMMyyyy"         (e.g., "01062024" → 2024-06-01 00:00:00)
          - "yyyy-mm-dd"       (e.g., "2024-06-01")
          - "yyyy/mm/dd"       (e.g., "2024/06/01")
          - ISO-8601 datetime  (e.g., "2024-06-01T12:34:56Z")
          - datetime (naive or tz-aware). Naive is treated as UTC.
        If a date-only string is provided, time defaults to 00:00:00.
    interval : str, default "1m"
        Binance interval string (case-sensitive). Supported:
        {'1s','1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M'}.
        (Integer shorthand like 1 is not accepted.)
    root_dir : str, default "./market_data"
        Root folder for output files. Will create subfolders per {source}/{symbol}/{YYYY}/{MM}.
    overwrite : bool, default False
        If True, fetched data overwrites existing weekly CSVs (for intersecting files).
        If False, new data is merged and deduplicated with existing files.
    end_date : str | datetime | None, default None
        End of the requested window (UTC). Same accepted formats as start_date.
        If a date-only string is provided, it is interpreted as 23:59:59 that day.
        If None, the forward portion (if any) is open-ended (up to "now" per fetch call).
    mode : {"auto","backfill","forward"}, default "auto"
        - "auto": backfill if requested start is before existing earliest; also forward resume if needed.
        - "backfill": only plan the backfill portion (never forward in the same call).
        - "forward": only plan the forward portion (never backfill in the same call).

    Behavior
    --------
    - Backfill example:
        Have starts at 2024-06-01 00:00:00; start_date="01052024"
        => Fetches [2024-05-01 00:00:00, 2024-05-31 23:59:59.999]
    - Forward resume example:
        Have through 2024-06-10 12:34:00.000; start_date="01062024"
        => Resumes from 2024-06-10 12:34:00.001 onward.
    - No-op:
        If the requested window is fully covered, prints "Nothing to fetch".
    """
    print(f"Downloading {symbol} interval={interval} from {start_date} to folder: {root_dir}")

    one_ms = timedelta(milliseconds=1)
    suffix = _interval_to_suffix(interval)

    # Requested window parsing
    req_start = _parse_date(start_date, end_of_day=False)
    req_end = _parse_date(end_date, end_of_day=True) if end_date is not None else None

    # Existing coverage
    have_min_ms, have_max_ms = _existing_min_max_timestamp_ms(root_dir, SOURCE, symbol, suffix)
    have_min = _from_millis(have_min_ms) if have_min_ms is not None else None
    have_max = _from_millis(have_max_ms) if have_max_ms is not None else None

    # Overwrite: fetch the requested window directly
    if overwrite:
        df = _fetch_klines_range(symbol, interval=interval, start_dt=req_start, end_dt=req_end)
        _save_weekly_csv(df, symbol, interval=interval, root_dir=root_dir, overwrite=True)
        return

    # Plan ranges
    planned_ranges: list[tuple[datetime, datetime | None]] = []

    if have_min is None or have_max is None:
        # No data yet → straight fetch from requested window
        planned_ranges.append((req_start, req_end))
    else:
        now_utc = datetime.now(tz=timezone.utc)
        effective_req_end = req_end or now_utc

        # Fully covered?
        if req_start >= have_min and effective_req_end <= have_max:
            print("Nothing to fetch")
            return

        # Backfill portion
        if req_start < have_min and mode in ("auto", "backfill"):
            backfill_end = min(effective_req_end, have_min - one_ms)
            if backfill_end >= req_start:
                planned_ranges.append((req_start, backfill_end))

        # Forward portion
        if (req_end is None or req_end > have_max) and mode in ("auto", "forward"):
            fwd_start = max(req_start, have_max + one_ms)
            fwd_end = req_end  # may be None (open-ended)
            if fwd_end is None or fwd_end >= fwd_start:
                planned_ranges.append((fwd_start, fwd_end))

    # Sort/dedupe
    planned_ranges = sorted(set(planned_ranges), key=lambda r: (r[0], r[1] or datetime.max.replace(tzinfo=timezone.utc)))

    if not planned_ranges:
        print("Nothing to fetch")
        return

    # Execute
    for i, (start_dt, end_dt) in enumerate(planned_ranges, 1):
        if end_dt is not None and end_dt < start_dt:
            continue
        label_end = end_dt.isoformat() if end_dt else "now"
        print(f"[{i}/{len(planned_ranges)}] Fetching {symbol}: {start_dt.isoformat()} → {label_end}")
        df = _fetch_klines_range(symbol, interval=interval, start_dt=start_dt, end_dt=end_dt)
        _save_weekly_csv(df, symbol, interval=interval, root_dir=root_dir, overwrite=False)


def download_multiple(
    symbols: list[str],
    start_date,
    *,
    interval=DEFAULT_INTERVAL,
    root_dir=DEFAULT_FOLDER,
    overwrite=False,
    end_date=None,
    mode: str = "auto",
):
    """
    Batch version of download(). Accepts the same date formats and mode.
    """
    for symbol in symbols:
        try:
            download(
                symbol, start_date, interval=interval, root_dir=root_dir, overwrite=overwrite, end_date=end_date,
                mode=mode
            )
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
