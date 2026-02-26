#!/usr/bin/env python3

# Analyze MKV implied frame timing against FT timestamps.
#
# Supports:
# - Single-file/default mode (TestData by default)
# - Batch mode across overlapping MKV/FT days with multiprocessing
# - Multiple diagnostic plots and filtering of noisy curves

from __future__ import annotations

import argparse
import contextlib
import io
import pickle
import random
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


OPTIMIZED_CURVE_STD_THRESHOLD_MS = 40.0


# Removed per-worker cache _WORKER_FT_TIMES. FT times are now passed directly to tasks.


FT_FILENAME_PATTERN = re.compile(
    r"^FT_(?P<station>.+)_(?P<date>\d{8})_(?P<time>\d{6})\.bin$"
)

FT_FILENAME_NO_STATION_PATTERN = re.compile(
    r"^FT_(?P<date>\d{8})_(?P<time>\d{6})\.bin$"
)

MKV_FILENAME_PATTERN = re.compile(
    r"^(?P<station>.+?)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<micro>\d{6})(?:_.+)?\.(?i:mkv)$"
)


@dataclass(frozen=True)
class FTFileMeta:
    """Metadata extracted from an FT file path and filename timestamp."""

    path: Path
    station_id: str
    timestamp: datetime


@dataclass(frozen=True)
class MKVFileMeta:
    """Metadata extracted from an MKV file path and filename start time."""

    path: Path
    station_id: str
    start_utc: datetime


def parse_ft_filename(path: Path) -> FTFileMeta | None:
    """Parse an FT filename into ``FTFileMeta`` if it matches known patterns."""

    # Preferred FT naming pattern with station id.
    match = FT_FILENAME_PATTERN.match(path.name)
    if match:
        station_id = match.group("station")
        timestamp_text = f"{match.group('date')}{match.group('time')}"
        timestamp = datetime.strptime(timestamp_text, "%Y%m%d%H%M%S")
        return FTFileMeta(path=path, station_id=station_id, timestamp=timestamp)

    # Backward-compatible FT naming pattern without station id.
    match_no_station = FT_FILENAME_NO_STATION_PATTERN.match(path.name)
    if not match_no_station:
        return None

    timestamp_text = f"{match_no_station.group('date')}{match_no_station.group('time')}"
    timestamp = datetime.strptime(timestamp_text, "%Y%m%d%H%M%S")
    return FTFileMeta(path=path, station_id="UNKNOWN", timestamp=timestamp)


def discover_ft_files(search_root: Path) -> list[FTFileMeta]:
    """Recursively discover FT files below ``search_root`` and sort by timestamp."""

    # Recursively find FT files and sort by parsed start timestamp.
    found: list[FTFileMeta] = []
    for path in search_root.rglob("FT_*.bin"):
        if not path.is_file():
            continue
        parsed = parse_ft_filename(path)
        if parsed is not None:
            found.append(parsed)

    found.sort(key=lambda item: (item.timestamp, str(item.path)))
    return found


def parse_mkv_start_from_filename(mkv_path: Path) -> tuple[str, datetime]:
    """Parse station id and start datetime from a supported MKV filename."""

    # Parse station and exact start timestamp from MKV filename.
    match = MKV_FILENAME_PATTERN.match(mkv_path.name)
    if not match:
        raise ValueError(
            "MKV filename does not match expected format "
            "Station_ID_YYYYMMDD_HHmmss_uuuuuu[optional_suffix].mkv"
        )

    station = match.group("station")
    dt_text = f"{match.group('date')}{match.group('time')}{match.group('micro')}"
    start_dt = datetime.strptime(dt_text, "%Y%m%d%H%M%S%f")
    return station, start_dt


def discover_mkv_files(search_root: Path) -> list[Path]:
    """Return all MKV file paths below ``search_root`` sorted lexicographically."""

    mkvs = [path for path in search_root.rglob("*.mkv") if path.is_file()]
    mkvs.sort()
    return mkvs


def discover_mkv_meta(search_root: Path) -> list[MKVFileMeta]:
    """Collect MKV metadata for files that match the expected filename pattern."""

    # Collect MKV files that match expected naming and store UTC start time.
    found: list[MKVFileMeta] = []
    for path in search_root.rglob("*.mkv"):
        if not path.is_file():
            continue
        try:
            station_id, start_dt = parse_mkv_start_from_filename(path)
        except ValueError:
            continue

        found.append(
            MKVFileMeta(
                path=path,
                station_id=station_id,
                start_utc=start_dt.replace(tzinfo=timezone.utc),
            )
        )

    found.sort(key=lambda item: (item.start_utc, str(item.path)))
    return found


def group_ft_by_day(ft_files: list[FTFileMeta]) -> dict[str, list[FTFileMeta]]:
    """Group FT files by calendar day key ``YYYYMMDD``."""

    # Group FT files by YYYYMMDD derived from FT filename timestamp.
    grouped: dict[str, list[FTFileMeta]] = {}
    for item in ft_files:
        day_key = item.timestamp.strftime("%Y%m%d")
        grouped.setdefault(day_key, []).append(item)
    return grouped


def group_mkv_by_day(mkv_files: list[MKVFileMeta]) -> dict[str, list[MKVFileMeta]]:
    """Group MKV files by calendar day key ``YYYYMMDD``."""

    # Group MKV files by YYYYMMDD derived from MKV filename timestamp.
    grouped: dict[str, list[MKVFileMeta]] = {}
    for item in mkv_files:
        day_key = item.start_utc.strftime("%Y%m%d")
        grouped.setdefault(day_key, []).append(item)
    return grouped


def run_ffprobe_frame_count(mkv_path: Path) -> int:
    """Return frame count for MKV using ffprobe with duration-based fallback."""

    # Try quick ffprobe first (no -count_frames).
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(mkv_path),
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        val = result.stdout.strip()
        if val.isdigit() and int(val) > 0:
            return int(val)

    # If nb_frames is missing or 0, try -count_frames which is slower but more reliable.
    command_deep = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(mkv_path),
    ]

    result = subprocess.run(command_deep, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        val = result.stdout.strip()
        if val.isdigit() and int(val) > 0:
            return int(val)

    # Fallback for cases where nb_read_frames/nb_frames is unavailable.
    fallback_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(mkv_path),
    ]
    fallback_result = subprocess.run(
        fallback_command, check=False, capture_output=True, text=True
    )
    if fallback_result.returncode == 0:
        lines = fallback_result.stdout.strip().splitlines()
        if len(lines) >= 1:
            duration = float(lines[0])
            fps = 25.0
            if len(lines) >= 2 and "/" in lines[1]:
                num, den = lines[1].split("/")
                if float(den) > 0:
                    fps = float(num) / float(den)
            
            frames = int(round(duration * fps))
            if frames > 0:
                return frames

    raise RuntimeError(f"Could not determine frame count for {mkv_path}")


def import_ftfile_read(rms_root: Path):
    """Import and return ``RMS.Formats.FTfile.read``, adding ``rms_root`` if needed."""

    try:
        from RMS.Formats import FTfile

        return FTfile.read
    except ModuleNotFoundError:
        sys.path.insert(0, str(rms_root))
        from RMS.Formats import FTfile

        return FTfile.read


def collect_ft_times_seconds(ft_files: list[FTFileMeta], rms_root: Path) -> np.ndarray:
    """Read all FT timestamp entries and return one sorted numpy array (seconds)."""

    # Read and merge all FT timestamps into one sorted vector for fast matching.
    read_ft = import_ftfile_read(rms_root)

    all_times: list[float] = []
    for ft_file in ft_files:
        ft = read_ft(str(ft_file.path.parent), ft_file.path.name)
        all_times.extend(timestamp for _, timestamp in ft.timestamps)

    if not all_times:
        raise RuntimeError("No timestamp entries were found in FT files")

    ft_seconds = np.array(all_times, dtype=np.float64)
    ft_seconds.sort()
    return ft_seconds


def nearest_offsets_seconds(
    implied_times: np.ndarray, ft_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match each implied time to nearest FT time and return nearest times + offsets."""

    # For each implied frame time, select the nearest FT timestamp.
    # Uses searchsorted for O(N log M) nearest-neighbor matching.
    insert_positions = np.searchsorted(ft_times, implied_times)

    right_indices = np.clip(insert_positions, 0, len(ft_times) - 1)
    left_indices = np.clip(insert_positions - 1, 0, len(ft_times) - 1)

    left_values = ft_times[left_indices]
    right_values = ft_times[right_indices]

    choose_right = np.abs(right_values - implied_times) < np.abs(left_values - implied_times)
    nearest_values = np.where(choose_right, right_values, left_values)
    offsets = nearest_values - implied_times
    return nearest_values, offsets


def implied_times_for_fps(start_timestamp: float, frame_indices: np.ndarray, fps: float) -> np.ndarray:
    """Compute implied frame Unix times from start timestamp and frame rate."""

    return start_timestamp + (frame_indices / fps)


def offset_deviation_from_first_for_fps(
    fps_array: np.ndarray,
    start_timestamp: float,
    frame_indices: np.ndarray,
    ft_times: np.ndarray,
) -> np.ndarray:
    """Objective values for FPS search: mean squared deviation from frame-0 offset."""

    # fps_array: (N,)
    # frame_indices: (M,)
    # implied_times: (N, M)
    implied_times = start_timestamp + (frame_indices[np.newaxis, :] / fps_array[:, np.newaxis])
    
    # ft_times is (K,)
    # searchsorted on (N, M)
    insert_positions = np.searchsorted(ft_times, implied_times)

    right_indices = np.clip(insert_positions, 0, len(ft_times) - 1)
    left_indices = np.clip(insert_positions - 1, 0, len(ft_times) - 1)

    left_values = ft_times[left_indices]
    right_values = ft_times[right_indices]

    choose_right = np.abs(right_values - implied_times) < np.abs(left_values - implied_times)
    nearest_values = np.where(choose_right, right_values, left_values)
    offsets = nearest_values - implied_times
    
    first_offsets = offsets[:, 0][:, np.newaxis]
    delta = offsets - first_offsets
    return np.mean(delta**2, axis=1)


def optimize_fps(
    initial_fps: float,
    start_timestamp: float,
    frame_indices_all: np.ndarray,
    ft_times: np.ndarray,
) -> float:
    """Optimize FPS with staged local search to flatten offsets around frame 0."""

    if frame_indices_all.size == 0:
        raise ValueError("No frame indices provided for FPS optimization")

    best_fps = initial_fps
    
    search_stages = [
        (0.5, 201),
        (0.05, 201),
        (0.001, 201),
        (0.0001, 201),
    ]

    for half_width, steps in search_stages:
        min_fps = max(0.1, best_fps - half_width)
        max_fps = best_fps + half_width
        candidates = np.linspace(min_fps, max_fps, num=steps, dtype=np.float64)
        
        errors = offset_deviation_from_first_for_fps(
            candidates, start_timestamp, frame_indices_all, ft_times
        )
        
        best_idx = np.argmin(errors)
        best_fps = float(candidates[best_idx])

    return best_fps


def summarize_offsets(offsets: np.ndarray) -> tuple[float, float, float, float]:
    """Return mean, median, min, max offsets in milliseconds."""

    mean_ms = float(np.mean(offsets) * 1000.0)
    median_ms = float(np.median(offsets) * 1000.0)
    min_ms = float(np.min(offsets) * 1000.0)
    max_ms = float(np.max(offsets) * 1000.0)
    return mean_ms, median_ms, min_ms, max_ms


def print_offset_summary(title: str, offsets: np.ndarray) -> None:
    """Print formatted summary statistics for an offset array."""

    # Convenience summary in milliseconds.
    mean_ms, median_ms, min_ms, max_ms = summarize_offsets(offsets)
    print(f"\n{title}")
    print(f"  mean [ms]:   {mean_ms:.6f}")
    print(f"  median [ms]: {median_ms:.6f}")
    print(f"  min [ms]:    {min_ms:.6f}")
    print(f"  max [ms]:    {max_ms:.6f}")


def anchor_error_metrics(offsets: np.ndarray) -> tuple[float, float]:
    """Return RMSE and max-abs deviation from first-frame offset (ms)."""

    first_offset = float(offsets[0])
    delta_from_first = offsets - first_offset
    rmse_ms = float(np.sqrt(np.mean(delta_from_first**2)) * 1000.0)
    max_abs_ms = float(np.max(np.abs(delta_from_first)) * 1000.0)
    return rmse_ms, max_abs_ms


def plot_offsets(
    nominal_offsets_seconds: np.ndarray,
    optimized_offsets_seconds: np.ndarray,
    nominal_fps: float,
    optimized_fps: float,
    output_path: Path,
    station_id: str = "Unknown",
) -> None:
    """Plot raw and optimized offsets for a single MKV analysis."""

    import matplotlib.pyplot as plt

    frame_idx = np.arange(len(nominal_offsets_seconds))
    nominal_offsets_ms = nominal_offsets_seconds * 1000.0
    optimized_offsets_ms = optimized_offsets_seconds * 1000.0
    all_offsets_ms = np.concatenate((nominal_offsets_ms, optimized_offsets_ms))
    min_ms = float(np.min(all_offsets_ms))
    max_ms = float(np.max(all_offsets_ms))

    if np.isclose(min_ms, max_ms):
        pad_ms = max(1.0, abs(min_ms) * 0.05)
    else:
        pad_ms = (max_ms - min_ms) * 0.05

    y_min = min_ms - pad_ms
    y_max = max_ms + pad_ms

    plt.figure(figsize=(10, 5))
    plt.plot(
        frame_idx,
        nominal_offsets_ms,
        linewidth=1.0,
        label=f"Offset @ nominal FPS ({nominal_fps:.6f})",
    )
    plt.plot(
        frame_idx,
        optimized_offsets_ms,
        linewidth=1.0,
        label=f"Offset @ optimized FPS ({optimized_fps:.6f})",
    )
    plt.xlabel("Frame index")
    plt.ylabel("Offset (FT - implied) [ms]")
    plt.title(f"MKV implied frame time vs nearest FT time (Station: {station_id})")
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_combined_optimized_offsets(
    curves: list[tuple[str, np.ndarray, float, np.ndarray, str]],
    output_path: Path,
) -> None:
    """Plot accepted optimized curves together on one combined chart."""

    # Plot all accepted optimized curves on one figure.
    import matplotlib.pyplot as plt

    if not curves:
        raise ValueError("No optimized curves available to plot")

    all_offsets_ms = np.concatenate([curve_offsets * 1000.0 for _, curve_offsets, _, _, _ in curves])
    min_ms = float(np.min(all_offsets_ms))
    max_ms = float(np.max(all_offsets_ms))

    if np.isclose(min_ms, max_ms):
        pad_ms = max(1.0, abs(min_ms) * 0.05)
    else:
        pad_ms = (max_ms - min_ms) * 0.05

    y_min = min_ms - pad_ms
    y_max = max_ms + pad_ms

    # Identify unique stations for the title.
    unique_stations = sorted(list(set(station_id for _, _, _, _, station_id in curves)))
    stations_str = ", ".join(unique_stations)

    # Determine time range for coloring based on the first timestamp of each curve.
    curve_times = np.array([unix_times[0] for _, _, _, unix_times, _ in curves])
    t_min, t_max = np.min(curve_times), np.max(curve_times)
    
    import matplotlib.colors as mcolors
    if np.isclose(t_min, t_max):
        norm = mcolors.Normalize(t_min - 1, t_max + 1)
    else:
        norm = mcolors.Normalize(t_min, t_max)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(12, 6))
    for (label, curve_offsets, optimized_fps, _), start_t in zip(curves, curve_times):
        frame_idx = np.arange(len(curve_offsets))
        curve_ms = curve_offsets * 1000.0
        ax.plot(frame_idx, curve_ms, linewidth=0.5, alpha=0.3, color=cmap(norm(start_t)))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax)
    colorbar.set_label("MKV start time (Unix)")

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Offset (FT - implied) [ms]")
    plt.title(f"Optimized MKV frame rate distribution ({len(curves)} files)\nStation(s): {stations_str}")
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_optimized_offset_density(
    curves: list[tuple[str, np.ndarray, float, np.ndarray, str]],
    output_path: Path,
) -> None:
    """Plot 2D density representation of optimized offsets across all curves."""

    import matplotlib.pyplot as plt

    if not curves:
        raise ValueError("No optimized curves available to plot")

    all_frame_indices = []
    all_offsets_ms = []

    for _, offsets, _, _, _ in curves:
        indices = np.arange(len(offsets))
        offsets_ms = offsets * 1000.0
        all_frame_indices.append(indices)
        all_offsets_ms.append(offsets_ms)

    x = np.concatenate(all_frame_indices)
    y = np.concatenate(all_offsets_ms)

    plt.figure(figsize=(12, 6))
    # Use hist2d for a clear density representation.
    plt.hist2d(x, y, bins=[100, 100], cmap="viridis")
    plt.colorbar(label="Density (counts)")
    plt.xlabel("Frame index")
    plt.ylabel("Offset (FT - implied) [ms]")
    unique_stations = sorted(list(set(station_id for _, _, _, _, station_id in curves)))
    stations_str = ", ".join(unique_stations)
    plt.title(f"2D Density of optimized MKV offsets\nStation(s): {stations_str}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def sanitize_for_filename(text: str) -> str:
    """Convert free text into a safe filename segment."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return sanitized.strip("_") or "curve"


def plot_single_optimized_curve(
    label: str,
    nominal_offsets_seconds: np.ndarray,
    offsets_seconds: np.ndarray,
    optimized_fps: float,
    nominal_fps: float,
    output_path: Path,
    station_id: str = "Unknown",
) -> None:
    """Plot raw and optimized offset curves for one sampled MKV result."""

    # Plot raw and optimized versions of one selected curve.
    import matplotlib.pyplot as plt

    frame_idx = np.arange(len(offsets_seconds))
    nominal_offsets_ms = nominal_offsets_seconds * 1000.0
    optimized_offsets_ms = offsets_seconds * 1000.0
    all_offsets_ms = np.concatenate((nominal_offsets_ms, optimized_offsets_ms))
    min_ms = float(np.min(all_offsets_ms))
    max_ms = float(np.max(all_offsets_ms))

    if np.isclose(min_ms, max_ms):
        pad_ms = max(1.0, abs(min_ms) * 0.05)
    else:
        pad_ms = (max_ms - min_ms) * 0.05

    plt.figure(figsize=(10, 5))
    plt.plot(
        frame_idx,
        nominal_offsets_ms,
        linewidth=1.0,
        label=f"raw @ FPS={nominal_fps:.6f}",
    )
    plt.plot(
        frame_idx,
        optimized_offsets_ms,
        linewidth=1.2,
        label=f"optimized @ FPS={optimized_fps:.6f}",
    )
    plt.xlabel("Frame index")
    plt.ylabel("Offset (FT - implied) [ms]")
    plt.title(f"Optimized MKV offset distribution ({len(all_offsets_ms)} frames from {len(curves)} files)\nStation(s): {stations_str}")
    plt.ylim(min_ms - pad_ms, max_ms + pad_ms)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_optimized_offset_summary_band(
    curves: list[tuple[str, np.ndarray, float, np.ndarray, str]],
    output_path: Path,
) -> None:
    """Plot per-curve mean offsets with min-max fill band."""

    # For each curve, show mean offset and min-max band.
    import matplotlib.pyplot as plt
    from scipy.stats import norm, gaussian_kde, skew

    all_fps = np.array([c[2] for c in curves], dtype=float)
    all_times = np.array([c[3][0] for c in curves], dtype=float)

    unique_stations = sorted(list(set(c[4] for c in curves)))
    stations_str = ", ".join(unique_stations)
    mean_ms = np.array([np.mean(offsets) * 1000.0 for _, offsets, _, _, _ in curves], dtype=float)
    min_ms = np.array([np.min(offsets) * 1000.0 for _, offsets, _, _, _ in curves], dtype=float)
    max_ms = np.array([np.max(offsets) * 1000.0 for _, offsets, _, _, _ in curves], dtype=float)
    indices = np.arange(len(curves))

    plt.figure(figsize=(max(10, len(curves) * 0.5), 5))
    plt.fill_between(indices, min_ms, max_ms, alpha=0.25, label="Min-Max range")
    plt.plot(indices, mean_ms, marker="o", linewidth=1.2, label="Mean offset")
    plt.xlabel("Curve index")
    plt.ylabel("Offset (FT - implied) [ms]")
    unique_stations = sorted(list(set(station_id for _, _, _, _, station_id in curves)))
    stations_str = ", ".join(unique_stations)
    plt.title(f"Optimized offset summary (mean with min-max band)\nStation(s): {stations_str}")
    plt.xticks(indices, labels, rotation=45, ha="right", fontsize=8)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_optimized_offset_histogram(
    curves: list[tuple[str, np.ndarray, float, np.ndarray, str]],
    output_path: Path,
    color_by_day_of_year: bool = False,
) -> None:
    """Plot histogram of optimized offsets with bars colored by average Unix time."""

    # Histogram of optimized offsets across all accepted curves.
    # Bar colors encode average implied Unix time in each bin.
    import matplotlib.pyplot as plt
    from scipy.stats import norm, gaussian_kde, skew

    all_offsets_ms = np.concatenate([c[1] * 1000.0 for c in curves])
    all_times = np.concatenate([c[3] for c in curves])

    unique_stations = sorted(list(set(c[4] for c in curves)))
    stations_str = ", ".join(unique_stations)
    all_unix_times = all_times

    color_vals = all_unix_times
    color_label = "Average Unix time"
    if color_by_day_of_year:
        # Vectorized conversion to day of year using NumPy datetime64.
        all_dt = all_unix_times.astype("datetime64[s]")
        day_of_year = (all_dt.astype("datetime64[D]") - all_dt.astype("datetime64[Y]") + np.timedelta64(1, "D")).astype(int)
        color_vals = day_of_year
        color_label = "Average Day of Year"

    counts, bin_edges = np.histogram(all_offsets_ms, bins=60)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    # Vectorized bin assignment and per-bin average color-value computation.
    raw_bin_idx = np.searchsorted(bin_edges, all_offsets_ms, side="right") - 1
    valid_mask = (raw_bin_idx >= 0) & (raw_bin_idx < len(counts))
    finite_unix_mask = np.isfinite(all_unix_times)
    use_mask = valid_mask & finite_unix_mask

    bar_avg_color = np.full(len(counts), np.nan, dtype=float)
    if np.any(use_mask):
        used_idx = raw_bin_idx[use_mask]
        used_color = color_vals[use_mask]
        bin_sums = np.bincount(used_idx, weights=used_color, minlength=len(counts))
        bin_counts = np.bincount(used_idx, minlength=len(counts))
        nonzero = bin_counts > 0
        bar_avg_color[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(bin_centers, counts, width=bin_widths, align="center", alpha=0.6)

    # PDF calculation and overlay
    if all_offsets_ms.size >= 2:
        from scipy.stats import gaussian_kde, skew
        kde = gaussian_kde(all_offsets_ms)
        x_kde = np.linspace(bin_edges[0], bin_edges[-1], 200)
        y_kde = kde(x_kde)
        
        # Scale KDE to match histogram counts
        bin_width = bin_edges[1] - bin_edges[0]
        y_kde_scaled = y_kde * len(all_offsets_ms) * bin_width
        
        ax.plot(x_kde, y_kde_scaled, color="red", linewidth=2, label="PDF (KDE)")
        
        # Stats
        peak_idx = np.argmax(y_kde)
        peak_loc = x_kde[peak_idx]
        std_val = np.std(all_offsets_ms)
        skew_val = skew(all_offsets_ms)
        
        stats_text = (
            f"Peak: {peak_loc:.6f} ms\n"
            f"Std: {std_val:.6f} ms\n"
            f"Skew: {skew_val:.6f}"
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    finite_bar_color = bar_avg_color[np.isfinite(bar_avg_color)]
    if finite_bar_color.size > 0:
        cmap = plt.get_cmap("viridis")
        color_min = float(np.min(finite_bar_color))
        color_max = float(np.max(finite_bar_color))
        if np.isclose(color_min, color_max):
            norm = plt.Normalize(color_min - 0.5, color_max + 0.5)
        else: norm = plt.Normalize(color_min, color_max)

        for bar, avg_c in zip(bars, bar_avg_color):
            if np.isfinite(avg_c):
                bar.set_color(cmap(norm(avg_c)))
            else:
                bar.set_color("tab:gray")

        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=ax)
        colorbar.set_label(color_label)
    else:
        for bar in bars:
            bar.set_color("tab:blue")

    ax.set_xlabel("Optimized offset (FT - implied) [ms]")
    ax.set_ylabel("Count")
    ax.set_title(f"Histogram of optimized offsets\nStation(s): {stations_str}")
    if "PDF (KDE)" in [l.get_label() for l in ax.get_lines()]:
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_optimized_fps_histogram(
    curves: list[tuple[str, np.ndarray, float, np.ndarray, str]],
    output_path: Path,
    color_by_day_of_year: bool = False,
) -> None:
    """Plot histogram of optimized frame rates across accepted curves."""

    # Distribution of optimized frame rates across accepted curves.
    import matplotlib.pyplot as plt

    fps_values = np.array([c[2] for c in curves], dtype=float)
    
    # We need a single time per curve to color the FPS histogram bars.
    # We'll use the mean time of each curve.
    curve_times = np.array([np.mean(c[3]) for c in curves], dtype=float)
    unique_stations = sorted(list(set(c[4] for c in curves)))
    stations_str = ", ".join(unique_stations)

    fig, ax = plt.subplots(figsize=(10, 5))
    counts, bins, bars = ax.hist(fps_values, bins=min(30, max(5, len(fps_values))), alpha=0.6, label="Histogram")
    
    color_vals = curve_times
    color_label = "Average Unix time"
    if color_by_day_of_year:
        # Convert to day of year.
        all_dt = curve_times.astype("datetime64[s]")
        day_of_year = (all_dt.astype("datetime64[D]") - all_dt.astype("datetime64[Y]") + np.timedelta64(1, "D")).astype(int)
        color_vals = day_of_year
        color_label = "Average Day of Year"

    # Color the bars by the average time of the values within each bin.
    # Determine which bin each curve's FPS belongs to.
    bin_idx = np.searchsorted(bins, fps_values, side="right") - 1
    # Handle values exactly on the right edge.
    bin_idx[fps_values == bins[-1]] = len(counts) - 1
    
    bar_avg_color = np.full(len(counts), np.nan, dtype=float)
    for i in range(len(counts)):
        mask = (bin_idx == i)
        if np.any(mask):
            bar_avg_color[i] = np.mean(color_vals[mask])

    finite_bar_color = bar_avg_color[np.isfinite(bar_avg_color)]
    if finite_bar_color.size > 0:
        cmap = plt.get_cmap("viridis")
        color_min = float(np.min(finite_bar_color))
        color_max = float(np.max(finite_bar_color))
        if np.isclose(color_min, color_max):
            norm = plt.Normalize(color_min - 0.5, color_max + 0.5)
        else: norm = plt.Normalize(color_min, color_max)

        for i, bar in enumerate(bars):
            avg_c = bar_avg_color[i]
            if np.isfinite(avg_c):
                bar.set_color(cmap(norm(avg_c)))
            else:
                bar.set_color("tab:gray")

        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=ax)
        colorbar.set_label(color_label)
    else:
        for bar in bars:
            bar.set_color("tab:blue")
    
    # PDF calculation and overlay
    if fps_values.size >= 2:
        from scipy.stats import gaussian_kde, skew
        kde = gaussian_kde(fps_values)
        x_kde = np.linspace(np.min(fps_values), np.max(fps_values), 200)
        y_kde = kde(x_kde)
        
        # Scale KDE to match histogram counts
        bin_width = bins[1] - bins[0]
        y_kde_scaled = y_kde * len(fps_values) * bin_width
        
        plt.plot(x_kde, y_kde_scaled, color="red", linewidth=2, label="PDF (KDE)")
        
        # Stats
        peak_idx = np.argmax(y_kde)
        peak_loc = x_kde[peak_idx]
        std_val = np.std(fps_values)
        skew_val = skew(fps_values)
        
        stats_text = (
            f"Peak: {peak_loc:.6f} fps\n"
            f"Std: {std_val:.6f} fps\n"
            f"Skew: {skew_val:.6f}"
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    ax.set_xlabel("Optimized frame rate [fps]")
    ax.set_ylabel("Count")
    ax.set_title(f"Histogram of optimized frame rates\nStation(s): {stations_str}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ft_interval_histogram(ft_times: np.ndarray, output_path: Path) -> None:
    """Plot histogram of successive FT timestamp intervals in milliseconds."""

    import matplotlib.pyplot as plt

    if ft_times.size < 2:
        raise ValueError("Need at least two FT timestamps to compute intervals")

    # FT timestamps are sorted; intervals are between successive entries.
    intervals_ms = np.diff(ft_times) * 1000.0

    plt.figure(figsize=(10, 5))
    plt.hist(intervals_ms, bins=200, range=(39.999, 40.001))
    plt.xlabel("Successive FT interval [ms]")
    plt.ylabel("Count")
    plt.title("Histogram of successive FT timestamp intervals")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def optimized_curve_std_ms(offsets_seconds: np.ndarray) -> float:
    """Compute standard deviation of an offset curve in milliseconds."""

    return float(np.std(offsets_seconds) * 1000.0)


def has_negative_offsets(offsets_seconds: np.ndarray) -> bool:
    """Return True if any offset in the curve is negative."""

    return bool(np.any(offsets_seconds < 0.0))


def try_create_plot(plot_name: str, create_plot_fn) -> bool:
    """Execute plotting callback safely and report failures without raising."""

    # Keep pipeline robust: one plot failure should not stop analysis.
    try:
        create_plot_fn()
        return True
    except Exception as exc:
        print(f"Warning: failed to create {plot_name}: {exc}")
        return False


def analyze_single_mkv(
    mkv_path: Path,
    station_id: str,
    mkv_start_utc: datetime,
    frame_rate: float,
    ft_times: np.ndarray,
    plot_output: Path | None,
    print_offsets_by_frame: bool,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, str]:
    """Run full MKV-vs-FT timing analysis and return raw/optimized artifacts."""

    # End-to-end analysis for one MKV against provided FT times.
    # Returns raw offsets, optimized offsets, optimized FPS, and optimized implied Unix times.
    frame_count = run_ffprobe_frame_count(mkv_path)
    print(f"\nMKV file: {mkv_path}")
    print(f"Station in filename: {station_id}")
    print(f"MKV start time (from filename): {mkv_start_utc.isoformat()}")
    print(f"Frame rate for implied times: {frame_rate}")
    print(f"Frame count: {frame_count}")

    frame_indices = np.arange(frame_count, dtype=np.float64)
    start_timestamp = mkv_start_utc.timestamp()
    implied_times = implied_times_for_fps(start_timestamp, frame_indices, frame_rate)
    _, offsets = nearest_offsets_seconds(implied_times, ft_times)

    # Optimization explicitly uses all frames.
    optimization_frame_indices = frame_indices
    print(f"Frames used for FPS optimization: {optimization_frame_indices.size} (all frames)")

    optimized_fps = optimize_fps(
        frame_rate, start_timestamp, optimization_frame_indices, ft_times
    )
    optimized_implied_times = implied_times_for_fps(start_timestamp, frame_indices, optimized_fps)
    _, optimized_offsets = nearest_offsets_seconds(
        optimized_implied_times, ft_times
    )

    print(f"Optimized frame rate [fps]: {optimized_fps:.9f}")
    print(
        "Optimization objective: keep all frame offsets as close as possible to first-frame offset"
    )
    print_offset_summary("Offset summary @ nominal FPS (FT - implied):", offsets)
    print_offset_summary("Offset summary @ optimized FPS (FT - implied):", optimized_offsets)

    nominal_anchor_rmse_ms, nominal_anchor_max_abs_ms = anchor_error_metrics(offsets)
    optimized_anchor_rmse_ms, optimized_anchor_max_abs_ms = anchor_error_metrics(
        optimized_offsets
    )
    print("Optimization errors relative to first-frame offset:")
    print(f"  nominal anchor RMSE [ms]:   {nominal_anchor_rmse_ms:.6f}")
    print(f"  optimized anchor RMSE [ms]: {optimized_anchor_rmse_ms:.6f}")
    print(f"  nominal max |delta| [ms]:   {nominal_anchor_max_abs_ms:.6f}")
    print(f"  optimized max |delta| [ms]: {optimized_anchor_max_abs_ms:.6f}")

    if print_offsets_by_frame:
        print("Offsets by frame [ms]:")
        print("  frame,nominal_offset_ms,optimized_offset_ms")
        for index in range(frame_count):
            offset_ms = offsets[index] * 1000.0
            optimized_offset_ms = optimized_offsets[index] * 1000.0
            print(f"  {index},{offset_ms:.6f},{optimized_offset_ms:.6f}")

    if plot_output is not None:
        created = try_create_plot(
            f"offset plot ({plot_output})",
            lambda: plot_offsets(offsets, optimized_offsets, frame_rate, optimized_fps, plot_output, station_id),
        )
        if created:
            print(f"Saved offset plot to: {plot_output}")

    return (
        offsets,
        optimized_offsets,
        optimized_fps,
        optimized_implied_times,
        station_id,
    )


def run_batch_mkv_task(
    mkv_path_str: str,
    station_id: str,
    mkv_start_iso: str,
    frame_rate: float,
    ft_times: np.ndarray,
) -> tuple[str, str, np.ndarray, np.ndarray, float, np.ndarray]:
    """Worker task for batch multiprocessing; returns logs and computed arrays."""

    # Worker entry-point for multiprocessing batch mode.
    # Captures print output so main process can emit clean sequential logs.
    mkv_path = Path(mkv_path_str)
    mkv_start_utc = datetime.fromisoformat(mkv_start_iso)

    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        nominal_offsets, optimized_offsets, optimized_fps, optimized_implied_times, station_id_out = analyze_single_mkv(
            mkv_path=mkv_path,
            station_id=station_id,
            mkv_start_utc=mkv_start_utc,
            frame_rate=frame_rate,
            ft_times=ft_times,
            plot_output=None,
            print_offsets_by_frame=False,
        )

    return (
        output_buffer.getvalue(),
        mkv_path.stem,
        nominal_offsets,
        optimized_offsets,
        optimized_fps,
        optimized_implied_times,
        station_id_out,
    )



def main() -> int:
    """CLI entry point for single-mode and batch-mode timing analysis."""

    # CLI entry-point.
    # - default mode: one MKV + all FT under search root
    # - batch mode: overlapping MKV/FT days, filtering, and summary plots
    parser = argparse.ArgumentParser(
        description=(
            "Compute frame-time offsets between an MKV timeline and nearest FT timestamps, "
            "then generate a plot."
        )
    )
    parser.add_argument(
        "--search-root",
        default="TestData",
        help="Default mode root directory where FT and MKV files are searched (default: TestData).",
    )
    parser.add_argument(
        "--mkv",
        default=None,
        help="Optional single MKV path for default mode.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Frame rate used to compute implied frame times (default: 25.0).",
    )
    parser.add_argument(
        "--rms-root",
        default="/home/mmazur/source/RMS",
        help="Path to RMS repository root for importing RMS modules.",
    )
    parser.add_argument(
        "--plot-output",
        default="mkv_ft_time_offsets.png",
        help="Output PNG path for offset plot.",
    )
    parser.add_argument(
        "--mkvdir",
        default=None,
        help="Root directory containing MKV files in YYYY/YYYYMMDD-XXX/YYYYMMDD-XXX_HH layout.",
    )
    parser.add_argument(
        "--ftdir",
        default=None,
        help="Root directory containing FT files in YYYY/YYYYMMDD-XXX/YYYYMMDD-XXX_HH layout.",
    )
    parser.add_argument(
        "--number-of-days",
        type=int,
        default=1,
        help="Number of overlapping days to process when --mkvdir and --ftdir are provided.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for batch mode MKV processing (default: 1).",
    )
    parser.add_argument(
        "--camera-id",
        default=None,
        help="Camera ID to automatically set mkvdir and ftdir under /mnt/RMS_data.",
    )
    parser.add_argument(
        "--pickle-output",
        default=None,
        help="Optional path to save calculated offsets and frame rates as a pickle file.",
    )
    args = parser.parse_args()

    # If camera-id is specified, override mkvdir and ftdir.
    if args.camera_id:
        args.mkvdir = f"/mnt/RMS_data/{args.camera_id}/VideoFiles"
        args.ftdir = f"/mnt/RMS_data/{args.camera_id}/TimeFiles"

    rms_root = Path(args.rms_root).expanduser().resolve()
    requested_plot_output = Path(args.plot_output).expanduser().resolve()
    figures_dir = requested_plot_output.parent / "Figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_output = figures_dir / requested_plot_output.name

    # If pickle-output not specified, default to same name/location as plot_output.
    if args.pickle_output is None:
        args.pickle_output = plot_output.with_suffix(".pkl")
    else:
        args.pickle_output = Path(args.pickle_output).expanduser().resolve()

    if args.number_of_days < 1:
        print("--number-of-days must be >= 1")
        return 1
    if args.workers < 1:
        print("--workers must be >= 1")
        return 1

    # Batch mode is enabled when either directory override is provided.
    use_batch_mode = args.mkvdir is not None or args.ftdir is not None
    if use_batch_mode and (args.mkvdir is None or args.ftdir is None):
        print("When using batch mode, both --mkvdir and --ftdir must be specified")
        return 1

    if not use_batch_mode:
        # Single/default mode.
        search_root = Path(args.search_root).expanduser().resolve()
        if not search_root.is_dir():
            print(f"Search root is not a directory: {search_root}")
            return 1

        if args.mkv is not None:
            mkv_path = Path(args.mkv).expanduser().resolve()
            if not mkv_path.is_file():
                print(f"MKV file does not exist: {mkv_path}")
                return 1
        else:
            mkv_files = discover_mkv_files(search_root)
            if not mkv_files:
                print(f"No MKV files found under {search_root}")
                return 1
            mkv_path = mkv_files[0]

        ft_files = discover_ft_files(search_root)
        if not ft_files:
            print(f"No FT files found under {search_root}")
            return 1

        station_id, mkv_start = parse_mkv_start_from_filename(mkv_path)
        mkv_start_utc = mkv_start.replace(tzinfo=timezone.utc)

        print(f"FT files discovered: {len(ft_files)}")
        ft_times = collect_ft_times_seconds(ft_files, rms_root)
        print(f"Total FT time entries: {len(ft_times)}")

        ft_interval_plot = plot_output.with_name(
            f"{plot_output.stem}_ft_interval_hist.png"
        )
        created = try_create_plot(
            f"FT interval histogram ({ft_interval_plot})",
            lambda: plot_ft_interval_histogram(ft_times, ft_interval_plot),
        )
        if created:
            print(f"Saved FT interval histogram to: {ft_interval_plot}")

        nominal_offsets, optimized_offsets, optimized_fps, optimized_implied_times, station_id = analyze_single_mkv(
            mkv_path=mkv_path,
            station_id=station_id,
            mkv_start_utc=mkv_start_utc,
            frame_rate=args.fps,
            ft_times=ft_times,
            plot_output=plot_output,
            print_offsets_by_frame=True,
        )
        
        # Populate combined_curves for pickle output even in single mode.
        combined_curves = [(mkv_path.stem, offsets, optimized_offsets, optimized_fps, optimized_implied_times)]
    else:
        mkv_root = Path(args.mkvdir).expanduser().resolve()
        ft_root = Path(args.ftdir).expanduser().resolve()

        if not mkv_root.is_dir():
            print(f"MKV directory is not valid: {mkv_root}")
            return 1
        if not ft_root.is_dir():
            print(f"FT directory is not valid: {ft_root}")
            return 1

        # Batch mode: discover files and keep only overlapping days.
        mkv_meta = discover_mkv_meta(mkv_root)
        if not mkv_meta:
            print(f"No matching MKV files found under {mkv_root}")
            return 1

        ft_files = discover_ft_files(ft_root)
        if not ft_files:
            print(f"No matching FT files found under {ft_root}")
            return 1

        mkv_by_day = group_mkv_by_day(mkv_meta)
        ft_by_day = group_ft_by_day(ft_files)
        overlap_days = sorted(set(mkv_by_day).intersection(ft_by_day))

        if not overlap_days:
            print("No overlapping MKV/FT days found between the provided directories")
            return 1

        selected_days = overlap_days[: args.number_of_days]
        print(f"Overlapping days found: {len(overlap_days)}")
        print(f"Processing first {len(selected_days)} day(s): {', '.join(selected_days)}")

        # Accepted curves after filtering; used for aggregate plots.
        combined_curves: list[tuple[str, np.ndarray, np.ndarray, float, np.ndarray]] = []

        # Process MKVs either serially or in parallel.
        if args.workers == 1:
            for day_key in selected_days:
                day_mkv_files = mkv_by_day[day_key]
                day_ft_files = ft_by_day[day_key]
                print(f"\n=== Day {day_key}: MKV files={len(day_mkv_files)}, FT files={len(day_ft_files)} ===")

                day_ft_times = collect_ft_times_seconds(day_ft_files, rms_root)
                print(f"FT time entries for day {day_key}: {len(day_ft_times)}")

                day_ft_interval_plot = plot_output.with_name(
                    f"{plot_output.stem}_ft_interval_hist_{day_key}.png"
                )
                try_create_plot(
                    f"FT interval histogram ({day_ft_interval_plot})",
                    lambda: plot_ft_interval_histogram(day_ft_times, day_ft_interval_plot),
                )

                for mkv_item in day_mkv_files:
                    (
                        nominal_offsets,
                        optimized_offsets,
                        optimized_fps,
                        optimized_implied_times,
                    ) = analyze_single_mkv(
                        mkv_path=mkv_item.path,
                        station_id=mkv_item.station_id,
                        mkv_start_utc=mkv_item.start_utc,
                        frame_rate=args.fps,
                        ft_times=day_ft_times,
                        plot_output=None,
                        print_offsets_by_frame=False,
                    )
                    
                    curve_std_ms = optimized_curve_std_ms(optimized_offsets)
                    if has_negative_offsets(optimized_offsets):
                        print(f"Discarding curve {mkv_item.path.stem}: contains negative offsets")
                    elif curve_std_ms > OPTIMIZED_CURVE_STD_THRESHOLD_MS:
                        print(f"Discarding curve {mkv_item.path.stem}: std={curve_std_ms:.3f} ms > {OPTIMIZED_CURVE_STD_THRESHOLD_MS:.1f} ms")
                    else:
                        combined_curves.append((mkv_item.path.stem, nominal_offsets, optimized_offsets, optimized_fps, optimized_implied_times))
        else:
            print(f"Processing MKV files with multiprocessing workers={args.workers}")
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for day_key in selected_days:
                    day_mkv_files = mkv_by_day[day_key]
                    day_ft_files = ft_by_day[day_key]
                    print(f"\n=== Day {day_key}: MKV files={len(day_mkv_files)}, FT files={len(day_ft_files)} ===")

                    day_ft_times = collect_ft_times_seconds(day_ft_files, rms_root)
                    print(f"FT time entries for day {day_key}: {len(day_ft_times)}")

                    day_ft_interval_plot = plot_output.with_name(
                        f"{plot_output.stem}_ft_interval_hist_{day_key}.png"
                    )
                    try_create_plot(
                        f"FT interval histogram ({day_ft_interval_plot})",
                        lambda: plot_ft_interval_histogram(day_ft_times, day_ft_interval_plot),
                    )

                    futures = []
                    for mkv_item in day_mkv_files:
                        futures.append(
                            executor.submit(
                                run_batch_mkv_task,
                                str(mkv_item.path),
                                mkv_item.station_id,
                                mkv_item.start_utc.isoformat(),
                                args.fps,
                                day_ft_times,
                            )
                        )

                    for future in as_completed(futures):
                        (
                            output_text,
                            label,
                            nominal_offsets,
                            optimized_offsets,
                            optimized_fps,
                            optimized_implied_times,
                        ) = future.result()
                        print(output_text, end="")
                        
                        curve_std_ms = optimized_curve_std_ms(optimized_offsets)
                        if has_negative_offsets(optimized_offsets):
                            print(f"Discarding curve {label}: contains negative offsets")
                        elif curve_std_ms > OPTIMIZED_CURVE_STD_THRESHOLD_MS:
                            print(f"Discarding curve {label}: std={curve_std_ms:.3f} ms > {OPTIMIZED_CURVE_STD_THRESHOLD_MS:.1f} ms")
                        else:
                            station_id = result[5]
                            combined_curves.append((label, nominal_offsets, optimized_offsets, optimized_fps, optimized_implied_times, station_id))

        if combined_curves:
            # Build derived curve representations for different aggregate plots.
            optimized_only_curves = [
                (label, optimized_offsets, optimized_fps)
                for label, _, optimized_offsets, optimized_fps, _, _ in combined_curves
            ]
            optimized_curves_with_time_and_station = [
                (label, optimized_offsets, optimized_fps, optimized_implied_times, station_id)
                for label, _, optimized_offsets, optimized_fps, optimized_implied_times, station_id in combined_curves
            ]
            created = try_create_plot(
                f"combined optimized-offset plot ({plot_output})",
                lambda: plot_combined_optimized_offsets(optimized_curves_with_time_and_station, plot_output),
            )
            if created:
                print(f"\nSaved combined optimized-offset plot to: {plot_output}")

            density_plot = plot_output.with_name(
                f"{plot_output.stem}_optimized_offset_density_2d.png"
            )
            created = try_create_plot(
                f"2D density offset plot ({density_plot})",
                lambda: plot_optimized_offset_density(optimized_curves_with_time_and_station, density_plot),
            )
            if created:
                print(f"Saved 2D density offset plot to: {density_plot}")

            sample_count = min(5, len(combined_curves))
            sampled_curves = random.sample(combined_curves, k=sample_count)
            print(f"Creating {sample_count} randomly selected single-curve optimized plots")
            for index, (label, nominal_offsets, offsets, optimized_fps, _, station_id) in enumerate(
                sampled_curves, start=1
            ):
                sample_plot = plot_output.with_name(
                    f"{plot_output.stem}_sample_{index}_{sanitize_for_filename(label)}.png"
                )
                created = try_create_plot(
                    f"single optimized curve plot ({sample_plot})",
                    lambda label=label, nominal_offsets=nominal_offsets, offsets=offsets, optimized_fps=optimized_fps, station_id=station_id, sample_plot=sample_plot: plot_single_optimized_curve(
                        label,
                        nominal_offsets,
                        offsets,
                        optimized_fps,
                        args.fps,
                        sample_plot,
                        station_id,
                    ),
                )
                if created:
                    print(f"Saved single optimized curve plot to: {sample_plot}")

            summary_band_plot = plot_output.with_name(
                f"{plot_output.stem}_optimized_offset_summary_band.png"
            )
            created = try_create_plot(
                f"optimized offset-summary band plot ({summary_band_plot})",
                lambda: plot_optimized_offset_summary_band(optimized_curves_with_time_and_station, summary_band_plot),
            )
            if created:
                print(f"Saved optimized offset-summary band plot to: {summary_band_plot}")

            offset_hist_plot = plot_output.with_name(
                f"{plot_output.stem}_optimized_offset_hist.png"
            )
            created = try_create_plot(
                f"optimized offset histogram ({offset_hist_plot})",
                lambda: plot_optimized_offset_histogram(
                    optimized_curves_with_time_and_station,
                    offset_hist_plot,
                    color_by_day_of_year=(args.number_of_days > 1)
                ),
            )
            if created:
                print(f"Saved optimized offset histogram to: {offset_hist_plot}")

            fps_hist_plot = plot_output.with_name(
                f"{plot_output.stem}_optimized_fps_hist.png"
            )
            created = try_create_plot(
                f"optimized frame-rate histogram ({fps_hist_plot})",
                lambda: plot_optimized_fps_histogram(
                    optimized_curves_with_time_and_station,
                    fps_hist_plot,
                    color_by_day_of_year=(args.number_of_days > 1)
                ),
            )
            if created:
                print(f"Saved optimized frame-rate histogram to: {fps_hist_plot}")
            else:
                print(
                    "\nNo optimized curves met the standard deviation threshold; combined plot was not created"
                )

    # Save to pickle.
    if "combined_curves" in locals() and combined_curves:
        pickle_path = args.pickle_output
        try:
            with open(pickle_path, "wb") as f:
                pickle.dump(combined_curves, f)
            print(f"\nSaved results for {len(combined_curves)} curve(s) to: {pickle_path}")
        except Exception as exc:
            print(f"Warning: failed to save pickle file: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
