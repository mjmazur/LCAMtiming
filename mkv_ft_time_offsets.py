#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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
    path: Path
    station_id: str
    timestamp: datetime


@dataclass(frozen=True)
class MKVFileMeta:
    path: Path
    station_id: str
    start_utc: datetime


def parse_ft_filename(path: Path) -> FTFileMeta | None:
    match = FT_FILENAME_PATTERN.match(path.name)
    if match:
        station_id = match.group("station")
        timestamp_text = f"{match.group('date')}{match.group('time')}"
        timestamp = datetime.strptime(timestamp_text, "%Y%m%d%H%M%S")
        return FTFileMeta(path=path, station_id=station_id, timestamp=timestamp)

    match_no_station = FT_FILENAME_NO_STATION_PATTERN.match(path.name)
    if not match_no_station:
        return None

    timestamp_text = f"{match_no_station.group('date')}{match_no_station.group('time')}"
    timestamp = datetime.strptime(timestamp_text, "%Y%m%d%H%M%S")
    return FTFileMeta(path=path, station_id="UNKNOWN", timestamp=timestamp)


def discover_ft_files(search_root: Path) -> list[FTFileMeta]:
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
    mkvs = [path for path in search_root.rglob("*.mkv") if path.is_file()]
    mkvs.sort()
    return mkvs


def discover_mkv_meta(search_root: Path) -> list[MKVFileMeta]:
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
    grouped: dict[str, list[FTFileMeta]] = {}
    for item in ft_files:
        day_key = item.timestamp.strftime("%Y%m%d")
        grouped.setdefault(day_key, []).append(item)
    return grouped


def group_mkv_by_day(mkv_files: list[MKVFileMeta]) -> dict[str, list[MKVFileMeta]]:
    grouped: dict[str, list[MKVFileMeta]] = {}
    for item in mkv_files:
        day_key = item.start_utc.strftime("%Y%m%d")
        grouped.setdefault(day_key, []).append(item)
    return grouped


def match_ft_to_mkv(ft_files: list[FTFileMeta], mkv_files: list[MKVFileMeta]) -> list[tuple[FTFileMeta, MKVFileMeta]]:
    mkv_by_day = group_mkv_by_day(mkv_files)
    matched: list[tuple[FTFileMeta, MKVFileMeta]] = []

    for ft_file in ft_files:
        day_key = ft_file.timestamp.strftime("%Y%m%d")
        day_mkvs = mkv_by_day.get(day_key, [])
        if not day_mkvs:
            continue

        ft_ts = ft_file.timestamp.replace(tzinfo=timezone.utc).timestamp()
        closest_mkv = min(
            day_mkvs,
            key=lambda mkv_item: abs(mkv_item.start_utc.timestamp() - ft_ts),
        )
        matched.append((ft_file, closest_mkv))

    return matched


def run_ffprobe_frame_count(mkv_path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(mkv_path),
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {mkv_path}: {result.stderr.strip()}")

    for line in result.stdout.splitlines():
        value = line.strip()
        if value.isdigit():
            parsed = int(value)
            if parsed > 0:
                return parsed

    fallback_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(mkv_path),
    ]
    fallback_result = subprocess.run(
        fallback_command, check=False, capture_output=True, text=True
    )
    if fallback_result.returncode != 0:
        raise RuntimeError(
            f"ffprobe duration fallback failed for {mkv_path}: {fallback_result.stderr.strip()}"
        )

    duration_text = fallback_result.stdout.strip()
    if not duration_text:
        raise RuntimeError(f"Could not determine frame count for {mkv_path}")

    duration = float(duration_text)
    fallback_fps = 25.0
    frames = int(round(duration * fallback_fps))
    if frames <= 0:
        raise RuntimeError(f"Duration-based frame count invalid for {mkv_path}")
    return frames


def import_ftfile_read(rms_root: Path):
    try:
        from RMS.Formats import FTfile

        return FTfile.read
    except ModuleNotFoundError:
        sys.path.insert(0, str(rms_root))
        from RMS.Formats import FTfile

        return FTfile.read


def collect_ft_times_seconds(ft_files: list[FTFileMeta], rms_root: Path) -> np.ndarray:
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
    return start_timestamp + (frame_indices / fps)


def offset_deviation_from_first_for_fps(
    fps: float,
    start_timestamp: float,
    frame_indices: np.ndarray,
    ft_times: np.ndarray,
) -> float:
    implied_times = implied_times_for_fps(start_timestamp, frame_indices, fps)
    _, offsets = nearest_offsets_seconds(implied_times, ft_times)
    first_offset = float(offsets[0])
    return float(np.mean((offsets - first_offset) ** 2))


def optimize_fps(
    initial_fps: float,
    start_timestamp: float,
    frame_indices_all: np.ndarray,
    ft_times: np.ndarray,
) -> float:
    if frame_indices_all.size == 0:
        raise ValueError("No frame indices provided for FPS optimization")

    best_fps = initial_fps
    best_error = offset_deviation_from_first_for_fps(
        best_fps, start_timestamp, frame_indices_all, ft_times
    )

    search_stages = [
        (0.5, 201),
        (0.05, 201),
        (0.001, 201),
        (0.0001, 201),
    ]

    for half_width, steps in search_stages:
        min_fps = max(0.1, best_fps - half_width)
        max_fps = best_fps + half_width
        candidates = np.linspace(min_fps, max_fps, num=steps)

        for candidate_fps in candidates:
            error = offset_deviation_from_first_for_fps(
                float(candidate_fps), start_timestamp, frame_indices_all, ft_times
            )
            if error < best_error:
                best_error = error
                best_fps = float(candidate_fps)

    return best_fps


def summarize_offsets(offsets: np.ndarray) -> tuple[float, float, float, float]:
    mean_ms = float(np.mean(offsets) * 1000.0)
    median_ms = float(np.median(offsets) * 1000.0)
    min_ms = float(np.min(offsets) * 1000.0)
    max_ms = float(np.max(offsets) * 1000.0)
    return mean_ms, median_ms, min_ms, max_ms


def print_offset_summary(title: str, offsets: np.ndarray) -> None:
    mean_ms, median_ms, min_ms, max_ms = summarize_offsets(offsets)
    print(f"\n{title}")
    print(f"  mean [ms]:   {mean_ms:.6f}")
    print(f"  median [ms]: {median_ms:.6f}")
    print(f"  min [ms]:    {min_ms:.6f}")
    print(f"  max [ms]:    {max_ms:.6f}")


def anchor_error_metrics(offsets: np.ndarray) -> tuple[float, float]:
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
) -> None:
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
    plt.title("MKV implied frame time vs nearest FT time")
    plt.ylim(y_min, y_max)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_batch_offsets(
    batch_series: list[tuple[np.ndarray, np.ndarray, str]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not batch_series:
        return

    all_offsets = np.concatenate([series[1] for series in batch_series])
    min_ms = float(np.min(all_offsets))
    max_ms = float(np.max(all_offsets))

    if np.isclose(min_ms, max_ms):
        pad_ms = max(1.0, abs(min_ms) * 0.05)
    else:
        pad_ms = (max_ms - min_ms) * 0.05

    y_min = min_ms - pad_ms
    y_max = max_ms + pad_ms

    plt.figure(figsize=(12, 6))
    for frame_idx, offsets_ms, label in batch_series:
        plt.plot(frame_idx, offsets_ms, linewidth=1.0, label=label)

    plt.xlabel("Frame index")
    plt.ylabel("Offset (FT - implied) [ms]")
    plt.title("Optimized offsets for matched FT/MKV files")
    plt.ylim(y_min, y_max)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def analyze_single_mkv(
    mkv_path: Path,
    station_id: str,
    mkv_start_utc: datetime,
    frame_rate: float,
    ft_times: np.ndarray,
    plot_output: Path,
    print_offsets_by_frame: bool,
    generate_plot: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
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

    if generate_plot:
        plot_offsets(offsets, optimized_offsets, frame_rate, optimized_fps, plot_output)
        print(f"Saved offset plot to: {plot_output}")

    label = f"{mkv_path.stem} vs {Path(plot_output).stem}"
    return np.arange(frame_count), optimized_offsets * 1000.0, label


def analyze_batch_pair_task(
    task: tuple[int, str, str, str, float, float, str]
) -> tuple[int, np.ndarray, np.ndarray, str, list[str]]:
    (
        pair_index,
        ft_path_str,
        mkv_path_str,
        station_id,
        mkv_start_ts,
        frame_rate,
        rms_root_str,
    ) = task

    ft_path = Path(ft_path_str)
    mkv_path = Path(mkv_path_str)
    rms_root = Path(rms_root_str)

    ft_meta = parse_ft_filename(ft_path)
    if ft_meta is None:
        raise ValueError(f"FT filename format not recognized: {ft_path}")

    ft_times = collect_ft_times_seconds([ft_meta], rms_root)

    frame_count = run_ffprobe_frame_count(mkv_path)
    frame_indices = np.arange(frame_count, dtype=np.float64)
    implied_times = implied_times_for_fps(mkv_start_ts, frame_indices, frame_rate)
    _, offsets = nearest_offsets_seconds(implied_times, ft_times)

    optimized_fps = optimize_fps(frame_rate, mkv_start_ts, frame_indices, ft_times)
    optimized_implied_times = implied_times_for_fps(mkv_start_ts, frame_indices, optimized_fps)
    _, optimized_offsets = nearest_offsets_seconds(optimized_implied_times, ft_times)

    nominal_anchor_rmse_ms, nominal_anchor_max_abs_ms = anchor_error_metrics(offsets)
    optimized_anchor_rmse_ms, optimized_anchor_max_abs_ms = anchor_error_metrics(
        optimized_offsets
    )
    mean_ms, median_ms, min_ms, max_ms = summarize_offsets(optimized_offsets)

    report_lines = [
        f"=== Pair {pair_index}: FT={ft_path.name}  MKV={mkv_path.name} ===",
        f"FT time entries in file: {len(ft_times)}",
        f"Frame count: {frame_count}",
        f"Optimized frame rate [fps]: {optimized_fps:.9f}",
        "Offset summary @ optimized FPS (FT - implied):",
        f"  mean [ms]:   {mean_ms:.6f}",
        f"  median [ms]: {median_ms:.6f}",
        f"  min [ms]:    {min_ms:.6f}",
        f"  max [ms]:    {max_ms:.6f}",
        "Optimization errors relative to first-frame offset:",
        f"  nominal anchor RMSE [ms]:   {nominal_anchor_rmse_ms:.6f}",
        f"  optimized anchor RMSE [ms]: {optimized_anchor_rmse_ms:.6f}",
        f"  nominal max |delta| [ms]:   {nominal_anchor_max_abs_ms:.6f}",
        f"  optimized max |delta| [ms]: {optimized_anchor_max_abs_ms:.6f}",
    ]

    batch_label = f"{ft_path.stem} -> {mkv_path.stem}"
    return pair_index, frame_indices, optimized_offsets * 1000.0, batch_label, report_lines


def main() -> int:
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
        "--number-of-ft-files",
        type=int,
        default=1,
        help="Number of matched FT files to process when --mkvdir and --ftdir are provided.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes for batch mode (0 = auto).",
    )
    args = parser.parse_args()

    rms_root = Path(args.rms_root).expanduser().resolve()
    plot_output = Path(args.plot_output).expanduser().resolve()

    if args.number_of_ft_files < 1:
        print("--number-of-ft-files must be >= 1")
        return 1

    use_batch_mode = args.mkvdir is not None or args.ftdir is not None
    if use_batch_mode and (args.mkvdir is None or args.ftdir is None):
        print("When using batch mode, both --mkvdir and --ftdir must be specified")
        return 1

    if not use_batch_mode:
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

        analyze_single_mkv(
            mkv_path=mkv_path,
            station_id=station_id,
            mkv_start_utc=mkv_start_utc,
            frame_rate=args.fps,
            ft_times=ft_times,
            plot_output=plot_output,
            print_offsets_by_frame=True,
            generate_plot=True,
        )
        return 0

    mkv_root = Path(args.mkvdir).expanduser().resolve()
    ft_root = Path(args.ftdir).expanduser().resolve()

    if not mkv_root.is_dir():
        print(f"MKV directory is not valid: {mkv_root}")
        return 1
    if not ft_root.is_dir():
        print(f"FT directory is not valid: {ft_root}")
        return 1

    mkv_meta = discover_mkv_meta(mkv_root)
    if not mkv_meta:
        print(f"No matching MKV files found under {mkv_root}")
        return 1

    ft_files = discover_ft_files(ft_root)
    if not ft_files:
        print(f"No matching FT files found under {ft_root}")
        return 1

    matched_pairs = match_ft_to_mkv(ft_files, mkv_meta)
    if not matched_pairs:
        print("No overlapping FT/MKV files found between the provided directories")
        return 1

    selected_pairs = matched_pairs[: args.number_of_ft_files]
    print(f"Matched FT/MKV pairs found: {len(matched_pairs)}")
    print(f"Processing first {len(selected_pairs)} matched FT file(s)")

    batch_plot_series: list[tuple[np.ndarray, np.ndarray, str]] = []

    tasks: list[tuple[int, str, str, str, float, float, str]] = []
    for index, (ft_item, mkv_item) in enumerate(selected_pairs, start=1):
        tasks.append(
            (
                index,
                str(ft_item.path),
                str(mkv_item.path),
                mkv_item.station_id,
                mkv_item.start_utc.timestamp(),
                args.fps,
                str(rms_root),
            )
        )

    requested_workers = args.workers
    auto_workers = os.cpu_count() or 1
    max_workers = auto_workers if requested_workers == 0 else requested_workers
    if max_workers < 1:
        print("--workers must be >= 1 (or 0 for auto)")
        return 1

    print(f"Using {max_workers} worker process(es) for batch analysis")

    if max_workers == 1:
        results = [analyze_batch_pair_task(task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(analyze_batch_pair_task, task): task[0] for task in tasks}
            for future in as_completed(future_map):
                results.append(future.result())

    results.sort(key=lambda item: item[0])
    for _, frame_idx, optimized_offsets_ms, batch_label, report_lines in results:
        print()
        for line in report_lines:
            print(line)
        batch_plot_series.append((frame_idx, optimized_offsets_ms, batch_label))

    plot_batch_offsets(batch_plot_series, plot_output)
    print(f"\nSaved combined batch plot to: {plot_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
