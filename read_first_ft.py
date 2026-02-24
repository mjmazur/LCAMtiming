#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


FT_FILENAME_PATTERN = re.compile(
    r"^FT_(?P<station>.+)_(?P<date>\d{8})_(?P<time>\d{6})\.bin$"
)


def parse_ft_filename(path: Path) -> tuple[str, datetime] | None:
    """Parse FT filename and return (station_id, timestamp) if valid."""

    match = FT_FILENAME_PATTERN.match(path.name)
    if not match:
        return None

    station_id = match.group("station")
    timestamp_text = f"{match.group('date')}{match.group('time')}"
    timestamp = datetime.strptime(timestamp_text, "%Y%m%d%H%M%S")
    return station_id, timestamp


def discover_ft_files(search_root: Path) -> list[tuple[Path, str, datetime]]:
    """Recursively discover FT files below search_root and sort by timestamp."""

    found: list[tuple[Path, str, datetime]] = []
    for path in search_root.rglob("FT_*.bin"):
        if not path.is_file():
            continue
        parsed = parse_ft_filename(path)
        if parsed is None:
            continue
        station_id, timestamp = parsed
        found.append((path, station_id, timestamp))

    found.sort(key=lambda item: (item[2], str(item[0])))
    return found


def import_ftfile_read(rms_root: Path):
    """Import RMS FTfile.read, optionally adding the RMS repo root to sys.path."""

    try:
        from RMS.Formats import FTfile

        return FTfile.read
    except ModuleNotFoundError:
        sys.path.insert(0, str(rms_root))
        from RMS.Formats import FTfile

        return FTfile.read


def main() -> int:
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("must be >= 1")
        return parsed

    parser = argparse.ArgumentParser(
        description=(
            "Find FT_StationID_YYYYMMDD_HHmmss.bin files, sort them by time, "
            "and read the first one using RMS.Formats.FTfile.read."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="TestData",
        help="Directory to search recursively (default: TestData).",
    )
    parser.add_argument(
        "--rms-root",
        default="/home/mmazur/source/RMS",
        help="Path to RMS repository root for importing RMS modules.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only summary and first-file read result.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="In non-quiet mode, print only the first N sorted files.",
    )
    args = parser.parse_args()

    search_root = Path(args.directory).expanduser().resolve()
    if not search_root.exists() or not search_root.is_dir():
        print(f"Search directory does not exist or is not a directory: {search_root}")
        return 1

    ft_files = discover_ft_files(search_root)
    if not ft_files:
        print(f"No matching FT files found below {search_root}")
        return 1

    print(f"Found {len(ft_files)} FT files below {search_root}")
    if not args.quiet:
        files_to_print = ft_files if args.limit is None else ft_files[: args.limit]
        for index, (path, station_id, timestamp) in enumerate(files_to_print, start=1):
            print(f"{index:4d}. {timestamp.isoformat(sep=' ')}  {station_id:12s}  {path}")

    first_path, _, first_timestamp = ft_files[0]
    print("\nReading first file:")
    print(f"  {first_path}  (timestamp: {first_timestamp.isoformat(sep=' ')})")

    read_ft = import_ftfile_read(Path(args.rms_root).expanduser().resolve())
    ft = read_ft(str(first_path.parent), first_path.name)

    print("\nRead result:")
    print(f"  timestamps entries: {len(ft.timestamps)}")
    if ft.timestamps:
        print(f"  first timestamp record: {ft.timestamps[0]}")
        print(f"  last timestamp record:  {ft.timestamps[-1]}")
        print("\nTime entries:")
        for index, (frame_number, timestamp) in enumerate(ft.timestamps, start=1):
            iso_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
            print(f"  {index:4d}. frame={frame_number}  time={timestamp}  iso={iso_time}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())