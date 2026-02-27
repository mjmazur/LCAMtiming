#!/usr/bin/env python3

import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def parse_log_file(filepath):
    dates = []
    offsets = []
    drifts = []
    jitters = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            # Skip short lines or lines that don't look like data (e.g. headers)
            if len(parts) < 10:
                continue

            try:
                # Based on our schema:
                # 0: Date (YYYY-MM-DD)
                # 1: Time (HH:MM:SS)
                # 4: Frequency (ppm) [Drift]
                # 6: Offset (seconds)
                # 9: Std Dev (seconds) [Jitter]

                date_str = parts[0]
                time_str = parts[1]

                # Check if date format is valid to filter out header lines
                datetime.datetime.strptime(date_str, "%Y-%m-%d")

                dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

                frequency = float(parts[4])
                offset = float(parts[6])
                std_dev = float(parts[9])

                dates.append(dt)
                offsets.append(offset)
                drifts.append(frequency)
                jitters.append(std_dev)

            except ValueError:
                continue

    return dates, offsets, drifts, jitters

def plot_data(dates, offsets, drifts, jitters, site_name, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot Offset
    axes[0].plot(dates, offsets, label='Offset (s)', color='blue', linewidth=0.8)
    axes[0].set_ylabel('Offset (s)')
    axes[0].set_title(f'Clock Offset - {site_name}')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # Plot Drift (Frequency)
    axes[1].plot(dates, drifts, label='Frequency (ppm)', color='green', linewidth=0.8)
    axes[1].set_ylabel('Frequency (ppm)')
    axes[1].set_title(f'Clock Drift (Frequency) - {site_name}')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    # Plot Jitter (Std Dev)
    axes[2].plot(dates, jitters, label='Std Dev (s)', color='red', linewidth=0.8)
    axes[2].set_ylabel('Std Dev (s)')
    axes[2].set_xlabel('Time')
    axes[2].set_title(f'Jitter (Std Dev) - {site_name}')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    plt.tight_layout()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'chrony_analysis_{site_name}.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")

def print_stats(site_name, dates, offsets, drifts, jitters):
    if not dates:
        print(f"No data for {site_name}")
        return

    print(f"\nStats for {site_name}:")
    print(f"  Entries: {len(dates)}")
    print(f"  Offset: Mean={np.mean(offsets):.6e}, Min={np.min(offsets):.6e}, Max={np.max(offsets):.6e}, Std={np.std(offsets):.6e}")
    print(f"  Drift:  Mean={np.mean(drifts):.6f}, Min={np.min(drifts):.6f}, Max={np.max(drifts):.6f}, Std={np.std(drifts):.6f}")
    print(f"  Jitter: Mean={np.mean(jitters):.6e}, Min={np.min(jitters):.6e}, Max={np.max(jitters):.6e}, Std={np.std(jitters):.6e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Chrony tracking logs and plot metrics.")
    parser.add_argument("--log-dir", default="ChronyLogs", help="Directory containing site subdirectories with tracking.log files")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    figures_dir = Path("Figures")

    if not log_dir.exists():
        print(f"Error: Log directory '{log_dir}' not found.")
        return

    # Walk through the directory structure
    found_logs = False
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file == "tracking.log":
                found_logs = True
                filepath = Path(root) / file

                # Use the parent directory name as the site name
                site_name = filepath.parent.name

                print(f"Processing {site_name}...")
                dates, offsets, drifts, jitters = parse_log_file(filepath)

                if dates:
                    plot_data(dates, offsets, drifts, jitters, site_name, figures_dir)
                    print_stats(site_name, dates, offsets, drifts, jitters)
                else:
                    print(f"Warning: No valid data found in {filepath}")

    if not found_logs:
        print(f"No tracking.log files found in subdirectories of {log_dir}")

if __name__ == "__main__":
    main()
