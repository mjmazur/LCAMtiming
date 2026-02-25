# LCAM Timing Analysis

A suite of tools for analyzing and optimizing MKV video frame timing against FT (Frame Time) binary timestamps. This project provides high-performance vectorized analysis, automated batch processing, and comprehensive diagnostic visualizations.

## Tools

### 1. `mkv_ft_time_offsets.py`
The primary analysis script. It compares the implied frame times of an MKV (based on its internal FPS) with the nearest corresponding FT timestamps to calculate timing offsets.

#### Key Features
- **Vectorized FPS Optimization**: Rapidly searches for the "true" FPS that minimizes timing jitter relative to frame 0.
- **Multiprocessing Batch Mode**: Efficiently processes thousands of MKVs across multiple days using a shared process pool.
- **Advanced Plotting**:
  - **Combined Offset Plots**: Visualizes multiple optimized curves, colored by file time using the `viridis` colormap.
  - **2D Density Plots**: Shows the distribution of offsets across all processed files.
  - **Statistical Histograms**: Provides distribution density (PDF), peak locations, standard deviation, and skewness for both offsets and frame rates.
  - **Summary Bands**: Visualizes mean offsets with min-max shaded regions.
- **Statistical Filtering**: Automatically discards noisy or invalid curves based on standard deviation thresholds.
- **Data Export**: Saves all optimized results to a `.pkl` file for downstream analysis.

#### Basic Usage
```bash
# Analyze a single directory (default mode)
python mkv_ft_time_offsets.py /path/to/data

# Process specific camera data with shortcut
python mkv_ft_time_offsets.py --camera-id CAWEA2 --number-of-days 7 --workers 8
```

#### Important Arguments
- `--camera-id`: Automatically sets `mkvdir` and `ftdir` to standard locations under `/mnt/RMS_data/`.
- `--number-of-days`: Specifies how many consecutive days to process in batch mode.
- `--workers`: Number of parallel processes to use during batch analysis.
- `--pickle-output`: Specifies where to save the results (defaults to the `Figures` directory).

### 2. `extract_tar_bz2.sh`
A utility script for batch-extracting compression archives common in raw data captures.

#### Usage
```bash
./extract_tar_bz2.sh /path/to/archives
```

## Requirements
- **Python 3.10+**
- **Libraries**: `numpy`, `matplotlib`, `scipy`
- **System Tools**: `ffprobe` (part of FFmpeg) must be available in your system PATH.

## Results and Diagnostics
All plots and exported results are saved to the `Figures` directory (or a custom path specified via `--plot-output`). Diagnostic plots include individual curve comparisons, combined histograms, and density representations to help identify systematic timing drifts or anomalies.
