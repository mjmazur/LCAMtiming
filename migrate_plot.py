import re

path = r'c:\Users\mjmaz\Documents\GitHub\LCAMtiming\mkv_ft_time_offsets.py'
with open(path, 'r') as f:
    content = f.read()

# Pattern to match the function body of plot_combined_optimized_offsets
# We match from the signature to the start of the next function
pattern = r'(def plot_combined_optimized_offsets\(.*?output_path: Path,.*?\n)(.*?)(\n\ndef plot_optimized_offset_density)'

replacement_body = r'''    """Plot accepted optimized curves together on one combined chart."""

    # Plot all accepted optimized curves on one figure.
    import matplotlib.pyplot as plt

    if not curves:
        raise ValueError("No optimized curves available to plot")

    all_offsets_ms = np.concatenate([curve_offsets * 1000.0 for _, curve_offsets, _, _ in curves])
    min_ms = float(np.min(all_offsets_ms))
    max_ms = float(np.max(all_offsets_ms))

    # Determine time range for coloring based on the first timestamp of each curve.
    curve_times = np.array([unix_times[0] for _, _, _, unix_times in curves])
    t_min, t_max = np.min(curve_times), np.max(curve_times)
    
    import matplotlib.colors as mcolors
    if np.isclose(t_min, t_max):
        norm = mcolors.Normalize(t_min - 1, t_max + 1)
    else:
        norm = mcolors.Normalize(t_min, t_max)
    cmap = plt.get_cmap("viridis")

    if np.isclose(min_ms, max_ms):
        pad_ms = max(1.0, abs(min_ms) * 0.05)
    else:
        pad_ms = (max_ms - min_ms) * 0.05

    y_min = min_ms - pad_ms
    y_max = max_ms + pad_ms

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
    ax.set_title(f"Combined accepted optimized curves ({len(curves)} files)")
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
'''

# Use lambda or separate components to avoid backslash issues in replacement string if any
new_content = re.sub(pattern, lambda m: m.group(1) + replacement_body + m.group(3), content, flags=re.DOTALL)

with open(path, 'w') as f:
    f.write(new_content)
