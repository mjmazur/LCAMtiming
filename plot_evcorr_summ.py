import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Process evcorr summ.txt files.")
    parser.add_argument(
        "--klingon-dir",
        type=str,
        default="/srv/meteor/klingon/evcorr",
        help="Directory containing klingon evcorr data (default: /srv/meteor/klingon/evcorr)",
    )
    parser.add_argument(
        "--romulan-dir",
        type=str,
        default="/srv/meteor/romulan/evcorr",
        help="Directory containing romulan evcorr data (default: /srv/meteor/romulan/evcorr)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Figures",
        help="Directory to save the plots (default: Figures)",
    )
    return parser.parse_args()

def load_and_process_data(base_dir):
    """Finds all summ.txt in base_dir, loads them into a DataFrame, and sorts by date/time."""
    search_pattern = os.path.join(base_dir, "**", "summ.txt")
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        print(f"No summ.txt files found in {base_dir}")
        return pd.DataFrame()

    dfs = []
    for file in files:
        try:
            # Assuming space-separated or varying whitespace
            df = pd.read_csv(file, sep=r'\s+')
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by date and time
    if 'date' in combined_df.columns and 'time' in combined_df.columns:
        combined_df = combined_df.sort_values(by=['date', 'time'])
    else:
        print(f"Warning: 'date' and/or 'time' column not found in data from {base_dir}. Cannot sort.")

    return combined_df

def plot_data(df, name, output_dir):
    """Creates the required plots for a given DataFrame."""
    if df.empty:
        print(f"DataFrame for {name} is empty, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Plot H_beg vs speed
    if 'H_beg' in df.columns and 'speed' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['speed'], df['H_beg'], alpha=0.5)
        plt.title(f"{name}: H_beg vs speed")
        plt.xlabel("speed")
        plt.ylabel("H_beg")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{name}_H_beg_vs_speed.png"))
        plt.close()
    else:
        print(f"Warning: 'H_beg' and/or 'speed' column missing in {name} data.")

    # Plot mag vs vel
    if 'mag' in df.columns and 'vel' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['vel'], df['mag'], alpha=0.5)
        plt.title(f"{name}: mag vs vel")
        plt.xlabel("vel")
        plt.ylabel("mag")
        plt.grid(True)
        # Often magnitudes are plotted inverted, but the instructions say just "mag vs vel"
        plt.savefig(os.path.join(output_dir, f"{name}_mag_vs_vel.png"))
        plt.close()
    else:
        print(f"Warning: 'mag' and/or 'vel' column missing in {name} data.")


def main():
    args = parse_args()

    print(f"Processing Klingon data from {args.klingon_dir}")
    klingon_df = load_and_process_data(args.klingon_dir)
    print(f"Klingon data shape: {klingon_df.shape}")

    print(f"Processing Romulan data from {args.romulan_dir}")
    romulan_df = load_and_process_data(args.romulan_dir)
    print(f"Romulan data shape: {romulan_df.shape}")

    print(f"Saving plots to {args.output_dir}")
    plot_data(klingon_df, "Klingon", args.output_dir)
    plot_data(romulan_df, "Romulan", args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
