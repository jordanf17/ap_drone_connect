import pandas as pd
import os
import re
import numpy as np

# --- Configuration ---
# PPS (Packets Per Second) window size for the rolling average.
PPS_WINDOW_SIZE = 20

# --- Feature Engineering Functions (copied for consistency) ---

def extract_ports(info_series):
    """
    Extracts source and destination ports from the 'info' column using regex.
    """
    port_pattern = re.compile(r'.*?:(\d{1,5})\s*->\s*.*?:(\d{1,5})')
    
    def parse_info(info_str):
        if not isinstance(info_str, str):
            return None, None
        match = port_pattern.search(info_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    ports = info_series.apply(parse_info)
    src_ports = ports.apply(lambda x: x[0])
    dst_ports = ports.apply(lambda x: x[1])
    return src_ports, dst_ports

def calculate_pps(df):
    """
    Calculates the packets per second using a rolling window average.
    """
    if 'timestamp' not in df.columns:
        print("    [!] 'timestamp' column not found. Skipping PPS calculation.")
        df['packets_per_second'] = 0
        return df
        
    df_sorted = df.sort_values(by='timestamp').copy()
    time_diffs = df_sorted['timestamp'].diff().fillna(0)
    instant_pps = 1.0 / time_diffs.replace(0, np.nan)
    rolling_pps = instant_pps.rolling(window=PPS_WINDOW_SIZE, min_periods=1).mean()
    rolling_pps = rolling_pps.bfill().ffill().fillna(0)
    
    df['packets_per_second'] = rolling_pps
    return df

# --- Main Execution ---
if __name__ == "__main__":
    # The script will run in the directory where it is placed.
    # It will look for a 'normal' subdirectory.
    script_directory = os.path.dirname(os.path.abspath(__file__))
    normal_directory = os.path.join(script_directory, 'normal')
    
    print(f"[*] Starting in directory: {script_directory}")
    print(f"[*] Looking for normal capture files in subdirectories of: {normal_directory}")
    print("-" * 30)

    if not os.path.isdir(normal_directory):
        print(f"[!] Error: 'normal' subdirectory not found in {script_directory}")
        exit()

    # Recursively find all .csv files in the 'normal' directory and its subdirectories
    for root, dirs, files in os.walk(normal_directory):
        for file in files:
            if file.endswith('.csv') and not file.endswith('_labeled.csv'):
                input_filepath = os.path.join(root, file)
                # Get the name of the parent folder (e.g., 'Square_flight')
                subfolder_name = os.path.basename(root)
                # Create a new, descriptive name for the output file
                output_filename = f"{subfolder_name}_labeled.csv"
                # Save the labeled file in the top-level 'normal' directory
                output_filepath = os.path.join(normal_directory, output_filename)

                try:
                    print(f"[*] Processing: {input_filepath}")
                    df = pd.read_csv(input_filepath)

                    # --- 1. Engineer New Features for consistency ---
                    print("    -> Calculating packets per second...")
                    df = calculate_pps(df)
                    
                    print("    -> Extracting port numbers...")
                    df['src_port'], df['dst_port'] = extract_ports(df['info'])

                    # --- 2. Apply 'normal' Label ---
                    print("    -> Assigning 'normal' label to all packets...")
                    df['attack_label'] = 'normal'

                    # --- 3. Save the final output ---
                    df.to_csv(output_filepath, index=False)
                    
                    print(f"    -> Success! Labeled data saved to '{output_filepath}'")
                    print(f"    -> Total rows processed: {len(df)}")
                    
                    label_counts = df['attack_label'].value_counts()
                    print("    -> Label Summary:")
                    for label, count in label_counts.items():
                        print(f"        - {label}: {count} packets")

                    print("-" * 30)

                except Exception as e:
                    print(f"    -> [!] ERROR processing {input_filepath}: {e}")
                    print("-" * 30)

    print("[*] All normal files processed.")
