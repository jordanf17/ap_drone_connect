import pandas as pd
import os
import re
import numpy as np

# --- Configuration ---
# PPS (Packets Per Second) window size for the rolling average.
PPS_WINDOW_SIZE = 20
# PPS threshold to flag a potential flood attack.
FLOOD_THRESHOLD = 1000
# Number of unique source IPs required to classify a TCP flood as DDoS.
DDOS_IP_THRESHOLD = 10

# --- Feature Engineering and Labeling Functions ---

def extract_ports(info_series):
    """
    Extracts source and destination ports from the 'info' column using regex.
    Handles various packet info formats.
    """
    # Regex to find patterns like '1.2.3.4:port -> 5.6.7.8:port'
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
        
    # Ensure data is sorted by timestamp for accurate calculation
    df_sorted = df.sort_values(by='timestamp').copy()
    # Calculate time difference between consecutive packets
    time_diffs = df_sorted['timestamp'].diff().fillna(0)
    # Calculate instantaneous PPS, handling potential division by zero
    instant_pps = 1.0 / time_diffs.replace(0, np.nan)
    # Calculate rolling average PPS
    rolling_pps = instant_pps.rolling(window=PPS_WINDOW_SIZE, min_periods=1).mean()
    # Fill initial NaN values with the first valid PPS calculation
    rolling_pps = rolling_pps.bfill().ffill().fillna(0)
    
    # Add the result back to the original dataframe index
    df['packets_per_second'] = rolling_pps
    return df

def assign_attack_label(row):
    """
    Assigns an attack label to a packet based on a set of rules.
    The logic prioritizes specific attacks before labeling general flood attacks.
    """
    # Rule 1: Telnet Attack (insecure protocol)
    if row['src_port'] == 23 or row['dst_port'] == 23:
        return 'telnet_attack'

    # Rule 2: Deauthentication/Disassociation Attack
    if isinstance(row['info'], str):
        info_lower = row['info'].lower()
        if 'deauth' in info_lower or 'disassoc' in info_lower:
            return 'deauth_attack'

    # Rule 3: Check for specific, known flood attack protocols if PPS is high
    if row['packets_per_second'] > FLOOD_THRESHOLD:
        protocol = str(row.get('protocol', '')).upper()
        
        if 'ICMP' in protocol:
            return 'icmp_flood'
        
        if 'UDP' in protocol:
            return 'udp_flood'
        
        # A high rate of TCP packets is a flood, but we need more context (DoS vs DDoS)
        # So we give it a temporary label to be refined later.
        if 'TCP' in protocol:
            return 'tcp_flood'
            
    return 'normal'

# --- Main Execution ---
if __name__ == "__main__":
    # The script will run in the directory where it is placed.
    # It will look for an 'attacks' subdirectory.
    script_directory = os.path.dirname(os.path.abspath(__file__))
    attacks_directory = os.path.join(script_directory, 'attacks')
    
    print(f"[*] Starting in directory: {script_directory}")
    print(f"[*] Looking for attack folders in: {attacks_directory}")
    print("-" * 30)

    if not os.path.isdir(attacks_directory):
        print(f"[!] Error: 'attacks' subdirectory not found in {script_directory}")
        exit()

    # Find all subdirectories in the 'attacks' directory
    attack_folders = [f.path for f in os.scandir(attacks_directory) if f.is_dir()]

    if not attack_folders:
        print("[!] No attack subdirectories found to process.")

    # Loop through each attack subdirectory
    for folder in attack_folders:
        # Find the first .csv file in the directory
        csv_file = None
        for file in os.listdir(folder):
            if file.endswith('.csv') and not file.endswith('_labeled.csv'):
                csv_file = file
                break # Process the first one we find

        if csv_file:
            input_filepath = os.path.join(folder, csv_file)
            base_name = os.path.splitext(csv_file)[0]
            output_filename = f"{base_name}_labeled.csv"
            output_filepath = os.path.join(folder, output_filename)

            try:
                print(f"[*] Processing: {input_filepath}")
                df = pd.read_csv(input_filepath)

                # --- 1. Engineer New Features ---
                print("    -> Calculating packets per second...")
                df = calculate_pps(df)
                
                print("    -> Extracting port numbers...")
                df['src_port'], df['dst_port'] = extract_ports(df['info'])

                # --- 2. Apply Initial Attack Labels ---
                print("    -> Assigning initial attack labels...")
                df['attack_label'] = df.apply(assign_attack_label, axis=1)

                # --- 3. Refine TCP Flood labels (DoS vs DDoS) ---
                if 'tcp_flood' in df['attack_label'].unique():
                    print("    -> Analyzing TCP flood type...")
                    # Count unique source IPs only within the TCP flood packets
                    flood_ips = df[df['attack_label'] == 'tcp_flood']['src_ip'].nunique()
                    
                    if flood_ips > DDOS_IP_THRESHOLD:
                        final_label = 'ddos_attack'
                        print(f"    -> Found {flood_ips} unique IPs. Classifying as DDoS.")
                    else:
                        final_label = 'dos_attack'
                        print(f"    -> Found {flood_ips} unique IPs. Classifying as DoS.")
                    
                    # Replace the temporary label with the final, correct one
                    df.loc[df['attack_label'] == 'tcp_flood', 'attack_label'] = final_label

                # --- 4. Save the final output ---
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
        else:
            print(f"[*] Skipping folder: {os.path.basename(folder)} (No .csv file found)")
            print("-" * 30)

    print("[*] All folders processed.")
