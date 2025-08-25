import sys
import pandas as pd
from scapy.all import sniff, Dot11, IP, UDP, TCP, Dot11Deauth, Dot11Disassoc
import joblib
import warnings
import time
import numpy as np
from collections import deque

# Suppress scapy's noisy IPv6 warning
warnings.filterwarnings("ignore", category=UserWarning, module="scapy.layers.inet6")

# --- Configuration ---
MONITOR_INTERFACE = "wlan0mon"  # Change this to your monitor mode interface
BSSID_TO_MONITOR = "a0:14:3d:ed:92:14" # Change this to the BSSID of the drone
MODEL_FILE = "behavioral_drone_detector.joblib" # The trained model file
TIME_WINDOW = 0.5  # Must be the same as in train_model.py (e.g., "0.5s")

class RealTimePredictor:
    """
    This class implements a hybrid threat detection system. It uses simple rules
    for obvious attacks and a machine learning model for complex behavioral analysis.
    """
    def __init__(self, model_path):
        print(f"[*] Loading trained model from '{model_path}'...")
        try:
            self.model = joblib.load(model_path)
            # Get the feature names the model was trained on
            self.model_features = self.get_model_features()
            print(f"[*] Model loaded successfully. Expecting {len(self.model_features)} features.")
        except FileNotFoundError:
            print(f"[!] ERROR: Model file not found: '{model_path}'")
            sys.exit(1)
        except Exception as e:
            print(f"[!] ERROR: Could not load the model. {e}")
            sys.exit(1)

        # State management for behavioral analysis
        self.packet_buffer = []
        self.last_window_time = time.time()

    def get_model_features(self):
        """Extracts feature names from the model pipeline."""
        try:
            # For scikit-learn pipelines, the feature names are often stored in the first step (scaler)
            # or can be inferred if the final step is a tree-based model.
            if hasattr(self.model.steps[0][1], 'feature_names_in_'):
                 return self.model.steps[0][1].feature_names_in_
            # Fallback for some model types
            if hasattr(self.model.steps[-1][1], 'feature_names_in_'):
                 return self.model.steps[-1][1].feature_names_in_
            print("[!] Warning: Could not automatically determine feature names from model. Using hardcoded list.")
            # This hardcoded list MUST match the output of the training script
            return ['pps_mean', 'pps_max', 'pps_std', 'length_sum', 'length_mean', 
                    'src_ip_nunique', 'src_port_nunique', 'dst_port_nunique']
        except Exception as e:
            print(f"[!] Error extracting feature names: {e}")
            sys.exit(1)


    def process_packet(self, packet):
        """
        This function is the entry point for each captured packet.
        It performs rule-based checks and buffers packets for ML analysis.
        """
        # --- 1. Rule-Based Detection (Fast Path) ---
        # These are simple, high-confidence checks that don't require the ML model.

        # Rule: Deauthentication or Disassociation Attack
        if packet.haslayer(Dot11Deauth) or packet.haslayer(Dot11Disassoc):
            print(f"[!] [DEAUTH ATTACK] Deauthentication or Disassociation frame detected from {packet.addr2} to {packet.addr1}")
            return # We've made our decision, no need to buffer

        # Rule: Telnet Traffic (Unencrypted & Insecure)
        if packet.haslayer(TCP) and (packet[TCP].sport == 23 or packet[TCP].dport == 23):
             print(f"[!] [TELNET ATTACK] Insecure Telnet traffic detected: {packet[IP].src}:{packet[TCP].sport} -> {packet[IP].dst}:{packet[TCP].dport}")
             return # We've made our decision, no need to buffer

        # --- 2. Packet Buffering for Behavioral Analysis ---
        # If no simple rule was triggered, add the packet to our buffer for the ML model.
        current_time = time.time()
        packet_data = {
            'timestamp': current_time,
            'length': len(packet),
            'src_ip': packet[IP].src if packet.haslayer(IP) else None,
            'dst_ip': packet[IP].dst if packet.haslayer(IP) else None,
            'src_port': packet.sport if packet.haslayer(TCP) or packet.haslayer(UDP) else None,
            'dst_port': packet.dport if packet.haslayer(TCP) or packet.haslayer(UDP) else None,
        }
        self.packet_buffer.append(packet_data)

        # Check if the time window has elapsed
        if current_time - self.last_window_time >= TIME_WINDOW:
            self.analyze_window()
            self.packet_buffer = [] # Clear buffer for the next window
            self.last_window_time = current_time

    def analyze_window(self):
        """
        Performs feature engineering on the buffered packets and uses the ML model to predict threats.
        """
        if not self.packet_buffer:
            return

        # --- 3. Real-Time Feature Engineering ---
        # Create a pandas DataFrame from the buffer, just like in the training script
        df = pd.DataFrame(self.packet_buffer)
        
        # Calculate PPS for this window
        num_packets = len(df)
        duration = df['timestamp'].max() - df['timestamp'].min()
        if duration == 0: duration = 0.0001 # Avoid division by zero
        
        # This is a simplified PPS for the window. For the model, we need the stats.
        # The model was trained on stats of a rolling PPS, which is hard to do live.
        # We will approximate this by creating a synthetic pps column and getting stats from it.
        pps = num_packets / duration
        df['packets_per_second'] = pps # Assign the calculated PPS to all packets in the window

        # Create the exact same aggregated features the model was trained on
        features = {
            'pps_mean': df['packets_per_second'].mean(),
            'pps_max': df['packets_per_second'].max(),
            'pps_std': df['packets_per_second'].std(),
            'length_sum': df['length'].sum(),
            'length_mean': df['length'].mean(),
            'src_ip_nunique': df['src_ip'].nunique(),
            'src_port_nunique': df['src_port'].nunique(),
            'dst_port_nunique': df['dst_port'].nunique()
        }
        
        live_df = pd.DataFrame([features]).fillna(0)
        
        # Ensure the columns are in the exact same order as during training
        live_df = live_df[self.model_features]

        # --- 4. Machine Learning Prediction ---
        try:
            prediction = self.model.predict(live_df)[0]
            
            if prediction != 'normal':
                # Get the probability scores for the prediction
                probabilities = self.model.predict_proba(live_df)
                confidence = np.max(probabilities) * 100
                
                print(f"[!] [{prediction.upper()}] Detected with {confidence:.2f}% confidence. (PPS: {pps:.2f}, Unique IPs: {features['src_ip_nunique']})")
        except Exception as e:
            print(f"[!] Error during prediction: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Create an instance of our predictor class, which loads the model
    predictor = RealTimePredictor(MODEL_FILE)

    print(f"\n[*] Starting hybrid threat detection on '{MONITOR_INTERFACE}'...")
    print("[*] Monitoring for rule-based threats and behavioral anomalies.")
    print("[*] Press Ctrl+C to stop.")

    try:
        # Start sniffing packets and send each one to our predictor
        sniff(iface=MONITOR_INTERFACE, prn=predictor.process_packet, store=False)
    except KeyboardInterrupt:
        print("\n[*] Capture stopped by user.")
    except Exception as e:
        print(f"\n[!] An error occurred during capture: {e}")

