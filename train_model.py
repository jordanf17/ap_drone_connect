import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# --- Configuration ---
BASE_DATA_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_OUTPUT_FILE = "behavioral_drone_detector.joblib"
TIME_WINDOW = "0.5s"

def create_behavioral_features(df):
    """
    Transforms a raw packet DataFrame into a behavioral profile, now including
    protocol counts to better distinguish attack types.
    """
    # --- 1. Prepare DataFrame ---
    required_cols = ['timestamp', 'length', 'src_ip', 'protocol', 'src_port', 'dst_port', 'packets_per_second', 'attack_label']
    for col in required_cols:
        if col not in df.columns:
            print(f"    [!] Warning: Missing essential column '{col}'. Skipping file.")
            return None
    
    df[['length', 'src_port', 'dst_port', 'packets_per_second']] = df[['length', 'src_port', 'dst_port', 'packets_per_second']].fillna(0)
    df['protocol'] = df['protocol'].fillna('Other')

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df = df.set_index('timestamp').sort_index()

    # --- 2. Create Protocol Dummies for Counting ---
    # This allows us to count each protocol type within the time window
    protocol_dummies = pd.get_dummies(df['protocol'], prefix='protocol')
    df = pd.concat([df, protocol_dummies], axis=1)

    # --- 3. Define Advanced Aggregations ---
    agg_dict = {
        'packets_per_second': ['mean', 'max', 'std'],
        'length': ['sum', 'mean'],
        'src_ip': ['nunique'],
        'src_port': ['nunique'],
        'dst_port': ['nunique']
    }
    # Add the protocol dummy columns to our aggregation dictionary to be summed
    for col in protocol_dummies.columns:
        agg_dict[col] = 'sum'

    windowed_df = df.groupby(pd.Grouper(freq=TIME_WINDOW)).agg(agg_dict)
    
    windowed_df.columns = ['_'.join(col).strip() for col in windowed_df.columns.values]
    windowed_df.rename(columns={'packets_per_second_mean': 'pps_mean', 
                                'packets_per_second_max': 'pps_max',
                                'packets_per_second_std': 'pps_std'}, inplace=True)

    # --- 4. Determine Window Label ---
    def aggregate_labels(series):
        series = series.dropna()
        attack_labels = series[series != 'normal']
        return attack_labels.mode().iloc[0] if not attack_labels.empty else 'normal'
        
    windowed_labels = df['attack_label'].groupby(pd.Grouper(freq=TIME_WINDOW)).apply(aggregate_labels)
    
    final_df = pd.concat([windowed_df, windowed_labels], axis=1).fillna(0)
    final_df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return final_df

def train_and_save_model(df, output_path):
    """Trains a model on the behavioral features and saves it."""
    print("\n--- Final Label distribution for training ---")
    print(df['attack_label'].value_counts())
    print("-------------------------------------------\n")

    if len(df['attack_label'].unique()) < 2:
        print(f"[!] Error: Only one class '{df['attack_label'].unique()[0]}' was found.")
        return

    print("[*] Preparing and training advanced behavioral model...")
    features = [col for col in df.columns if col != 'attack_label']
    target = 'attack_label'
    X = df[features]
    y = df[target]

    model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model_pipeline.fit(X_train, y_train)
    
    print("\n--- Model Evaluation ---")
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[*] Model Accuracy: {accuracy * 100:.2f}%")
    print("\n[*] Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print(f"\n[*] Saving trained model to '{output_path}'...")
    joblib.dump(model_pipeline, output_path)
    print("[*] Model saved successfully.")

if __name__ == "__main__":
    all_featured_dfs = []
    
    print(f"[*] Searching for labeled data in subdirectories of: {BASE_DATA_PATH}")
    
    attacks_path = os.path.join(BASE_DATA_PATH, 'attacks')
    normal_path = os.path.join(BASE_DATA_PATH, 'normal')
    file_paths_to_process = []

    if os.path.isdir(attacks_path):
        for subdir, _, files in os.walk(attacks_path):
            for file in files:
                if file.endswith('_labeled.csv'):
                    file_paths_to_process.append(os.path.join(subdir, file))

    if os.path.isdir(normal_path):
        for file in os.listdir(normal_path):
             if file.endswith('_labeled.csv'):
                file_paths_to_process.append(os.path.join(normal_path, file))

    for filepath in file_paths_to_process:
        print(f"\n[*] Processing file: {filepath}")
        try:
            df = pd.read_csv(filepath, low_memory=False)
            featured_df = create_behavioral_features(df)
            if featured_df is not None and not featured_df.empty:
                all_featured_dfs.append(featured_df)
            else:
                print(f"    [!] Skipping file {filepath} due to processing errors or empty result.")
        except Exception as e:
            print(f"    [!] FAILED to process {filepath}: {e}")

    if not all_featured_dfs:
        print("\n[!] No data could be processed. Cannot train model. Exiting.")
        sys.exit(1)

    final_data = pd.concat(all_featured_dfs).fillna(0)
    train_and_save_model(final_data, MODEL_OUTPUT_FILE)
