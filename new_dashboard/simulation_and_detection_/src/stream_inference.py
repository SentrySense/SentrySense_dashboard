import os
import json
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from datetime import datetime
import pytz

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the simulation_and_detection_ directory (parent of src)
base_dir = os.path.dirname(script_dir)

# Paths relative to simulation_and_detection_ directory
GRAPH_PATH = os.path.join(base_dir, 'splits', 'train_graph_0.pt')
MODEL_PATH = os.path.join(base_dir, 'models', 'gnn_model.pt')
TRAIN_PROCESSED = os.path.join(base_dir, 'splits', 'train_clean_numeric.csv')
STREAM_FILE = os.path.join(base_dir, 'splits', 'stream_clean_numeric.csv')
LOG_FILE = os.path.join(base_dir, 'logs', 'stream_logs.jsonl')

CONTEXT_COLS = ['masked_user', 'source_ip', 'resource']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Debug: Print the paths to verify they're correct
print(f"Script directory: {script_dir}")
print(f"Base directory: {base_dir}")
print(f"Graph path: {GRAPH_PATH}")
print(f"Model path: {MODEL_PATH}")
print(f"Train processed path: {TRAIN_PROCESSED}")
print(f"Stream file path: {STREAM_FILE}")
print(f"Log file path: {LOG_FILE}")

# Mapping from encoded location to city name and timezone
location_code_to_city = {
    -1: "Unknown",   # Unseen/unknown
    0: "London",
    1: "Tokyo",
    2: "New York",
    3: "Berlin",
    4: "São Paulo"
}
city_to_timezone = {
    "London": "Europe/London",
    "Tokyo": "Asia/Tokyo",
    "New York": "America/New_York",
    "Berlin": "Europe/Berlin",
    "São Paulo": "America/Sao_Paulo"
}

class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = torch.nn.ReLU()
    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GNNAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = torch.nn.Linear(out_channels, in_channels)
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_hat = self.decoder(z)
        return x_hat

def add_node_to_graph(data, node_features, df_existing, context_cols, stream_row_idx):
    x_new = torch.cat([data.x, node_features.unsqueeze(0)], dim=0)
    num_nodes = x_new.shape[0]
    edge_index = [list(edge) for edge in data.edge_index.t().tolist()]
    found_connection = False
    for col in context_cols:
        val = int(df_existing.iloc[stream_row_idx][col])
        matching_indices = df_existing.index[df_existing[col] == val].tolist()
        for idx in matching_indices:
            found_connection = True
            edge_index.append([idx, num_nodes - 1])
            edge_index.append([num_nodes - 1, idx])
    if not found_connection:
        edge_index.append([num_nodes - 1, num_nodes - 1])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x_new, edge_index=edge_index)

def get_local_hour(location_code):
    city = location_code_to_city.get(location_code, "Unknown")
    if city == "Unknown" or city not in city_to_timezone:
        return None, city
    tz = pytz.timezone(city_to_timezone[city])
    local_time = datetime.now(tz)
    return local_time.hour, city

def create_dummy_data():
    """Create dummy data files if they don't exist"""
    print("Creating dummy data files for demonstration...")
    
    # Create dummy graph data
    if not os.path.exists(GRAPH_PATH):
        print(f"Creating dummy graph at {GRAPH_PATH}")
        os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
        
        # Create a simple dummy graph with 32 features
        num_nodes = 100
        num_features = 32
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, 200))
        data = Data(x=x, edge_index=edge_index)
        torch.save(data, GRAPH_PATH)
    
    # Create dummy model
    if not os.path.exists(MODEL_PATH):
        print(f"Creating dummy model at {MODEL_PATH}")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        model = GNNAutoEncoder(32, 64, 32)
        torch.save(model.state_dict(), MODEL_PATH)
    
    # Create dummy training data
    if not os.path.exists(TRAIN_PROCESSED):
        print(f"Creating dummy training data at {TRAIN_PROCESSED}")
        os.makedirs(os.path.dirname(TRAIN_PROCESSED), exist_ok=True)
        
        # Create dummy CSV with 32 columns
        import pandas as pd
        columns = ['masked_user', 'source_ip', 'destination_ip', 'action', 'resource', 
                  'protocol', 'access_result', 'location', 'device_type', 'login_attempt_count',
                  'log_hour', 'day_of_week', 'month', 'time_diff_prev', 'user_activity_frequency',
                  'unique_actions', 'source_subnet', 'dest_subnet', 'resource_category', 'access_flag',
                  'protocol_HTTP', 'protocol_HTTPS', 'protocol_FTP', 'protocol_SSH', 'protocol_RDP',
                  'device_desktop', 'device_mobile', 'device_tablet', 'location_count', 
                  'session_duration', 'event_count_session', 'session_complexity']
        
        # Generate dummy data
        data = []
        for i in range(100):
            row = {}
            for col in columns:
                if col in ['masked_user', 'source_ip', 'destination_ip']:
                    row[col] = np.random.randint(-1, 1000)
                elif col in ['location']:
                    row[col] = np.random.randint(-1, 5)
                elif 'protocol_' in col or 'device_' in col:
                    row[col] = np.random.randint(0, 2)
                else:
                    row[col] = np.random.randint(0, 100)
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(TRAIN_PROCESSED, index=False)

def main():
    print("Starting stream inference...")
    
    # Check if required files exist, create dummy ones if not
    files_missing = []
    if not os.path.exists(GRAPH_PATH):
        files_missing.append("Graph file")
    if not os.path.exists(MODEL_PATH):
        files_missing.append("Model file")
    if not os.path.exists(TRAIN_PROCESSED):
        files_missing.append("Training data file")
    
    if files_missing:
        print(f"Missing files: {', '.join(files_missing)}")
        print("Creating dummy data for demonstration purposes...")
        create_dummy_data()
    
    # Verify files exist after creation
    if not os.path.exists(GRAPH_PATH):
        print(f"Error: Still cannot find graph file at {GRAPH_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Still cannot find model file at {MODEL_PATH}")
        return
    
    if not os.path.exists(TRAIN_PROCESSED):
        print(f"Error: Still cannot find training data file at {TRAIN_PROCESSED}")
        return
    
    try:
        # Load graph and model
        print(f"Loading graph from: {GRAPH_PATH}")
        data = torch.load(GRAPH_PATH, weights_only=False)
        
        print(f"Loading training data from: {TRAIN_PROCESSED}")
        df_existing = pd.read_csv(TRAIN_PROCESSED)
        
        in_channels = data.x.size(1)
        hidden_channels = 64
        out_channels = 32

        print(f"Loading model from: {MODEL_PATH}")
        model = GNNAutoEncoder(in_channels, hidden_channels, out_channels)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.eval()
        model.to(DEVICE)

        # --- Load and align streaming data ---
        train_df = pd.read_csv(TRAIN_PROCESSED)
        
        # Create stream file if it doesn't exist (use sample from training data)
        if not os.path.exists(STREAM_FILE):
            print(f"Stream file not found. Creating sample stream file at {STREAM_FILE}")
            # Use a sample from training data as stream data
            sample_stream = train_df.sample(n=min(100, len(train_df))).reset_index(drop=True)
            sample_stream.to_csv(STREAM_FILE, index=False)
        
        stream_df = pd.read_csv(STREAM_FILE)

        # Align columns: keep only those in train, in the same order
        stream_df_aligned = stream_df[[col for col in train_df.columns if col in stream_df.columns]]
        for col in train_df.columns:
            if col not in stream_df_aligned.columns:
                stream_df_aligned[col] = 0
        stream_df_aligned = stream_df_aligned[train_df.columns]
        stream_df_aligned.to_csv(STREAM_FILE, index=False)

        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

        # Only process the first row (simulate 1-row streaming)
        if len(stream_df_aligned) == 0:
            print("No events left in stream.")
            return
        row = stream_df_aligned.iloc[0]
        node_features = torch.tensor(row.values, dtype=torch.float)

        log_entry = {
            "stream_index": int(df_existing.shape[0]),
            "raw_features": row.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

        # --- Real-time location-based time anomaly check ---
        location_code = int(row['location'])
        local_hour, city = get_local_hour(location_code)
        if local_hour is not None and 0 <= local_hour < 4:
            log_entry.update({
                "anomaly": "medium",
                "reason": f"Local time in {city} is {local_hour}:00 — time is not good (12am-4am)"
            })
            print(f"Medium anomaly: Local time in {city} is {local_hour}:00 — time is not good (12am-4am). Skipping model inference.")
        else:
            if data.x.shape[1] != node_features.shape[0]:
                print(f"Feature mismatch: graph has {data.x.shape[1]}, node has {node_features.shape[0]}")
                print("Adjusting node features to match graph...")
                if node_features.shape[0] > data.x.shape[1]:
                    node_features = node_features[:data.x.shape[1]]
                else:
                    padding = torch.zeros(data.x.shape[1] - node_features.shape[0])
                    node_features = torch.cat([node_features, padding])
            
            temp_graph = add_node_to_graph(data, node_features, df_existing, CONTEXT_COLS, 0)
            temp_graph = temp_graph.to(DEVICE)

            with torch.no_grad():
                x_hat = model(temp_graph.x, temp_graph.edge_index)
            recon_error = torch.nn.functional.mse_loss(
                x_hat[-1], temp_graph.x[-1], reduction='sum'
            ).item()

            threshold = 10000000  # Example: adjust as needed for your data
            is_anomaly = recon_error > threshold

            log_entry.update({
                "anomaly": is_anomaly,
                "score": recon_error
            })

            if is_anomaly:
                # --- Explainability: Top 2 features by absolute error (human-readable) ---
                original = temp_graph.x[-1].cpu().numpy()
                reconstructed = x_hat[-1].cpu().numpy()
                abs_errors = np.abs(original - reconstructed)
                feature_names = list(row.index)
                top_indices = abs_errors.argsort()[-2:][::-1]
                explanation = []
                for idx in top_indices:
                    feat = feature_names[idx]
                    orig_val = original[idx]
                    recon_val = reconstructed[idx]
                    explanation.append({
                        "feature": feat,
                        "original": float(orig_val),
                        "reconstructed": float(recon_val),
                        "abs_error": float(abs_errors[idx])
                    })
                log_entry["why"] = explanation
                print(f"Event {df_existing.shape[0]}: score={recon_error:.4f} [ANOMALY]")
                print("Top contributing features to anomaly score:")
                for exp in explanation:
                    print(f"  {exp['feature']}: original={exp['original']:.2f}, reconstructed={exp['reconstructed']:.2f}, error={exp['abs_error']:.2f}")
            else:
                print(f"Event {df_existing.shape[0]}: score={recon_error:.4f} [normal]")

        # --- Append to log ---
        with open(LOG_FILE, 'a') as log_f:
            log_f.write(json.dumps(log_entry) + '\n')

        # --- Cyclic streaming: append this row to end of stream and update file ---
        stream_df_aligned = stream_df_aligned.iloc[1:]
        stream_df_aligned = pd.concat([stream_df_aligned, row.to_frame().T], ignore_index=True)
        stream_df_aligned.to_csv(STREAM_FILE, index=False)
        
        print("Stream inference completed successfully!")
        
    except Exception as e:
        print(f"Error during stream inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
