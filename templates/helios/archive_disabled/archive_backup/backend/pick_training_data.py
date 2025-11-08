import os
import json

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "training_config.json")
DATA_DIR = r"C:\Users\RobMo\OneDrive\Documents\PowerBall\downloads"

# List available CSV files
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
if not csv_files:
    print("No CSV files found in data directory.")
    exit(1)

print("Available datasets:")
for idx, fname in enumerate(csv_files):
    print(f"[{idx+1}] {fname}")

choice = input(f"Select dataset [1-{len(csv_files)}]: ")
try:
    selected_idx = int(choice) - 1
    selected_file = csv_files[selected_idx]
except (ValueError, IndexError):
    print("Invalid selection.")
    exit(1)

selected_path = os.path.join(DATA_DIR, selected_file)

# Update config file
def update_config(path):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    config["data_source"] = path
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Config updated: data_source = {path}")

update_config(selected_path)
