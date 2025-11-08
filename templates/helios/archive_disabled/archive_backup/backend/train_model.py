
from logger import setup_logger, log_epoch, log_error
import torch
import pandas as pd
import json
import os

# Load config from training_config.json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "training_config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Ensure checkpoint directory exists
checkpoint_dir = os.path.join(os.path.dirname(__file__), config["checkpoint_dir"])
os.makedirs(checkpoint_dir, exist_ok=True)

# Load training data
try:
    df = pd.read_csv(config["data_source"])
    print(f"✅ Loaded training data: {len(df)} rows from {config['data_source']}")
except Exception as e:
    log_error(f"Failed to load training data: {e}")
    raise

# Example model, replace with your actual model
class DummyModel:
    def state_dict(self):
        return {}
model = DummyModel()

setup_logger("training_log.txt")

try:
    print(f"🚀 Starting training for {config['num_epochs']} epochs...")
    for epoch in range(config["num_epochs"]):
        # Simulate training
        train_loss, val_loss = 0.01 * epoch, 0.02 * epoch
        log_epoch(epoch, train_loss, val_loss)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if epoch % config["checkpoint_interval"] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

    print("✅ Training completed successfully!")
except Exception as e:
    log_error(str(e))
    print(f"❌ Training failed: {e}")
