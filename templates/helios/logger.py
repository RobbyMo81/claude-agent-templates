import logging
from datetime import datetime

def setup_logger(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def log_epoch(epoch, train_loss, val_loss):
    logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

def log_error(error_msg):
    logging.error(f"Error: {error_msg}")
    with open("error_flag.txt", "w") as f:
        f.write(f"{datetime.now()} - {error_msg}")
