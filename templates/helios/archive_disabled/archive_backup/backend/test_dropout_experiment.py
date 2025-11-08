"""
Quick Dropout Increase Experiment
Test increased dropout (0.4) for better regularization.
"""

from full_training_suite import PowerballTrainer
from logger import setup_logger
import json
import os
import os

def main():
    # Load dropout experiment config
    config_path = os.path.join(os.path.dirname(__file__), "optimization_configs", "config_dropout_increase.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"🧪 Running Dropout Increase Experiment (dropout=0.4)")
    setup_logger("logs/dropout_experiment.txt")
    
    # Run training
    trainer = PowerballTrainer(config)
    train_losses, val_losses = trainer.train()
    
    print(f"\n📊 DROPOUT EXPERIMENT RESULTS:")
    print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
    print(f"Total Epochs Completed: {len(train_losses)}")
    print(f"Early Stopping: {'Yes' if len(train_losses) < config['num_epochs'] else 'No'}")

if __name__ == "__main__":
    main()
