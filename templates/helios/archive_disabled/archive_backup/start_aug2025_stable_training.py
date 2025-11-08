#!/usr/bin/env python3
"""
Stable training starter for PowerBall Aug 2025 dataset with conservative settings
"""

import sys
import os
import json
import time
from datetime import datetime

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

try:
    from trainer import ModelTrainer, TrainingConfig
    print("✅ Successfully imported training modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def start_stable_training():
    """Start training with stable, conservative settings"""
    
    # Load configuration
    config_file = os.path.join(os.path.dirname(__file__), 'start_training_aug2025_stable.json')
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    print("🎯 Starting Aug 2025 Stable PowerBall Training")
    print(f"📊 Data Source: {config_data['dataSource']}")
    print(f"🔄 Epochs: {config_data['epochs']}")
    print(f"📦 Batch Size: {config_data['batchSize']} (Conservative)")
    print(f"📏 Sequence Length: {config_data['sequenceLength']} (Memory Optimized)")
    
    try:
        # Create config override dictionary with conservative settings
        config_override = {
            'epochs': config_data['epochs'],
            'learning_rate': config_data['learningRate'],
            'batch_size': config_data['batchSize'],
            'sequence_length': config_data['sequenceLength'],
            'validation_split': config_data['validationSplit'],
            'early_stopping_patience': config_data['earlyStoppingPatience'],
            'min_delta': config_data['minDelta'],
            'save_best_only': config_data['saveBestOnly']
        }
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        print(f"🚀 Starting stable training job for model: {config_data['modelName']}")
        print("⚡ Using conservative settings to ensure memory stability")
        
        # Start training
        result = trainer.start_training_job(
            model_name=config_data['modelName'],
            data_source=config_data['dataSource'],
            config_override=config_override
        )
        
        print("✅ Training completed successfully!")
        print(f"📈 Final Result: {result}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_stable_training()
