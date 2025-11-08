"""
Minimal test to reproduce the ML training error
"""

import os
import sys
sys.path.append('backend')

import torch
from trainer import ModelTrainer

def test_training_api():
    """Test the training API to reproduce the error"""
    
    print("🧪 Testing training API...")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Start a training job
        result = trainer.start_training_job(
            model_name="test_model",
            data_source="mock",
            config_override={"epochs": 1}
        )
        
        print(f"✅ Training completed successfully: {result}")
        
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        print(f"   Error type: {type(e)}")
        
        # Print the full traceback
        import traceback
        print(f"   Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_training_api()
