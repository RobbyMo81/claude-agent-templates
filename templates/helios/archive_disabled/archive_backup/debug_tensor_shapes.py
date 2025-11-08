"""
Debug script to understand tensor shapes in the ML training pipeline
"""

import os
import sys
sys.path.append('backend')

import torch
import numpy as np
import pandas as pd
from agent import MLPowerballAgent
from trainer import ModelTrainer

def debug_tensor_shapes():
    """Debug the tensor shapes in the training pipeline"""
    
    print("🔍 Starting tensor shape debugging...")
    
    # Initialize trainer and load mock data
    trainer = ModelTrainer()
    historical_data = trainer.preprocessor.load_historical_data('mock')
    
    print(f"📊 Historical data shape: {historical_data.shape}")
    print(f"📊 Historical data columns: {list(historical_data.columns)}")
    print("📊 Sample data:")
    print(historical_data.head())
    
    # Initialize agent
    agent = MLPowerballAgent(sequence_length=50)
    
    print(f"🧠 Agent sequence_length: {agent.sequence_length}")
    
    # Prepare data
    (X_white, X_powerball), (y_white, y_powerball) = agent.prepare_data(historical_data)
    
    print(f"\n🎯 Input tensor shapes:")
    print(f"   X_white: {X_white.shape}")
    print(f"   X_powerball: {X_powerball.shape}")
    
    print(f"\n🎯 Target tensor shapes:")
    print(f"   y_white: {y_white.shape}")
    print(f"   y_powerball: {y_powerball.shape}")
    
    # Forward pass
    print(f"\n🔄 Running forward pass...")
    agent.model.eval()
    with torch.no_grad():
        outputs = agent.model(X_white, X_powerball)
    
    print(f"\n📤 Output tensor shapes:")
    for key, tensor in outputs.items():
        if isinstance(tensor, torch.Tensor):
            print(f"   {key}: {tensor.shape}")
    
    # Test loss calculation for first position
    print(f"\n🧮 Testing loss calculation...")
    criterion = torch.nn.CrossEntropyLoss()
    
    try:
        print(f"   outputs['white_balls'] shape: {outputs['white_balls'].shape}")
        print(f"   y_white[:, 0] shape: {y_white[:, 0].shape}")
        print(f"   y_white[:, 0] min/max: {y_white[:, 0].min()}/{y_white[:, 0].max()}")
        
        # Test the loss calculation
        loss = criterion(outputs['white_balls'], y_white[:, 0])
        print(f"   ✅ Loss calculation successful: {loss.item()}")
        
    except Exception as e:
        print(f"   ❌ Loss calculation failed: {str(e)}")
        
        # Additional debugging
        print(f"   Detailed shapes:")
        print(f"     outputs['white_balls']: {outputs['white_balls'].shape}")
        print(f"     y_white: {y_white.shape}")
        print(f"     y_white[:, 0]: {y_white[:, 0].shape}")
        
        # Check data types
        print(f"   Data types:")
        print(f"     outputs['white_balls']: {outputs['white_balls'].dtype}")
        print(f"     y_white: {y_white.dtype}")

if __name__ == "__main__":
    debug_tensor_shapes()
