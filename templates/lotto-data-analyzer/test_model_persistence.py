#!/usr/bin/env python3
"""
Model Persistence Validation Script
----------------------------------
Tests the Phase 5 model persistence implementation to ensure models are saved
to disk and automatically loaded on startup.
"""

import sys
import os
import logging
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append('.')

from core.model_training_service import ModelTrainingService
from core.persistent_model_predictions import get_current_model_paths
from core import store

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_persistence():
    """Test complete model persistence workflow."""
    
    print("\n=== Model Persistence Validation Test ===\n")
    
    # Step 1: Initialize service
    print("1. Initializing ModelTrainingService...")
    service = ModelTrainingService()
    
    # Step 2: Check models directory exists
    models_dir = Path("models")
    print(f"2. Models directory exists: {models_dir.exists()}")
    
    # Step 3: Get test data
    print("3. Loading test data...")
    try:
        df = store.latest()
        if df is None or df.empty:
            print("   No powerball data available for testing")
            return False
        print(f"   Loaded {len(df)} records")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return False
    
    # Step 4: Train a single model
    print("4. Training Ridge Regression model...")
    try:
        results = service.train_models(df, ["Ridge Regression"])
        result = results[0] if results else {}
        if not result.get('training_completed', False):
            print(f"   Training failed: {result.get('error', 'Unknown error')}")
            return False
        
        model_artifact_path = result.get('model_artifact_path')
        print(f"   Model saved to: {model_artifact_path}")
        
        if not model_artifact_path or not os.path.exists(model_artifact_path):
            print("   ERROR: Model artifact not found on disk")
            return False
        
    except Exception as e:
        print(f"   Training error: {e}")
        return False
    
    # Step 5: Verify model paths in database
    print("5. Checking database model paths...")
    model_paths = get_current_model_paths()
    print(f"   Found {len(model_paths)} models in database:")
    for model_name, path in model_paths.items():
        print(f"     {model_name}: {path}")
        if not os.path.exists(path):
            print(f"     ERROR: File not found: {path}")
            return False
    
    # Step 6: Test model loading on new service instance
    print("6. Testing model loading with new service instance...")
    try:
        new_service = ModelTrainingService()
        loaded_models = list(new_service.models.keys())
        print(f"   Loaded models: {loaded_models}")
        
        # Check if Ridge Regression models were loaded
        ridge_models = [m for m in loaded_models if 'Ridge Regression' in m]
        if not ridge_models:
            print("   ERROR: Ridge Regression models not loaded")
            return False
        
        print(f"   Successfully loaded: {ridge_models}")
        
    except Exception as e:
        print(f"   Model loading error: {e}")
        return False
    
    # Step 7: Test prediction generation with loaded models
    print("7. Testing prediction generation with loaded models...")
    try:
        predictions = new_service.generate_predictions("Ridge Regression", df, 3)
        if not predictions or len(predictions) == 0:
            print("   ERROR: No predictions generated")
            return False
        
        print(f"   Generated {len(predictions)} predictions successfully")
        print(f"   Sample prediction: {predictions[0]}")
        
    except Exception as e:
        print(f"   Prediction generation error: {e}")
        return False
    
    print("\n✅ Model persistence validation completed successfully!")
    print("\nKey achievements:")
    print("- Models are saved to disk after training")
    print("- Model artifact paths are stored in database")
    print("- Models are automatically loaded on service initialization")
    print("- Loaded models can generate predictions")
    
    return True

def cleanup_test_models():
    """Clean up test model files."""
    print("\nCleaning up test models...")
    models_dir = Path("models")
    if models_dir.exists():
        for model_file in models_dir.glob("*.joblib"):
            if "ridge_regression" in model_file.name.lower():
                try:
                    model_file.unlink()
                    print(f"   Removed: {model_file}")
                except Exception as e:
                    print(f"   Failed to remove {model_file}: {e}")

if __name__ == "__main__":
    try:
        success = test_model_persistence()
        if not success:
            print("\n❌ Model persistence validation failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)