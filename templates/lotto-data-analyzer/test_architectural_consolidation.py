#!/usr/bin/env python3
"""
Comprehensive test suite for Phase 2 architectural consolidation
"""
import sys
sys.path.append('.')

from core.feature_engineering_service import FeatureEngineeringService
from core.model_training_service import ModelTrainingService
from core.modernized_prediction_system import ModernizedPredictionSystem
from core.storage import get_store
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def test_feature_engineering_service():
    """Test centralized feature engineering service."""
    print("Testing FeatureEngineeringService...")
    
    # Get test data
    df = get_store().latest()
    if df is None or len(df) < 50:
        print("✗ Insufficient test data")
        return False
    
    service = FeatureEngineeringService()
    
    # Test comprehensive feature engineering
    features = service.engineer_features(df)
    if features is None or features.shape[0] != len(df):
        print("✗ Feature engineering failed")
        return False
    
    print(f"✓ Feature engineering successful: {features.shape}")
    
    # Test specific prediction features
    for pred_type in ['frequency', 'recency', 'trends', 'combinations', 'statistical', 'dow']:
        pred_features = service.get_prediction_features(df, pred_type)
        if not pred_features:
            print(f"✗ {pred_type} prediction features failed")
            return False
        print(f"✓ {pred_type} prediction features generated")
    
    return True

def test_model_training_service():
    """Test unified model training service."""
    print("\nTesting ModelTrainingService...")
    
    df = get_store().latest()
    if df is None or len(df) < 50:
        print("✗ Insufficient training data")
        return False
    
    service = ModelTrainingService()
    
    # Test single model training
    result = service.train_and_predict(df, 'Ridge Regression', prediction_count=3)
    
    if not result.get('success'):
        print(f"✗ Model training failed: {result.get('error', 'Unknown error')}")
        return False
    
    predictions = result.get('predictions', [])
    if len(predictions) != 3:
        print("✗ Incorrect prediction count")
        return False
    
    # Validate prediction format
    for pred in predictions:
        if not all(key in pred for key in ['white_numbers', 'powerball', 'probability']):
            print("✗ Invalid prediction format")
            return False
        
        white_nums = pred['white_numbers']
        if len(white_nums) != 5 or not all(1 <= n <= 69 for n in white_nums):
            print("✗ Invalid white ball numbers")
            return False
        
        if not 1 <= pred['powerball'] <= 26:
            print("✗ Invalid powerball number")
            return False
    
    print(f"✓ Model training and prediction successful")
    print(f"  - Set ID: {result.get('set_id', 'N/A')}")
    print(f"  - Training duration: {result.get('training_duration', 0):.2f}s")
    
    return True

def test_modernized_system_integration():
    """Test integration of modernized prediction system with centralized services."""
    print("\nTesting ModernizedPredictionSystem integration...")
    
    df = get_store().latest()
    if df is None or len(df) < 50:
        print("✗ Insufficient data")
        return False
    
    system = ModernizedPredictionSystem(df)
    
    # Test feature engineering integration
    try:
        features = system._engineer_features(df)
        if features is None or features.shape[0] != len(df):
            print("✗ Integrated feature engineering failed")
            return False
        print("✓ Feature engineering integration successful")
    except Exception as e:
        print(f"✗ Feature engineering integration error: {e}")
        return False
    
    # Test tool predictions with centralized features
    try:
        freq_pred = system._frequency_prediction()
        recency_pred = system._recency_prediction()
        
        for pred_name, pred in [('frequency', freq_pred), ('recency', recency_pred)]:
            if not pred or 'white_numbers' not in pred:
                print(f"✗ {pred_name} prediction failed")
                return False
            
            white_nums = pred['white_numbers']
            if len(white_nums) != 5 or not all(1 <= n <= 69 for n in white_nums):
                print(f"✗ Invalid {pred_name} prediction")
                return False
        
        print("✓ Tool predictions with centralized features successful")
    except Exception as e:
        print(f"✗ Tool prediction integration error: {e}")
        return False
    
    return True

def test_prediction_consistency():
    """Test consistency between old and new implementations."""
    print("\nTesting prediction consistency...")
    
    df = get_store().latest()
    if df is None or len(df) < 50:
        print("✗ Insufficient data")
        return False
    
    # Test modernized system predictions
    system = ModernizedPredictionSystem(df)
    predictions = system.generate_weighted_predictions(count=2)
    
    if len(predictions) != 2:
        print("✗ Incorrect prediction count from modernized system")
        return False
    
    # Validate prediction format and content
    for i, pred in enumerate(predictions):
        if not all(key in pred for key in ['white_numbers', 'powerball', 'probability']):
            print(f"✗ Invalid prediction format (prediction {i+1})")
            return False
        
        white_nums = pred['white_numbers']
        if len(white_nums) != 5 or not all(1 <= n <= 69 for n in white_nums):
            print(f"✗ Invalid white numbers (prediction {i+1})")
            return False
        
        if not 1 <= pred['powerball'] <= 26:
            print(f"✗ Invalid powerball (prediction {i+1})")
            return False
    
    print("✓ Prediction consistency validated")
    return True

def test_storage_integration():
    """Test storage integration with unified services."""
    print("\nTesting storage integration...")
    
    df = get_store().latest()
    service = ModelTrainingService()
    
    # Test training and storage
    result = service.train_and_predict(df, 'Random Forest', prediction_count=1)
    
    if not result.get('success'):
        print("✗ Training and storage failed")
        return False
    
    set_id = result.get('set_id')
    if not set_id:
        print("✗ No set_id returned from storage")
        return False
    
    # Verify storage worked
    storage_manager = service.storage_manager
    stored_predictions = storage_manager.get_current_predictions('Random Forest')
    
    if not stored_predictions:
        print("✗ Predictions not found in storage")
        return False
    
    print(f"✓ Storage integration successful (set_id: {set_id})")
    return True

def run_comprehensive_test():
    """Run all consolidation tests."""
    print("=== Phase 2 Architectural Consolidation Test Suite ===\n")
    
    tests = [
        ("Feature Engineering Service", test_feature_engineering_service),
        ("Model Training Service", test_model_training_service),
        ("Modernized System Integration", test_modernized_system_integration),
        ("Prediction Consistency", test_prediction_consistency),
        ("Storage Integration", test_storage_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name}: PASSED\n")
            else:
                print(f"✗ {test_name}: FAILED\n")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}\n")
    
    success_rate = (passed / total) * 100
    print(f"=== Test Results: {passed}/{total} tests passed ({success_rate:.1f}%) ===")
    
    if passed == total:
        print("🎉 All consolidation tests PASSED - Architecture successfully unified!")
        return True
    else:
        print("⚠️ Some tests failed - consolidation needs attention")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)