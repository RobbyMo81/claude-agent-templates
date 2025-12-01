#!/usr/bin/env python3
"""
Test script to verify storage API fixes
"""
import sys
sys.path.append('.')

from core.persistent_model_predictions import PersistentModelPredictionManager
from core.modernized_prediction_system import ModernizedPredictionSystem
import logging

logging.basicConfig(level=logging.INFO)

def test_storage_api_fixes():
    """Test the fixed storage API interface"""
    print("Testing storage API fixes...")
    
    # Test 1: Direct storage manager
    manager = PersistentModelPredictionManager()
    test_prediction = {
        'white_numbers': [5, 15, 25, 35, 45],
        'powerball': 10,
        'probability': 0.00123
    }
    
    try:
        # Test single prediction storage with standardized API
        set_id = manager.store_predictions(
            model_name='Test_Model_Single',
            predictions=[test_prediction],
            features_used=['frequency', 'recency'],
            hyperparameters={'method': 'test'},
            performance_metrics={'accuracy': 0.85},
            training_duration=1.0,
            notes='Single prediction API test'
        )
        print(f"✓ Single prediction stored successfully: {set_id}")
        
        # Test batch prediction storage
        batch_predictions = [test_prediction] * 5
        set_id2 = manager.store_predictions(
            model_name='Test_Model_Batch',
            predictions=batch_predictions,
            features_used=['frequency', 'recency'],
            hyperparameters={'method': 'test_batch'},
            performance_metrics={'accuracy': 0.90},
            training_duration=2.0,
            notes='Batch prediction API test'
        )
        print(f"✓ Batch predictions stored successfully: {set_id2}")
        
        # Test 2: Modernized prediction system integration
        print("\nTesting modernized prediction system...")
        modern_system = ModernizedPredictionSystem()
        
        # Generate a prediction to test storage
        prediction_data = {
            'white_numbers': [7, 17, 27, 37, 47],
            'powerball': 12,
            'probability': 0.00234,
            'features_used': ['statistical_patterns', 'frequency_analysis'],
            'hyperparameters': {'method': 'modernized_statistical'},
            'performance_metrics': {'accuracy_estimate': 0.75}
        }
        
        set_id3 = modern_system.store_prediction(prediction_data)
        print(f"✓ Modernized system prediction stored: {set_id3}")
        
        print("\n=== ALL TESTS PASSED ===")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_storage_api_fixes()
    sys.exit(0 if success else 1)