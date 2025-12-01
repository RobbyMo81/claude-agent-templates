#!/usr/bin/env python3
"""
Test Historical Predictions Function - Phase 4
==============================================
Tests the new get_historical_predictions_by_model function to verify it retrieves
all predictions regardless of is_active status.
"""

import sys
sys.path.insert(0, '.')

from core.persistent_model_predictions import get_prediction_manager

def test_historical_predictions():
    """Test the new historical predictions function."""
    
    print("=" * 60)
    print("TESTING HISTORICAL PREDICTIONS FUNCTION")
    print("=" * 60)
    
    pm = get_prediction_manager()
    
    # Test each model
    models = ['Ridge Regression', 'Random Forest', 'Gradient Boosting']
    
    for model in models:
        print(f"\n--- Testing {model} ---")
        
        # Get current predictions (only active)
        current_preds = pm.get_current_predictions(model)
        current_count = len(current_preds) if current_preds else 0
        
        # Get historical predictions (all predictions)
        historical_preds = pm.get_historical_predictions_by_model(model)
        historical_count = len(historical_preds) if historical_preds else 0
        
        print(f"Current predictions (active only): {current_count}")
        print(f"Historical predictions (all): {historical_count}")
        
        if historical_preds:
            # Count active vs inactive
            active_count = sum(1 for pred in historical_preds if pred.get('is_active', False))
            inactive_count = historical_count - active_count
            
            print(f"  - Active: {active_count}")
            print(f"  - Inactive: {inactive_count}")
            
            # Show sample predictions
            print(f"\nSample historical predictions for {model}:")
            for i, pred in enumerate(historical_preds[:3]):  # Show first 3
                status = "ACTIVE" if pred.get('is_active') else "INACTIVE"
                print(f"  {i+1}. {pred['white_numbers']} | PB: {pred['powerball']} | {status} | {pred['created_at'][:10]}")
        else:
            print(f"No predictions found for {model}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_historical_predictions()