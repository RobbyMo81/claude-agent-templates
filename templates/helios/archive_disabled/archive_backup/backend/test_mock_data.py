#!/usr/bin/env python3
"""
Test script for mock data generator validation
"""

from trainer import DataPreprocessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_mock_data_generator():
    """Test the mock data generator functionality."""
    print('🧪 Testing Mock Data Generator...')
    print('=' * 50)
    
    try:
        # Create preprocessor instance
        preprocessor = DataPreprocessor()
        
        # Test mock data generation
        print('📊 Generating mock data...')
        mock_df = preprocessor.load_historical_data('mock')
        
        print(f'✅ Mock data generated successfully!')
        print(f'📊 Shape: {mock_df.shape}')
        print(f'📅 Date range: {mock_df["draw_date"].min()} to {mock_df["draw_date"].max()}')
        print()
        
        print('🎱 Sample data (first 5 records):')
        print(mock_df.head().to_string())
        print()
        
        print('📈 Data validation:')
        white_cols = ['white_ball_1', 'white_ball_2', 'white_ball_3', 'white_ball_4', 'white_ball_5']
        white_min = mock_df[white_cols].min().min()
        white_max = mock_df[white_cols].max().max()
        pb_min = mock_df['powerball'].min()
        pb_max = mock_df['powerball'].max()
        
        print(f'  • White ball range: {white_min} - {white_max} (should be 1-69)')
        print(f'  • Powerball range: {pb_min} - {pb_max} (should be 1-26)')
        
        # Validate ranges
        if white_min < 1 or white_max > 69:
            print(f'    ❌ White ball range violation!')
        else:
            print(f'    ✅ White ball range is correct')
            
        if pb_min < 1 or pb_max > 26:
            print(f'    ❌ Powerball range violation!')
        else:
            print(f'    ✅ Powerball range is correct')
        
        # Check for unique white balls per draw
        print(f'  • Checking white ball uniqueness per draw...')
        unique_check = True
        for idx, row in mock_df.head(3).iterrows():
            white_balls = [row[col] for col in white_cols]
            if len(set(white_balls)) != 5:
                unique_check = False
                print(f'    ❌ Row {idx}: Duplicate white balls found: {white_balls}')
            else:
                print(f'    ✅ Row {idx}: Unique white balls: {sorted(white_balls)}')
        
        print()
        print('📊 Column info:')
        print(mock_df.dtypes)
        print()
        
        # Test feature engineering
        print('🔧 Testing feature engineering...')
        enhanced_df = preprocessor.add_features(mock_df)
        print(f'✅ Features added successfully!')
        print(f'📊 Enhanced shape: {enhanced_df.shape}')
        print(f'📋 New columns: {enhanced_df.shape[1] - mock_df.shape[1]} features added')
        
        # Show some feature columns
        feature_cols = [col for col in enhanced_df.columns if col not in mock_df.columns]
        print(f'🎯 Sample features: {feature_cols[:10]}')
        
        print()
        print('🎉 Mock data generator test completed successfully!')
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_mock_data_generator()
    exit(0 if success else 1)
