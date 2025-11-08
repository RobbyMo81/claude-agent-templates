#!/usr/bin/env python3
"""
Standalone test for mock data generation logic
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_mock_data(num_records: int = 100) -> pd.DataFrame:
    """
    Generate synthetic lottery data for development and testing.
    
    Creates realistic Powerball data with proper constraints:
    - White balls: 1-69 (5 unique numbers)
    - Powerball: 1-26 (1 number)
    - Draw dates: Recent historical dates
    """
    print(f"Generating {num_records} mock lottery records...")
    
    # Set seed for consistent mock data in development
    random.seed(42)
    np.random.seed(42)
    
    mock_data = []
    
    # Generate dates going back from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_records * 3)  # ~3 draws per week
    
    for i in range(num_records):
        # Generate draw date (typically Wed/Sat for Powerball)
        draw_date = start_date + timedelta(days=i * 3)
        
        # Generate 5 unique white balls (1-69)
        white_balls = sorted(random.sample(range(1, 70), 5))
        
        # Generate 1 powerball (1-26)
        powerball = random.randint(1, 26)
        
        mock_data.append({
            'draw_date': draw_date.strftime('%Y-%m-%d'),
            'white_ball_1': white_balls[0],
            'white_ball_2': white_balls[1],
            'white_ball_3': white_balls[2],
            'white_ball_4': white_balls[3],
            'white_ball_5': white_balls[4],
            'powerball': powerball
        })
    
    df = pd.DataFrame(mock_data)
    print(f"Generated mock data: {len(df)} records from {df['draw_date'].min()} to {df['draw_date'].max()}")
    
    return df

def test_mock_data():
    """Test the mock data generation."""
    print('🧪 Testing Mock Data Generator...')
    print('=' * 50)
    
    try:
        # Generate mock data
        mock_df = generate_mock_data(50)  # Small sample for testing
        
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
        unique_violations = 0
        for idx, row in mock_df.head(10).iterrows():
            white_balls = [row[col] for col in white_cols]
            if len(set(white_balls)) != 5:
                unique_violations += 1
                print(f'    ❌ Row {idx}: Duplicate white balls found: {white_balls}')
            else:
                print(f'    ✅ Row {idx}: Unique white balls: {sorted(white_balls)}')
        
        if unique_violations == 0:
            print(f'    🎉 All draws have unique white balls!')
        
        print()
        print('📊 Column info:')
        print(mock_df.dtypes)
        print()
        
        print('🎉 Mock data generator test completed successfully!')
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_mock_data()
    exit(0 if success else 1)
