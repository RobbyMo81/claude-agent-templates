"""
Centralized Feature Engineering Service
=====================================
Unified service for all feature engineering operations across ML systems.
Eliminates code duplication and provides consistent feature computation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import logging
from datetime import datetime, timedelta
from .frequency import calc_frequency

logger = logging.getLogger(__name__)

class FeatureEngineeringService:
    """
    Centralized service for feature engineering operations.
    Provides unified feature computation for all prediction systems. 
    """
    
    def __init__(self):
        """Initialize the feature engineering service."""
        self.feature_cache = {}
        
    def engineer_features(
        self, df: pd.DataFrame, feature_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Engineer comprehensive features for machine learning models.
        
        Args:
            df: DataFrame with historical draw data
            feature_types: List of feature types to include (default: all)
            
        Returns:
            Tuple containing:
            - Array of engineered features
            - List of feature names
        """
        if feature_types is None:
            feature_types = ['temporal', 'frequency', 'statistical', 'recency', 'trends', 'lag']
            
        try:
            df = df.copy()
            features = []
            
            # Ensure draw_date is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['draw_date']):
                df['draw_date'] = pd.to_datetime(df['draw_date'])
            
            # Temporal features
            if 'temporal' in feature_types:
                temporal_features = self._engineer_temporal_features(df)
                features.append(temporal_features)
            
            # Frequency-based features
            if 'frequency' in feature_types:
                frequency_features = self._engineer_frequency_features(df)
                features.append(frequency_features)
            
            # Statistical features
            if 'statistical' in feature_types:
                statistical_features = self._engineer_statistical_features(df)
                features.append(statistical_features)
            
            # Recency features
            if 'recency' in feature_types:
                recency_features = self._engineer_recency_features(df)
                features.append(recency_features)
            
            # Trend features
            if 'trends' in feature_types:
                trend_features = self._engineer_trend_features(df)
                features.append(trend_features)
            
            # Lag features
            if 'lag' in feature_types:
                lag_features = self._engineer_lag_features(df)
                features.append(lag_features)
            
            # Combine all features
            if features:
                combined = pd.concat(features, axis=1).fillna(0)
                # Return a DataFrame for richer downstream use and for tests that expect .shape
                return combined
            else:
                # Return an empty DataFrame with expected number of rows
                return pd.DataFrame(index=df.index)
                
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            # Return an empty DataFrame on error to keep API stable for callers/tests
            return pd.DataFrame(index=df.index)
    
    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer temporal features from draw dates."""
        temporal_df = pd.DataFrame(index=df.index)
        
        # Basic temporal features
        temporal_df['day_of_week'] = df['draw_date'].dt.dayofweek
        temporal_df['month'] = df['draw_date'].dt.month
        temporal_df['year'] = df['draw_date'].dt.year
        temporal_df['day_of_year'] = df['draw_date'].dt.dayofyear
        temporal_df['quarter'] = df['draw_date'].dt.quarter
        
        # One-hot encode day of week
        dow_dummies = pd.get_dummies(temporal_df['day_of_week'], prefix='dow')
        temporal_df = pd.concat([temporal_df, dow_dummies], axis=1)
        
        # Month dummies
        month_dummies = pd.get_dummies(temporal_df['month'], prefix='month')
        temporal_df = pd.concat([temporal_df, month_dummies], axis=1)
        
        # Draw sequence number
        temporal_df['draw_sequence'] = np.arange(len(df))
        
        return temporal_df
    
    def _engineer_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer frequency-based features for each number."""
        freq_df = pd.DataFrame(index=df.index)
        
        try:
            # Get overall frequency analysis
            frequency_analysis = calc_frequency(df)
            frequency_dict = dict(zip(frequency_analysis['number'], frequency_analysis['frequency']))
            
            # Create frequency features for each draw
            white_cols = ['n1', 'n2', 'n3', 'n4', 'n5']
            
            for i, col in enumerate(white_cols):
                freq_df[f'freq_{col}'] = df[col].map(frequency_dict).fillna(0)
            
            # Powerball frequency
            pb_freq = df['powerball'].value_counts().to_dict()
            freq_df['freq_powerball'] = df['powerball'].map(pb_freq).fillna(0)
            
            # Aggregate frequency features
            freq_df['avg_white_frequency'] = freq_df[[f'freq_{col}' for col in white_cols]].mean(axis=1)
            freq_df['min_white_frequency'] = freq_df[[f'freq_{col}' for col in white_cols]].min(axis=1)
            freq_df['max_white_frequency'] = freq_df[[f'freq_{col}' for col in white_cols]].max(axis=1)
            
        except Exception as e:
            logger.warning(f"Error in frequency features: {e}")
            # Fallback with basic frequency calculation
            for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'powerball']:
                freq_df[f'freq_{col}'] = 0
            freq_df['avg_white_frequency'] = 0
            freq_df['min_white_frequency'] = 0
            freq_df['max_white_frequency'] = 0
        
        return freq_df
    
    def _engineer_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer statistical features from draw data."""
        stat_df = pd.DataFrame(index=df.index)
        
        white_cols = ['n1', 'n2', 'n3', 'n4', 'n5']
        
        # Basic statistics
        stat_df['white_sum'] = df[white_cols].sum(axis=1)
        stat_df['white_mean'] = df[white_cols].mean(axis=1)
        stat_df['white_std'] = df[white_cols].std(axis=1)
        stat_df['white_min'] = df[white_cols].min(axis=1)
        stat_df['white_max'] = df[white_cols].max(axis=1)
        stat_df['white_range'] = stat_df['white_max'] - stat_df['white_min']
        
        # Rolling statistics
        for window in [5, 10, 20]:
            if len(df) >= window:
                stat_df[f'sum_mean_{window}'] = stat_df['white_sum'].rolling(window=window, min_periods=1).mean()
                stat_df[f'sum_std_{window}'] = stat_df['white_sum'].rolling(window=window, min_periods=1).std()
                stat_df[f'mean_mean_{window}'] = stat_df['white_mean'].rolling(window=window, min_periods=1).mean()
        
        # Number distribution features
        stat_df['low_numbers'] = df[white_cols].apply(lambda x: sum(1 for n in x if n <= 35), axis=1)
        stat_df['high_numbers'] = df[white_cols].apply(lambda x: sum(1 for n in x if n > 35), axis=1)
        stat_df['even_numbers'] = df[white_cols].apply(lambda x: sum(1 for n in x if n % 2 == 0), axis=1)
        stat_df['odd_numbers'] = df[white_cols].apply(lambda x: sum(1 for n in x if n % 2 == 1), axis=1)
        
        return stat_df
    
    def _engineer_recency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer recency-based features."""
        recency_df = pd.DataFrame(index=df.index)
        
        white_cols = ['n1', 'n2', 'n3', 'n4', 'n5']
        
        # Track last appearance of each number
        for col in white_cols + ['powerball']:
            recency_df[f'last_seen_{col}'] = 0
            
        # Calculate recency for each draw
        # Use integer row positions (pos) to avoid arithmetic on non-numeric index types
        last_seen = {}
        
        for pos, (idx, row) in enumerate(df.iterrows()):
            # Update recency features using integer positions
            for col in white_cols + ['powerball']:
                number = row[col]
                if number in last_seen:
                    recency_df.at[idx, f'last_seen_{col}'] = pos - last_seen[number]
                else:
                    recency_df.at[idx, f'last_seen_{col}'] = pos + 1  # New number (position-based)
                last_seen[number] = pos
        
        # Recent appearance features (last 10 draws)
        recent_window = 10
        for i in range(len(df)):
            start_idx = max(0, i - recent_window)
            recent_data = df.iloc[start_idx:i]
            
            if len(recent_data) > 0:
                recent_numbers = []
                for _, r in recent_data.iterrows():
                    recent_numbers.extend([r['n1'], r['n2'], r['n3'], r['n4'], r['n5']])
                
                recent_counter = Counter(recent_numbers)
                current_numbers = [df.iloc[i]['n1'], df.iloc[i]['n2'], df.iloc[i]['n3'], 
                                 df.iloc[i]['n4'], df.iloc[i]['n5']]
                
                recency_df.at[df.index[i], 'recent_appearances'] = sum(recent_counter.get(n, 0) for n in current_numbers)
        
        return recency_df
    
    def _engineer_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer trend-based features."""
        trend_df = pd.DataFrame(index=df.index)
        
        white_cols = ['n1', 'n2', 'n3', 'n4', 'n5']
        
        # Calculate trends over different windows
        for window in [3, 5, 10]:
            if len(df) >= window:
                for col in white_cols:
                    # Moving average
                    trend_df[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    
                    # Trend direction (difference from moving average)
                    trend_df[f'{col}_trend_{window}'] = df[col] - trend_df[f'{col}_ma_{window}']
        
        # Overall trend features
        trend_df['sum_trend_3'] = df[white_cols].sum(axis=1).rolling(window=3, min_periods=1).apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
        )
        
        trend_df['mean_trend_3'] = df[white_cols].mean(axis=1).rolling(window=3, min_periods=1).apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
        )
        
        return trend_df
    
    def _engineer_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer lag features from previous draws."""
        lag_df = pd.DataFrame(index=df.index)
        
        white_cols = ['n1', 'n2', 'n3', 'n4', 'n5']
        
        # Create lag features for different lags
        for lag in [1, 2, 3]:
            if len(df) > lag:
                for col in white_cols + ['powerball']:
                    lag_df[f'{col}_lag{lag}'] = df[col].shift(lag).fillna(0)
        
        # Lag differences
        for lag in [1, 2]:
            if len(df) > lag:
                for col in white_cols:
                    lag_df[f'{col}_diff{lag}'] = df[col] - df[col].shift(lag)
        
        return lag_df.fillna(0)
    
    def get_prediction_features(self, df: pd.DataFrame, prediction_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Get features for specific prediction types used by prediction tools.
        
        Args:
            df: Historical data
            prediction_type: Type of prediction features needed
            
        Returns:
            Dictionary with computed features for prediction generation
        """
        try:
            if prediction_type == 'frequency':
                return self._get_frequency_prediction_features(df)
            elif prediction_type == 'recency':
                return self._get_recency_prediction_features(df)
            elif prediction_type == 'trends':
                return self._get_trends_prediction_features(df)
            elif prediction_type == 'combinations':
                return self._get_combination_prediction_features(df)
            elif prediction_type == 'statistical':
                return self._get_statistical_prediction_features(df)
            elif prediction_type == 'dow':
                return self._get_dow_prediction_features(df)
            else:
                # Comprehensive features
                return {
                    'frequency': self._get_frequency_prediction_features(df),
                    'recency': self._get_recency_prediction_features(df),
                    'trends': self._get_trends_prediction_features(df),
                    'combinations': self._get_combination_prediction_features(df),
                    'statistical': self._get_statistical_prediction_features(df),
                    'dow': self._get_dow_prediction_features(df)
                }
        except Exception as e:
            logger.error(f"Error getting prediction features: {e}")
            return {}
    
    def _get_frequency_prediction_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get features for frequency-based predictions."""
        try:
            freq_analysis = calc_frequency(df)
            return {
                'top_numbers': freq_analysis.head(30)['number'].tolist(),
                'frequency_dict': dict(zip(freq_analysis['number'], freq_analysis['frequency'])),
                'white_candidates': [n for n in freq_analysis['number'].tolist() if n <= 69],
                'powerball_candidates': [n for n in freq_analysis['number'].tolist() if n <= 26]
            }
        except Exception as e:
            logger.error(f"Error in frequency prediction features: {e}")
            return {'top_numbers': list(range(1, 31)), 'frequency_dict': {}}
    
    def _get_recency_prediction_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get features for recency-based predictions."""
        recent_data = df.tail(10)
        recent_numbers = []
        
        for _, row in recent_data.iterrows():
            recent_numbers.extend([row['n1'], row['n2'], row['n3'], row['n4'], row['n5']])
        
        return {
            'recent_numbers': recent_numbers,
            'recent_set': set(recent_numbers),
            'avoided_numbers': list(set(recent_numbers)),
            'candidate_numbers': list(set(range(1, 70)) - set(recent_numbers))
        }
    
    def _get_trends_prediction_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get features for trend-based predictions."""
        recent_data = df.tail(5)
        trend_numbers = []
        
        for _, row in recent_data.iterrows():
            trend_numbers.extend([row['n1'], row['n2'], row['n3'], row['n4'], row['n5']])
        
        # Calculate average positions
        avg_numbers = []
        for i in range(5):
            position_numbers = [trend_numbers[j] for j in range(i, len(trend_numbers), 5)]
            if position_numbers:
                avg_numbers.append(sum(position_numbers) / len(position_numbers))
        
        return {
            'trend_numbers': trend_numbers,
            'average_positions': avg_numbers,
            'trend_direction': 'up' if len(trend_numbers) > 0 and trend_numbers[-5:] > trend_numbers[:5] else 'down'
        }
    
    def _get_combination_prediction_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get features for combination-based predictions."""
        combinations = {}
        
        for _, row in df.iterrows():
            numbers = [row['n1'], row['n2'], row['n3'], row['n4'], row['n5']]
            
            # Track pairs
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    pair = tuple(sorted([numbers[i], numbers[j]]))
                    combinations[pair] = combinations.get(pair, 0) + 1
        
        top_pairs = sorted(combinations.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            'all_combinations': combinations,
            'top_pairs': top_pairs,
            'frequent_combinations': [pair[0] for pair in top_pairs[:10]]
        }
    
    def _get_statistical_prediction_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get features for statistical-based predictions."""
        white_cols = ['n1', 'n2', 'n3', 'n4', 'n5']
        
        # Calculate sum statistics
        sums = df[white_cols].sum(axis=1)
        
        return {
            'mean_sum': sums.mean(),
            'std_sum': sums.std(),
            'min_sum': sums.min(),
            'max_sum': sums.max(),
            'target_sum': int(sums.mean()),
            'sum_range': (int(sums.mean() - sums.std()), int(sums.mean() + sums.std()))
        }
    
    def _get_dow_prediction_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get features for day-of-week based predictions."""
        from datetime import datetime, timedelta
        
        # Calculate next draw date (assuming Wednesday and Saturday)
        today = datetime.now()
        days_until_wed = (2 - today.weekday()) % 7
        days_until_sat = (5 - today.weekday()) % 7
        
        if days_until_wed <= days_until_sat and days_until_wed > 0:
            next_draw_date = today + timedelta(days=days_until_wed)
        else:
            next_draw_date = today + timedelta(days=days_until_sat if days_until_sat > 0 else 7)
        
        dow = next_draw_date.weekday()
        
        # Filter data by day of week
        df_copy = df.copy()
        df_copy['draw_date'] = pd.to_datetime(df_copy['draw_date'])
        dow_data = df_copy[df_copy['draw_date'].dt.weekday == dow]
        
        dow_numbers = []
        for _, row in dow_data.iterrows():
            dow_numbers.extend([row['n1'], row['n2'], row['n3'], row['n4'], row['n5']])
        
        dow_counter = Counter(dow_numbers)
        
        return {
            'next_draw_date': next_draw_date,
            'day_of_week': dow,
            'dow_data': dow_data,
            'dow_numbers': dow_numbers,
            'dow_frequency': dow_counter,
            'top_dow_numbers': [num for num, count in dow_counter.most_common(30)]
        }