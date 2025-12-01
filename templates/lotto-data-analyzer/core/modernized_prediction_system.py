"""
Modernized Prediction System for Powerball Lottery
-------------------------------------------------
SQLite-only prediction system that replaces the legacy joblib storage.
This module integrates multiple analysis tools with modern persistence.

The system leverages:
- Frequency Analysis
- Day of Week Analysis  
- Time Trends Analysis
- Inter-Draw Gap Analysis
- Combinatorial Analysis
- Sum Analysis

All predictions and models are stored exclusively in SQLite database.
"""

import pandas as pd
import numpy as np
import datetime
import logging
import json
import os
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
import random

from .persistent_model_predictions import get_prediction_manager
from .model_training_service import ModelTrainingService

def get_next_powerball_draw_date(from_date: Optional[datetime.date] = None) -> datetime.date:
    """
    Calculate the next Powerball draw date.
    Powerball draws occur on Monday, Wednesday, and Saturday.
    """
    if from_date is None:
        from_date = datetime.datetime.now().date()
    
    draw_days = [0, 2, 5]  # Monday=0, Wednesday=2, Saturday=5
    
    for days_ahead in range(8):
        check_date = from_date + datetime.timedelta(days=days_ahead)
        if check_date.weekday() in draw_days:
            return check_date
    
    return from_date + datetime.timedelta(days=1)

def validate_powerball_dates(df: pd.DataFrame, date_column: str = 'draw_date') -> Dict:
    """Validate that dates in the dataset follow the Powerball schedule."""
    if date_column not in df.columns:
        return {'valid': False, 'error': f'Column {date_column} not found'}
    
    try:
        dates = pd.to_datetime(df[date_column])
        draw_days = [0, 2, 5]  # Monday, Wednesday, Saturday
        
        valid_dates = dates.dt.weekday.isin(draw_days)
        invalid_count = (~valid_dates).sum()
        
        return {
            'valid': invalid_count == 0,
            'total_draws': len(df),
            'invalid_dates': invalid_count,
            'valid_percentage': (valid_dates.sum() / len(df)) * 100,
            'invalid_weekdays': dates[~valid_dates].dt.day_name().tolist()
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}

# Import analysis tools
from .frequency import calc_frequency
from .inter_draw import _days_since_last, _all_gaps
from .time_trends import _aggregate, _calc_trending
from .combos import _count_combos
from .dow_analysis import _compute_stats
from .sums import _prep as _prep_sums

# Constants
WHITE_MIN, WHITE_MAX = 1, 69
PB_MIN, PB_MAX = 1, 26
WHITE_BALLS_COUNT = 5

class ModernizedPredictionSystem:
    """SQLite-based prediction system that leverages multiple analysis tools."""
    
    def __init__(self, df_history: Optional[pd.DataFrame] = None):
        """Initialize the modernized prediction system with SQLite storage."""
        self.df_history = pd.DataFrame() if df_history is None else df_history.copy()
        self.prediction_manager = get_prediction_manager()
        
        self.feature_weights = {
            'frequency': 0.20,
            'recency': 0.15,
            'trends': 0.15,
            'combo': 0.15,
            'sum': 0.15,
            'dow': 0.10,
            'ml': 0.10
        }
        
        # In-memory model storage (training results)
        self.models = {
            'white_balls': None,
            'powerball': None
        }
        
        self.model_cv_performance = {
            'white_balls': None,
            'powerball': None
        }
        
        self.feature_importance = {
            'white_balls': None,
            'powerball': None
        }
        
        # Cache for expensive computations
        self._prediction_cache = {}
        
        if df_history is not None:
            self.set_dataframe(df_history)
        
    def _calculate_next_draw_date(self) -> str:
        """Calculate the next Powerball draw date."""
        try:
            next_draw = get_next_powerball_draw_date()
            return next_draw.isoformat()
        except Exception as e:
            logging.warning(f"Error calculating next draw date: {e}")
            return datetime.datetime.now().date().isoformat()
    
    def clear_cache(self):
        """Clear the prediction cache."""
        self._prediction_cache = {}
        
    def set_dataframe(self, df_history: pd.DataFrame):
        """Update the dataframe and clear the cache."""
        self.df_history = df_history
        self.clear_cache()
        
    def get_prediction_history(self) -> List[Dict]:
        """Get prediction history from SQLite database."""
        try:
            predictions = self.prediction_manager.get_predictions_by_model("Modernized_Prediction_System")
            return predictions or []
        except Exception as e:
            logging.warning(f"Could not load prediction history: {e}")
            return []
    
    def store_prediction(self, prediction_data: Dict) -> str:
        """Store a prediction in SQLite database."""
        try:
            # Store the prediction with required parameters
            return self.prediction_manager.store_predictions(
                model_name="Modernized_Prediction_System",
                predictions=[prediction_data],
                features_used=prediction_data.get('features_used', ['statistical_patterns', 'frequency_analysis']),
                hyperparameters=prediction_data.get('hyperparameters', {'method': 'modernized_statistical'}),
                performance_metrics=prediction_data.get('performance_metrics', {'accuracy_estimate': 0.0}),
                training_duration=0.0,
                notes="Generated by modernized prediction system"
            )
        except Exception as e:
            logging.error(f"Failed to store prediction: {e}")
            return ""
    
    def update_with_new_draw(self, new_draw: Dict):
        """Update the system with new draw results."""
        try:
            # Get recent predictions for accuracy calculation
            recent_predictions = self.get_prediction_history()
            
            if not recent_predictions:
                return
            
            # Find most recent prediction
            most_recent = max(recent_predictions, 
                             key=lambda p: p.get('created_at', ''))
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(most_recent, new_draw)
            
            # Store accuracy as metadata
            accuracy_data = {
                'prediction_id': most_recent.get('prediction_id'),
                'accuracy_metrics': accuracy,
                'actual_draw': new_draw,
                'evaluation_timestamp': datetime.datetime.now().isoformat()
            }
            
            # Store accuracy in database as a special metadata entry
            self._store_accuracy_data(accuracy_data)
            
            # Clear cache since data has changed
            self.clear_cache()
            
            # Retrain models with new data
            if self.df_history is not None and len(self.df_history) > 0:
                self._train_models()
                
        except Exception as e:
            logging.error(f"Error updating with new draw: {e}")
    
    def _store_accuracy_data(self, accuracy_data: Dict):
        """Store accuracy evaluation data."""
        try:
            set_id = f"accuracy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Use the public API to store an accuracy tracking entry as a prediction set.
            # We store the accuracy payload as a single prediction record and include the set_id in notes.
            self.prediction_manager.store_predictions(
                model_name="Accuracy_Tracking",
                predictions=[accuracy_data],
                features_used=["accuracy_evaluation"],
                hyperparameters={},
                performance_metrics={},
                training_duration=0.0,
                notes=json.dumps({"set_id": set_id, "stored_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            )
                 
        except Exception as e:
            logging.error(f"Failed to store accuracy data: {e}")
    
    def _calculate_accuracy(self, prediction: Dict, actual: Dict) -> Dict:
        """Calculate accuracy metrics between prediction and actual draw."""
        try:
            # Parse prediction data
            white_numbers = prediction.get('white_numbers', [])
            if isinstance(white_numbers, str):
                white_numbers = json.loads(white_numbers)
            
            pred_whites = set(white_numbers)
            actual_whites = set(actual.get('white_numbers', []))
            pred_pb = prediction.get('powerball', 0)
            actual_pb = actual.get('powerball', 0)
            
            # Calculate metrics
            white_matches = len(pred_whites.intersection(actual_whites))
            pb_match = pred_pb == actual_pb
            
            return {
                'white_matches': white_matches,
                'pb_match': pb_match,
                'white_accuracy': white_matches / WHITE_BALLS_COUNT,
                'full_match': white_matches == WHITE_BALLS_COUNT and pb_match
            }
        except Exception as e:
            logging.error(f"Error calculating accuracy: {e}")
            return {'white_matches': 0, 'pb_match': False, 'white_accuracy': 0.0, 'full_match': False}
    
    def _train_models(self):
        """Train machine learning models with cross-validation."""
        if self.df_history is None or len(self.df_history) < 50:
            return
        
        try:
            # Initialize ModelTrainingService
            training_service = ModelTrainingService()
            
            # Define models to train (using GradientBoosting as it was in original _train_models)
            model_names = ["Gradient Boosting"]
            
            # Train models using the centralized service
            training_results = training_service.train_models(self.df_history, model_names=model_names)
            
            if training_results and training_results.get('results'):
                gb_results = training_results['results'].get('Gradient Boosting')
                if gb_results and gb_results.get('training_completed'):
                    # Store the trained pipelines from the service into this system's models
                    self.models['white_balls'] = gb_results['white_pipeline']
                    self.models['powerball'] = gb_results['powerball_pipeline']
                    
                    # Store performance metrics
                    self.model_cv_performance['white_balls'] = gb_results['white_mae']
                    self.model_cv_performance['powerball'] = gb_results['powerball_mae']
                    
                    logging.info(f"Models trained via ModelTrainingService: White MAE: {gb_results['white_mae']:.2f}, Powerball MAE: {gb_results['powerball_mae']:.2f}")
                else:
                    logging.error(f"Gradient Boosting model training failed via ModelTrainingService: {gb_results.get('error', 'Unknown error')}")
            else:
                logging.error("No training results returned from ModelTrainingService.")
            
        except Exception as e:
            logging.error(f"Error training models: {e}")
    
    def _engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer features using centralized feature engineering service."""
        try:
            from .feature_engineering_service import FeatureEngineeringService
            feature_service = FeatureEngineeringService()
            features_df = feature_service.engineer_features(df)
            # Preserve historical API: always return an ndarray when callers expect arrays
            return np.asarray(features_df)
        except Exception as e:
            logging.error(f"Error engineering features: {e}")
            return np.array([[0] * 10] * len(df))  # Fallback with basic feature count
    
    def generate_weighted_predictions(self, count: int = 5) -> List[Dict]:
        """Generate weighted predictions using multiple analysis tools."""
        if self.df_history is None or len(self.df_history) == 0:
            return self._generate_fallback_predictions(count)
        
        try:
            predictions = []
            
            for i in range(count):
                # Get predictions from each tool
                tool_predictions = self._get_tool_predictions()
                
                # Combine using weighted approach
                combined_prediction = self._combine_tool_predictions(tool_predictions)
                
                # Add metadata
                combined_prediction.update({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'prediction_for_date': self._calculate_next_draw_date(),
                    'tool_contributions': tool_predictions,
                    'storage_version': '3.0_sqlite'
                })
                
                predictions.append(combined_prediction)
                
                # Store in database
                self.store_prediction(combined_prediction)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error generating predictions: {e}")
            return self._generate_fallback_predictions(count)
    
    def _get_tool_predictions(self) -> Dict:
        """Get predictions from each analysis tool."""
        tools = {}
        
        try:
            # Frequency analysis
            tools['frequency'] = self._frequency_prediction()
            
            # Recency analysis
            tools['recency'] = self._recency_prediction()
            
            # Trends analysis
            tools['trends'] = self._trends_prediction()
            
            # Combination analysis
            tools['combo'] = self._combo_prediction()
            
            # Sum analysis
            tools['sum'] = self._sum_prediction()
            
            # Day of week analysis
            tools['dow'] = self._dow_prediction()
            
            # ML prediction
            tools['ml'] = self._ml_prediction()
            
        except Exception as e:
            logging.error(f"Error getting tool predictions: {e}")
        
        return tools
    
    def _frequency_prediction(self) -> Dict:
        """Generate prediction based on frequency analysis using centralized service."""
        try:
            from .feature_engineering_service import FeatureEngineeringService
            feature_service = FeatureEngineeringService()
            
            freq_features = feature_service.get_prediction_features(self.df_history, 'frequency')
            
            # Select from top frequent numbers
            white_candidates = freq_features.get('white_candidates', list(range(1, 70)))
            pb_candidates = freq_features.get('powerball_candidates', list(range(1, 27)))
            
            white_selection = random.sample(white_candidates[:30], min(5, len(white_candidates)))
            powerball = random.choice(pb_candidates[:10]) if pb_candidates else random.randint(1, 26)
            
            return {
                'white_numbers': sorted(white_selection),
                'powerball': powerball,
                'probability': 0.001
            }
        except Exception:
            return self._fallback_tool_prediction()
    
    def _recency_prediction(self) -> Dict:
        """Generate prediction based on recent draws using centralized service."""
        try:
            from .feature_engineering_service import FeatureEngineeringService
            feature_service = FeatureEngineeringService()
            
            recency_features = feature_service.get_prediction_features(self.df_history, 'recency')
            
            # Use candidate numbers (avoiding recent)
            candidates = recency_features.get('candidate_numbers', list(range(1, 70)))
            
            white_selection = random.sample(candidates[:50] if len(candidates) >= 5 else list(range(1, 70)), 5)
            powerball = random.randint(1, 26)
            
            return {
                'white_numbers': sorted(white_selection),
                'powerball': powerball,
                'probability': 0.001
            }
        except Exception:
            return self._fallback_tool_prediction()
    
    def _trends_prediction(self) -> Dict:
        """Generate prediction based on trends analysis."""
        try:
            # Simple trend analysis - look at number increases/decreases
            recent_data = self.df_history.tail(5)
            trend_numbers = []
            
            for _, row in recent_data.iterrows():
                trend_numbers.extend([row['n1'], row['n2'], row['n3'], row['n4'], row['n5']])
            
            # Predict numbers in trending ranges
            avg_nums = [sum(trend_numbers[i::5]) // len(recent_data) for i in range(5)]
            white_selection = [max(1, min(69, num + random.randint(-5, 5))) for num in avg_nums]
            
            return {
                'white_numbers': sorted(white_selection),
                'powerball': random.randint(1, 26),
                'probability': 0.001
            }
        except Exception:
            return self._fallback_tool_prediction()
    
    def _combo_prediction(self) -> Dict:
        """Generate prediction based on combination analysis."""
        try:
            # Look for common pairs/triplets
            combinations = {}
            for _, row in self.df_history.iterrows():
                numbers = [row['n1'], row['n2'], row['n3'], row['n4'], row['n5']]
                for i in range(len(numbers)):
                    for j in range(i+1, len(numbers)):
                        pair = tuple(sorted([numbers[i], numbers[j]]))
                        combinations[pair] = combinations.get(pair, 0) + 1
            
            # Select from top combinations
            top_pairs = sorted(combinations.items(), key=lambda x: x[1], reverse=True)[:10]
            if top_pairs:
                selected_pair = random.choice(top_pairs)[0]
                remaining = random.sample([n for n in range(1, 70) if n not in selected_pair], 3)
                white_selection = list(selected_pair) + remaining
            else:
                white_selection = random.sample(range(1, 70), 5)
            
            return {
                'white_numbers': sorted(white_selection),
                'powerball': random.randint(1, 26),
                'probability': 0.001
            }
        except Exception:
            return self._fallback_tool_prediction()
    
    def _sum_prediction(self) -> Dict:
        """Generate prediction based on sum analysis."""
        try:
            # Calculate typical sum ranges
            sums = []
            for _, row in self.df_history.iterrows():
                total = row['n1'] + row['n2'] + row['n3'] + row['n4'] + row['n5']
                sums.append(total)
            
            target_sum = int(np.mean(sums)) if sums else 175
            
            # Generate numbers that sum to approximately target_sum
            white_selection = []
            remaining_sum = target_sum
            
            for i in range(4):
                if remaining_sum > 69:
                    num = random.randint(max(1, remaining_sum//3), min(69, remaining_sum//2))
                else:
                    num = random.randint(1, min(69, max(1, remaining_sum)))
                white_selection.append(num)
                remaining_sum -= num
            
            # Add final number
            final_num = max(1, min(69, remaining_sum))
            white_selection.append(final_num)
            
            return {
                'white_numbers': sorted(white_selection),
                'powerball': random.randint(1, 26),
                'probability': 0.001
            }
        except Exception:
            return self._fallback_tool_prediction()
    
    def _dow_prediction(self) -> Dict:
        """Generate prediction based on day of week analysis."""
        try:
            next_draw_date = get_next_powerball_draw_date()
            dow = next_draw_date.weekday()
            
            # Filter by day of week
            dow_data = self.df_history[pd.to_datetime(self.df_history['draw_date']).dt.weekday == dow]
            
            if len(dow_data) > 0:
                # Get numbers that appear on this day of week
                dow_numbers = []
                for _, row in dow_data.iterrows():
                    dow_numbers.extend([row['n1'], row['n2'], row['n3'], row['n4'], row['n5']])
                
                # Select from frequently appearing numbers on this day
                counter = Counter(dow_numbers)
                top_dow_numbers = [num for num, count in counter.most_common(30)]
                white_selection = random.sample(top_dow_numbers[:20] if len(top_dow_numbers) >= 20 else top_dow_numbers, 
                                              min(5, len(top_dow_numbers)))
                
                # Fill remaining slots if needed
                while len(white_selection) < 5:
                    num = random.randint(1, 69)
                    if num not in white_selection:
                        white_selection.append(num)
            else:
                white_selection = random.sample(range(1, 70), 5)
            
            return {
                'white_numbers': sorted(white_selection),
                'powerball': random.randint(1, 26),
                'probability': 0.001
            }
        except Exception:
            return self._fallback_tool_prediction()
    
    def _ml_prediction(self) -> Dict:
        """Generate prediction using trained ML models."""
        try:
            if self.models['white_balls'] is None or self.models['powerball'] is None:
                self._train_models()
            
            if self.models['white_balls'] is not None:
                # Create feature vector for next prediction
                X_pred = self._create_prediction_features()
                
                # Predict white balls
                white_pred = self.models['white_balls'].predict([X_pred])[0]
                # Ensure white_pred is iterable and numeric; handle scalar/non-numeric outputs gracefully
                try:
                    # Normalize to an array-like structure
                    pred_array = np.atleast_1d(white_pred)
                    numeric_selection = []
                    for num in pred_array:
                        try:
                            # Try to coerce each element to a numeric value
                            val = int(round(float(num)))
                            val = max(1, min(69, val))
                            numeric_selection.append(val)
                        except Exception:
                            # Non-numeric entry encountered, trigger fallback
                            numeric_selection = []
                            break

                    if not numeric_selection:
                        raise ValueError("Non-numeric or empty prediction from model")

                    # Remove duplicates while preserving order, then ensure we have 5 numbers
                    seen = {}
                    uniq_ordered = [x for x in numeric_selection if not (x in seen or seen.setdefault(x, True))]
                    white_selection = uniq_ordered[:5]
                    while len(white_selection) < 5:
                        n = random.randint(1, 69)
                        if n not in white_selection:
                            white_selection.append(n)
                    white_selection = sorted(white_selection)

                except Exception:
                    # Fallback to a safe random unique selection if anything goes wrong
                    white_selection = sorted(random.sample(range(1, 70), 5))
                
                # Predict powerball
                powerball = None
                try:
                    pb_model = self.models.get('powerball')
                    if pb_model is not None and hasattr(pb_model, 'predict'):
                        pb_pred = pb_model.predict([X_pred])[0]
                        powerball = max(1, min(26, int(round(pb_pred))))
                    else:
                        raise ValueError("Powerball model unavailable or does not support predict()")
                except Exception as e:
                    logging.warning(f"Powerball prediction failed or model missing: {e}")
                    powerball = random.randint(1, 26)
                
                return {
                    'white_numbers': sorted(white_selection),
                    'powerball': powerball,
                    'probability': 0.001
                }
        except Exception as e:
            logging.warning(f"ML prediction failed: {e}")
        
        return self._fallback_tool_prediction()
    
    def _create_prediction_features(self) -> List:
        """Create feature vector for prediction."""
        try:
            # Use last known data point as template
            last_row = self.df_history.iloc[-1]
            
            # Basic features similar to training
            features = []
            
            # Recent frequency features
            freq_df = calc_frequency(self.df_history)
            frequency_dict = dict(zip(freq_df['number'], freq_df['frequency']))
            
            # Average frequencies for typical numbers
            typical_numbers = [10, 20, 30, 40, 50]  # Mid-range numbers
            freq_features = [frequency_dict.get(num, 5) for num in typical_numbers]
            features.extend(freq_features)
            
            # Sum and average of typical draw
            features.append(175)  # Typical sum
            features.append(35)   # Typical average
            
            # Next draw day of week
            next_draw = get_next_powerball_draw_date()
            features.append(next_draw.weekday())
            
            # Recent appearance features (zeros for new prediction)
            features.extend([0] * 5)
            
            return features
            
        except Exception:
            return [5, 10, 15, 20, 25, 175, 35, 2, 0, 0, 0, 0, 0]  # Fallback features
    
    def _combine_tool_predictions(self, tool_predictions: Dict) -> Dict:
        """Combine predictions from multiple tools using weighted approach."""
        try:
            # Collect all predicted numbers with weights
            white_votes = {}
            pb_votes = {}
            
            for tool, prediction in tool_predictions.items():
                weight = self.feature_weights.get(tool, 0.1)
                
                # Weight white ball votes
                for num in prediction.get('white_numbers', []):
                    white_votes[num] = white_votes.get(num, 0) + weight
                
                # Weight powerball votes
                pb = prediction.get('powerball', 1)
                pb_votes[pb] = pb_votes.get(pb, 0) + weight
            
            # Select top 5 white balls
            top_whites = sorted(white_votes.items(), key=lambda x: x[1], reverse=True)[:5]
            white_numbers = [num for num, votes in top_whites]
            
            # Fill remaining slots if needed
            while len(white_numbers) < 5:
                num = random.randint(1, 69)
                if num not in white_numbers:
                    white_numbers.append(num)
            
            # Select top powerball
            if pb_votes:
                powerball = max(pb_votes.items(), key=lambda x: x[1])[0]
            else:
                powerball = random.randint(1, 26)
            
            # Calculate combined probability (simplified)
            total_weight = sum(white_votes.values()) + sum(pb_votes.values())
            probability = min(0.1, total_weight / 100)  # Normalize
            
            return {
                'white_numbers': sorted(white_numbers),
                'powerball': powerball,
                'probability': probability
            }
            
        except Exception:
            return self._fallback_tool_prediction()
    
    def _fallback_tool_prediction(self) -> Dict:
        """Generate fallback prediction when tools fail."""
        return {
            'white_numbers': sorted(random.sample(range(1, 70), 5)),
            'powerball': random.randint(1, 26),
            'probability': 0.001
        }
    
    def _generate_fallback_predictions(self, count: int) -> List[Dict]:
        """Generate fallback predictions when main system fails."""
        predictions = []
        for i in range(count):
            prediction = {
                'white_numbers': sorted(random.sample(range(1, 70), 5)),
                'powerball': random.randint(1, 26),
                'probability': 0.001,
                'timestamp': datetime.datetime.now().isoformat(),
                'prediction_for_date': self._calculate_next_draw_date(),
                'storage_version': '3.0_sqlite_fallback'
            }
            predictions.append(prediction)
        return predictions

# Legacy compatibility functions
def get_enhanced_predictions(df_history: pd.DataFrame, count: int = 5) -> List[Dict]:
    """Legacy compatibility function - uses modernized system."""
    system = ModernizedPredictionSystem(df_history)
    return system.generate_weighted_predictions(count)

def update_model_with_new_draw(new_draw: Dict, df_history: Optional[pd.DataFrame] = None):
    """Legacy compatibility function - uses modernized system."""
    system = ModernizedPredictionSystem(df_history)
    system.update_with_new_draw(new_draw)