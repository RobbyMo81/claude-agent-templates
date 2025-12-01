"""
Prediction Accuracy Evaluation System
====================================
Compares submitted draw results against stored predictions to calculate accuracy metrics
and store them in the database for performance tracking over time.
"""

import sqlite3
import json
import datetime
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from .datetime_manager import datetime_manager
from .persistent_model_predictions import PersistentModelPredictionManager

logger = logging.getLogger(__name__)

@dataclass
class AccuracyResult:
    """Data class for accuracy evaluation results."""
    prediction_set_id: str
    model_name: str
    draw_date: str
    white_numbers_matched: int
    powerball_matched: bool
    total_matches: int
    accuracy_score: float
    evaluated_at: datetime.datetime

class PredictionAccuracyEvaluator:
    """
    Evaluates prediction accuracy by comparing actual draw results
    against stored predictions and tracking performance metrics.
    """
    
    def __init__(self):
        """Initialize the accuracy evaluator."""
        self.db_path = "data/model_predictions.db"
        self.prediction_manager = PersistentModelPredictionManager()
        self._ensure_accuracy_table()
    
    def _ensure_accuracy_table(self):
        """Ensure the accuracy tracking table exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create accuracy tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_accuracy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_set_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    draw_date TEXT NOT NULL,
                    actual_white_numbers TEXT NOT NULL,
                    actual_powerball INTEGER NOT NULL,
                    white_numbers_matched INTEGER NOT NULL,
                    powerball_matched BOOLEAN NOT NULL,
                    total_matches INTEGER NOT NULL,
                    accuracy_score REAL NOT NULL,
                    evaluated_at TEXT NOT NULL,
                    FOREIGN KEY (prediction_set_id) REFERENCES prediction_sets(set_id)
                )
            ''')
            
            # Create index for efficient queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_accuracy_model_date 
                ON prediction_accuracy(model_name, draw_date)
            ''')
            
            conn.commit()
    
    def evaluate_predictions_against_draw(self, draw_date: str, white_numbers: List[int], 
                                        powerball: int) -> List[AccuracyResult]:
        """
        Evaluate all stored predictions against actual draw results.
        
        Args:
            draw_date: Date of the draw (YYYY-MM-DD format)
            white_numbers: List of 5 white ball numbers
            powerball: Powerball number
            
        Returns:
            List of accuracy results for all models
        """
        if len(white_numbers) != 5:
            raise ValueError("White numbers must be exactly 5 numbers")
        
        white_set = set(white_numbers)
        accuracy_results = []
        
        # Get all models with current predictions
        models = ['Ridge Regression', 'Random Forest', 'Gradient Boosting']
        
        for model_name in models:
            current_predictions = self.prediction_manager.get_current_predictions(model_name)
            
            if not current_predictions:
                logger.info(f"No current predictions found for {model_name}")
                continue
            
            # Get the prediction set ID from the first prediction
            prediction_set_id = current_predictions[0].get('prediction_set_id', '')
            
            # Evaluate each prediction in the set
            model_results = []
            
            for prediction in current_predictions:
                pred_white_set = set(prediction['white_numbers'])
                pred_powerball = prediction['powerball']
                
                # Calculate matches
                white_matches = len(white_set.intersection(pred_white_set))
                powerball_match = (powerball == pred_powerball)
                total_matches = white_matches + (1 if powerball_match else 0)
                
                # Calculate accuracy score (weighted)
                # 5 white balls = 5 points, powerball = 1 point, max = 6 points
                accuracy_score = (white_matches + (1 if powerball_match else 0)) / 6.0
                
                model_results.append({
                    'white_matches': white_matches,
                    'powerball_match': powerball_match,
                    'total_matches': total_matches,
                    'accuracy_score': accuracy_score
                })
            
            # Calculate average accuracy for the model
            if model_results:
                avg_white_matches = sum(r['white_matches'] for r in model_results) / len(model_results)
                avg_powerball_matches = sum(1 for r in model_results if r['powerball_match'])
                avg_total_matches = sum(r['total_matches'] for r in model_results) / len(model_results)
                avg_accuracy_score = sum(r['accuracy_score'] for r in model_results) / len(model_results)
                
                accuracy_result = AccuracyResult(
                    prediction_set_id=prediction_set_id,
                    model_name=model_name,
                    draw_date=draw_date,
                    white_numbers_matched=round(avg_white_matches, 2),
                    powerball_matched=(avg_powerball_matches > 0),
                    total_matches=round(avg_total_matches, 2),
                    accuracy_score=avg_accuracy_score,
                    evaluated_at=datetime.datetime.now()
                )
                
                accuracy_results.append(accuracy_result)
        
        # Store results in database
        self._store_accuracy_results(accuracy_results, white_numbers, powerball)
        
        return accuracy_results
    
    def _store_accuracy_results(self, results: List[AccuracyResult], 
                               actual_white: List[int], actual_powerball: int):
        """Store accuracy results in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for result in results:
                cursor.execute('''
                    INSERT INTO prediction_accuracy (
                        prediction_set_id, model_name, draw_date,
                        actual_white_numbers, actual_powerball,
                        white_numbers_matched, powerball_matched,
                        total_matches, accuracy_score, evaluated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.prediction_set_id,
                    result.model_name,
                    result.draw_date,
                    json.dumps(actual_white),
                    actual_powerball,
                    result.white_numbers_matched,
                    result.powerball_matched,
                    result.total_matches,
                    result.accuracy_score,
                    datetime_manager.format_for_database(result.evaluated_at)
                ))
            
            conn.commit()
            logger.info(f"Stored accuracy results for {len(results)} models")
    
    def get_accuracy_history(self, model_name: Optional[str] = None, 
                           limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get historical accuracy results.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if model_name:
                cursor.execute('''
                    SELECT * FROM prediction_accuracy WHERE model_name = ? ORDER BY evaluated_at DESC LIMIT ?
                ''', (model_name, limit))
            else:
                cursor.execute('''
                    SELECT * FROM prediction_accuracy ORDER BY evaluated_at DESC LIMIT ?
                ''', (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_model_performance_summary(self, model_name: str) -> Dict[str, Any]:
        """
        Get performance summary for a specific model.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*), AVG(accuracy_score), AVG(white_numbers_matched),
                       SUM(CASE WHEN powerball_matched THEN 1 ELSE 0 END),
                       MAX(accuracy_score), MIN(accuracy_score)
                FROM prediction_accuracy WHERE model_name = ?
            ''', (model_name,))
            
            row = cursor.fetchone()
            if row and row[0] > 0:
                total, avg_acc, avg_white, pb_hits, best, worst = row
                return {
                    'model_name': model_name, 'total_evaluations': total,
                    'average_accuracy': round(avg_acc, 4),
                    'average_white_matches': round(avg_white, 2),
                    'powerball_hit_rate': round(pb_hits / total, 4) if total > 0 else 0,
                    'best_accuracy': round(best, 4), 'worst_accuracy': round(worst, 4)
                }
            else:
                return {'model_name': model_name, 'total_evaluations': 0}
    
    def get_overall_performance_comparison(self) -> List[Dict[str, Any]]:
        """
        Get performance comparison across all models.
        """
        models = ['Ridge Regression', 'Random Forest', 'Gradient Boosting']
        comparison = [self.get_model_performance_summary(model) for model in models]
        comparison.sort(key=lambda x: x.get('average_accuracy', 0), reverse=True)
        return comparison

# Global instance for easy access
_accuracy_evaluator = None

def get_accuracy_evaluator() -> PredictionAccuracyEvaluator:
    """Get the global accuracy evaluator instance."""
    global _accuracy_evaluator
    if _accuracy_evaluator is None:
        _accuracy_evaluator = PredictionAccuracyEvaluator()
    return _accuracy_evaluator