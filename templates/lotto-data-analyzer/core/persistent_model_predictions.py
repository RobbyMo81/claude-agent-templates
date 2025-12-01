"""
Persistent Model Predictions System
==================================
Implements SQLite-based storage for model-specific predictions with comprehensive data management.

This system ensures that predictions from each ML model (Ridge Regression, Random Forest, 
Gradient Boosting) are stored persistently, independently managed, and always accessible
without requiring model retraining.
"""

import sqlite3
import json
import datetime
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
from .datetime_manager import datetime_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Data class for model prediction records."""
    model_name: str
    prediction_id: str
    white_numbers: List[int]
    powerball: int
    probability: float
    features_used: List[int]
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_at: datetime.datetime
    prediction_set_id: str  # Groups 5 predictions together
    version: int = 1

class PersistentModelPredictionManager:
    """
    Manages persistent storage of model predictions using SQLite database.
    
    Key Features:
    - Model-specific prediction storage
    - Version control and retention policies
    - Data integrity checks
    - Independent model management
    """
    
    def __init__(self, db_path: str = "data/model_predictions.db"):
        """Initialize the prediction manager with SQLite database."""
        self.db_path = db_path
        self.ensure_database_directory()
        self.initialize_database()
        
    def ensure_database_directory(self):
        """Ensure the database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def initialize_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    prediction_id TEXT NOT NULL,
                    prediction_set_id TEXT NOT NULL,
                    white_numbers TEXT NOT NULL,  -- JSON array
                    powerball INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    features_used TEXT NOT NULL,  -- JSON array
                    hyperparameters TEXT NOT NULL,  -- JSON object
                    performance_metrics TEXT NOT NULL,  -- JSON object
                    created_at TIMESTAMP NOT NULL,
                    version INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE(model_name, prediction_id, version)
                )
            ''')
            
            # Create prediction sets table for grouping
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_sets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    set_id TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    is_current BOOLEAN DEFAULT TRUE,
                    total_predictions INTEGER DEFAULT 5,
                    training_duration REAL,
                    notes TEXT,
                    model_artifact_path TEXT
                )
            ''')
            
            # Add model_artifact_path column if it doesn't exist (for existing databases)
            try:
                cursor.execute('ALTER TABLE prediction_sets ADD COLUMN model_artifact_path TEXT')
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_name ON model_predictions(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prediction_set ON model_predictions(prediction_set_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_active ON model_predictions(is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_current_sets ON prediction_sets(is_current)')
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def store_model_predictions(self, 
                              model_name: str,
                              predictions: List[Dict[str, Any]],
                              hyperparameters: Dict[str, Any],
                              performance_metrics: Dict[str, float],
                              features_used: List[str],
                              training_duration: Optional[float] = None,
                              notes: Optional[str] = None,
                              model_artifact_path: Optional[str] = None) -> str:
        """
        Store model predictions and their associated metadata in the database.
        
        Args:
            model_name: Name of the model (e.g., 'Ridge Regression')
            predictions: List of predictions (typically 5)
            hyperparameters: Model configuration used
            performance_metrics: Training performance metrics
            features_used: List of feature names used for training.
            training_duration: Time taken for model training in seconds.
            notes: Additional notes about the prediction set.
            model_artifact_path: Path to the persisted model file.
        
        Returns:
            The unique ID of the created prediction set.
        """
        if len(predictions) < 1:
            raise ValueError(f"Expected at least 1 prediction, got {len(predictions)}")
        
        if len(predictions) != 5:
            logger.info(f"Storing {len(predictions)} predictions for {model_name} (non-standard count)")
        
        # Generate unique set ID with microseconds for uniqueness
        timestamp = datetime_manager.format_for_database()
        unique_timestamp = datetime_manager.get_utc_timestamp().replace(':', '').replace('-', '').replace('T', '_').replace('.', '_')
        set_id = f"{model_name.lower().replace(' ', '_')}_{unique_timestamp}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                # Mark previous prediction sets as not current
                cursor.execute('''
                    UPDATE prediction_sets 
                    SET is_current = FALSE 
                    WHERE model_name = ? AND is_current = TRUE
                ''', (model_name,))
                
                # Mark previous predictions as inactive
                cursor.execute('''
                    UPDATE model_predictions 
                    SET is_active = FALSE 
                    WHERE model_name = ? AND is_active = TRUE
                ''', (model_name,))
                
                # Insert into prediction_sets table with updated columns
                cursor.execute("""
                    INSERT OR REPLACE INTO prediction_sets 
                    (set_id, model_name, created_at, is_current, total_predictions, training_duration, notes, model_artifact_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (set_id, model_name, timestamp, True, len(predictions), training_duration, notes, model_artifact_path))
                
                # Insert individual predictions
                for i, pred in enumerate(predictions):
                    prediction_id = f"{set_id}_pred_{i+1}"
                    
                    cursor.execute('''
                        INSERT INTO model_predictions
                        (model_name, prediction_id, prediction_set_id, white_numbers, powerball,
                         probability, features_used, hyperparameters, performance_metrics,
                         created_at, version, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, TRUE)
                    ''', (
                        model_name,
                        prediction_id,
                        set_id,
                        json.dumps(pred['white_numbers']),
                        pred['powerball'],
                        pred.get('probability', 0.0),
                        json.dumps(features_used),
                        json.dumps(hyperparameters),
                        json.dumps(performance_metrics),
                        timestamp
                    ))
                
                conn.commit()
                logger.info(f"Stored {len(predictions)} predictions for {model_name} (set: {set_id})")
                return set_id
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error storing predictions for {model_name}: {e}")
                raise
    
    def get_current_predictions(self, model_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get the current active predictions for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of prediction dictionaries or None if no predictions exist
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT prediction_id, white_numbers, powerball, probability,
                       features_used, hyperparameters, performance_metrics, created_at
                FROM model_predictions
                WHERE model_name = ? AND is_active = TRUE
                ORDER BY prediction_id
            ''', (model_name,))
            
            rows = cursor.fetchall()
            if not rows:
                return None
            
            predictions = []
            for row in rows:
                pred_id, white_nums, powerball, prob, features, params, metrics, created = row
                
                if isinstance(powerball, bytes):
                    powerball = int.from_bytes(powerball, byteorder='little')
                
                white_numbers = json.loads(white_nums)
                if isinstance(white_numbers, list):
                    white_numbers = [int(num) if isinstance(num, bytes) else int(num) for num in white_numbers]
                
                predictions.append({
                    'prediction_id': pred_id,
                    'white_numbers': white_numbers,
                    'powerball': int(powerball),
                    'probability': float(prob),
                    'features_used': json.loads(features),
                    'hyperparameters': json.loads(params),
                    'performance_metrics': json.loads(metrics),
                    'created_at': created
                })
            
            return predictions
    
    def get_all_current_predictions(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get current predictions for all models.
        
        Returns:
            Dictionary mapping model names to their current predictions
        """
        models = ['Ridge Regression', 'Random Forest', 'Gradient Boosting']
        all_predictions = {}
        
        for model in models:
            predictions = self.get_current_predictions(model)
            if predictions:
                all_predictions[model] = predictions
        
        return all_predictions
    
    def get_prediction_history(self, model_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get prediction history for a model.
        
        Args:
            model_name: Name of the model
            limit: Maximum number of prediction sets to return
            
        Returns:
            List of prediction set information
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT set_id, created_at, is_current, total_predictions, training_duration, notes
                FROM prediction_sets
                WHERE model_name = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (model_name, limit))
            
            rows = cursor.fetchall()
            history = []
            
            for row in rows:
                set_id, created_at, is_current, total_preds, duration, notes = row
                history.append({
                    'set_id': set_id,
                    'created_at': created_at,
                    'is_current': bool(is_current),
                    'total_predictions': total_preds,
                    'training_duration': duration,
                    'notes': notes or ""
                })
            
            return history
    
    def store_predictions(self, model_name: str, predictions: List[Dict[str, Any]], 
                         features_used: List[str], hyperparameters: Dict[str, Any],
                         performance_metrics: Dict[str, float], training_duration: float = 0.0,
                         notes: str = "") -> str:
        """
        Store predictions for a model (standardized interface).
        
        Args:
            model_name: Name of the ML model
            predictions: List of prediction dictionaries
            features_used: List of feature names used in training
            hyperparameters: Model hyperparameters
            performance_metrics: Model performance metrics
            training_duration: Time taken to train the model
            notes: Optional notes about the training session
            
        Returns:
            Prediction set ID
        """
        return self.store_model_predictions(
            model_name=model_name,
            predictions=predictions,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            features_used=features_used,
            training_duration=training_duration,
            notes=notes
        )
    
    def get_predictions_by_model(self, model_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get predictions by model name (alias for get_current_predictions).
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of prediction dictionaries or None if no predictions exist
        """
        return self.get_current_predictions(model_name)
    
    def get_historical_predictions_by_model(self, model_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get ALL historical predictions for a specific model, regardless of is_active status.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of prediction dictionaries ordered chronologically (most recent first)
            or None if no predictions exist
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()            
            cursor.execute('''
                SELECT prediction_id, white_numbers, powerball, probability,
                       features_used, hyperparameters, performance_metrics, 
                       created_at, is_active, prediction_set_id
                FROM model_predictions
                WHERE model_name = ?
                ORDER BY created_at DESC, prediction_id
            ''', (model_name,))
            
            rows = cursor.fetchall()
            if not rows:
                return None
            
            predictions = []
            for row in rows:
                pred_id, white_nums, powerball, prob, features, params, metrics, created, is_active, set_id = row
                
                if isinstance(powerball, bytes):
                    try:
                        powerball = int.from_bytes(powerball, byteorder='little')
                    except Exception:
                        powerball = powerball[0] if len(powerball) > 0 else 0
                
                try:
                    white_numbers = json.loads(white_nums)
                    if isinstance(white_numbers, list):
                        white_numbers = [int(num) if isinstance(num, bytes) else int(num) for num in white_numbers]
                except (json.JSONDecodeError, ValueError):
                    white_numbers = [1, 2, 3, 4, 5]
                
                predictions.append({
                    'prediction_id': pred_id,
                    'white_numbers': white_numbers,
                    'powerball': int(powerball),
                    'probability': float(prob) if prob is not None else 0.0,
                    'features_used': self._safe_json_parse(features, []),
                    'hyperparameters': self._safe_json_parse(params, {}),
                    'performance_metrics': self._safe_json_parse(metrics, {}),
                    'created_at': created,
                    'is_active': bool(is_active),
                    'prediction_set_id': set_id
                })
            
            return predictions
    
    def _safe_json_parse(self, json_str: str, default_value: Any) -> Any:
        """
        Safely parse JSON string with fallback to default value.
        
        Args:
            json_str: JSON string to parse
            default_value: Value to return if parsing fails
            
        Returns:
            Parsed JSON object or default value
        """
        try:
            return json.loads(json_str) if json_str else default_value
        except (json.JSONDecodeError, TypeError):
            return default_value
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT model_name, COUNT(*) as total,
                       SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active
                FROM model_predictions
                GROUP BY model_name
            ''')
            model_stats = {row[0]: {'total': row[1], 'active': row[2]} for row in cursor.fetchall()}
            
            cursor.execute('SELECT COUNT(*) FROM prediction_sets')
            total_sets = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM prediction_sets WHERE is_current = TRUE')
            current_sets = cursor.fetchone()[0]
            
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                'model_statistics': model_stats,
                'total_prediction_sets': total_sets,
                'current_prediction_sets': current_sets,
                'database_size_bytes': db_size,
                'database_size_mb': round(db_size / (1024 * 1024), 2)
            }

class DatabaseMaintenanceManager:
    """
    Manages maintenance tasks for the prediction database, including data retention,
    integrity checks, and optimization.
    """
    def __init__(self, prediction_manager: 'PersistentModelPredictionManager', retention_limit: int = 30):
        """
        Initialize the maintenance manager.

        Args:
            prediction_manager: An instance of PersistentModelPredictionManager.
            retention_limit: The number of prediction sets to retain for each model.
        """
        self.manager = prediction_manager
        self.db_path = self.manager.db_path
        self.retention_limit = retention_limit

    def vacuum_database(self):
        """Reclaim unused space in the database and optimize its structure."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")
            logger.info("Database vacuumed successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error vacuuming database: {e}")
            raise

    def clear_all_predictions(self):
        """Clear all prediction data from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM model_predictions")
                cursor.execute("DELETE FROM prediction_sets")
                conn.commit()
            logger.info("All predictions cleared from the database.")
        except sqlite3.Error as e:
            logger.error(f"Error clearing predictions: {e}")
            raise

    def apply_retention_policy(self) -> Dict[str, int]:
        """
        Apply the data retention policy by removing the oldest prediction sets.

        Returns:
            A dictionary containing the number of sets removed.
        """
        sets_removed_count = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT model_name FROM prediction_sets")
                models = [row[0] for row in cursor.fetchall()]

                for model in models:
                    cursor.execute("""
                        SELECT set_id FROM prediction_sets WHERE model_name = ?
                        ORDER BY created_at DESC
                    """, (model,))
                    
                    sets = [row[0] for row in cursor.fetchall()]
                    if len(sets) > self.retention_limit:
                        sets_to_remove = sets[self.retention_limit:]
                        placeholders = ','.join(['?'] * len(sets_to_remove))
                        cursor.execute(f"DELETE FROM prediction_sets WHERE set_id IN ({placeholders})", sets_to_remove)
                        cursor.execute(f"DELETE FROM model_predictions WHERE prediction_set_id IN ({placeholders})", sets_to_remove)
                        sets_removed_count += len(sets_to_remove)
                conn.commit()
            logger.info(f"Retention policy applied, removed {sets_removed_count} sets.")
        except sqlite3.Error as e:
            logger.error(f"Error applying retention policy: {e}")
            raise
        return {"sets_removed": sets_removed_count}

    def run_data_integrity_checks(self) -> Dict[str, Any]:
        """
        Run data integrity checks to find orphaned records or inconsistencies.

        Returns:
            A dictionary with check results, including issues found.
        """
        results = {
            "timestamp": datetime_manager.get_utc_timestamp(),
            "checks_performed": [],
            "issues_found": [],
            "recommendations": []
        }
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Check for orphaned predictions
                cursor.execute("""
                    SELECT mp.prediction_set_id FROM model_predictions mp
                    LEFT JOIN prediction_sets ps ON mp.prediction_set_id = ps.set_id
                    WHERE ps.set_id IS NULL
                """)
                orphaned_preds = cursor.fetchall()
                results["checks_performed"].append("Orphaned Predictions Check")
                if orphaned_preds:
                    results["issues_found"].append(f"Found {len(orphaned_preds)} orphaned prediction records.")
                    results["recommendations"].append("Run cleanup to remove orphaned prediction records.")
        except sqlite3.Error as e:
            logger.error(f"Error during integrity check: {e}")
            results["issues_found"].append(f"Database error: {e}")
        return results

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the prediction database."""
        return self.manager.get_database_stats()


# Global instances for easy access
_prediction_manager = None
_maintenance_manager = None

def get_prediction_manager() -> PersistentModelPredictionManager:
    """Get the global prediction manager instance."""
    global _prediction_manager
    if _prediction_manager is None:
        _prediction_manager = PersistentModelPredictionManager()
    return _prediction_manager

def get_maintenance_manager() -> 'DatabaseMaintenanceManager':
    """Get the global maintenance manager instance."""
    global _maintenance_manager
    if _maintenance_manager is None:
        _maintenance_manager = DatabaseMaintenanceManager(get_prediction_manager())
    return _maintenance_manager

def get_current_model_paths() -> Dict[str, str]:
    """
    Get the file paths for all models marked as 'current'.

    This function queries the database for all prediction sets where `is_current`
    is true, and returns a dictionary mapping each model's name to its
    persisted artifact path.

    Returns:
        A dictionary mapping model names to their file paths.
    """
    db_path = "data/model_predictions.db"
    model_paths = {}
    if not os.path.exists(db_path):
        return model_paths

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT model_name, model_artifact_path FROM prediction_sets
            WHERE is_current = TRUE AND model_artifact_path IS NOT NULL
        """)
        for row in cursor.fetchall():
            model_paths[row[0]] = row[1]
    return model_paths