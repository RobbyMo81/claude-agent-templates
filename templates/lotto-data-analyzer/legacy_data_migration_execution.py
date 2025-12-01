#!/usr/bin/env python3
"""
Legacy Data Migration Execution Script
=====================================
Migrates all historical prediction data from legacy .joblib files to the unified SQLite database.
All migrated records are marked with is_active=FALSE and is_current=FALSE for historical archival.
"""

import os
import sys
import sqlite3
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legacy_migration_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LegacyDataMigrationExecutor:
    """Executes migration from legacy joblib files to SQLite database."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.db_path = self.data_dir / "model_predictions.db"
        
        # Check for legacy files in backup directory first
        self.legacy_backup_dir = self.data_dir / "legacy_backup_20250604_165124"
        if self.legacy_backup_dir.exists():
            self.legacy_files = {
                'prediction_history': self.legacy_backup_dir / "prediction_history.joblib",
                'prediction_models': self.legacy_backup_dir / "prediction_models.joblib"
            }
            self.ml_memory_dir = self.legacy_backup_dir / "ml_memory"
        else:
            # Fallback to data directory
            self.legacy_files = {
                'prediction_history': self.data_dir / "prediction_history.joblib",
                'prediction_models': self.data_dir / "prediction_models.joblib"
            }
            self.ml_memory_dir = self.data_dir / "ml_memory"
        
        self.migration_stats = {
            'predictions_migrated': 0,
            'models_migrated': 0,
            'errors': [],
            'start_time': None,
            'end_time': None
        }
    
    def safe_json_convert(self, obj):
        """Convert numpy types and other objects to JSON-serializable format."""
        if obj is None:
            return None
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self.safe_json_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.safe_json_convert(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def initialize_database_schema(self):
        """Verify existing database schema is compatible."""
        logger.info("Verifying database schema compatibility")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['prediction_sets', 'individual_predictions']
            for table in required_tables:
                if table not in tables:
                    logger.error(f"Required table {table} not found in database")
                    raise Exception(f"Database schema missing required table: {table}")
            
            logger.info("Database schema verification completed successfully")
    
    def load_legacy_joblib_data(self) -> Dict[str, Any]:
        """Load historical data from legacy joblib files."""
        logger.info("Loading legacy joblib data")
        legacy_data = {}
        
        for data_type, file_path in self.legacy_files.items():
            if file_path.exists():
                try:
                    logger.info(f"Loading {data_type} from {file_path}")
                    data = joblib.load(file_path)
                    legacy_data[data_type] = data
                    logger.info(f"Successfully loaded {data_type}: {type(data)} with {len(data) if hasattr(data, '__len__') else 'N/A'} items")
                except Exception as e:
                    error_msg = f"Error loading {data_type}: {e}"
                    logger.error(error_msg)
                    self.migration_stats['errors'].append(error_msg)
            else:
                logger.warning(f"Legacy file not found: {file_path}")
        
        return legacy_data
    
    def migrate_prediction_history(self, legacy_data: Dict[str, Any]) -> int:
        """Migrate prediction history to SQLite database."""
        logger.info("Starting prediction history migration")
        
        prediction_history = legacy_data.get('prediction_history', {})
        if not prediction_history:
            logger.warning("No prediction history found in legacy data")
            return 0
        
        migrated_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for model_name, model_predictions in prediction_history.items():
                logger.info(f"Migrating predictions for model: {model_name}")
                
                if isinstance(model_predictions, dict):
                    for set_id, prediction_data in model_predictions.items():
                        migrated_count += self._migrate_prediction_set(
                            cursor, model_name, set_id, prediction_data, "prediction_history"
                        )
                elif isinstance(model_predictions, list):
                    # Handle list format - create individual sets for each item
                    for i, prediction_data in enumerate(model_predictions):
                        set_id = f"{model_name}_hist_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        if model_name == 'predictions' and isinstance(prediction_data, dict) and 'white_numbers' in prediction_data:
                            # This is actual prediction data
                            migrated_count += self._migrate_prediction_set(
                                cursor, model_name, set_id, prediction_data, "prediction_history"
                            )
                        else:
                            # This is metadata/tracking data - store as metadata record
                            migrated_count += self._migrate_metadata_record(
                                cursor, model_name, set_id, prediction_data, "prediction_history"
                            )
            
            conn.commit()
        
        logger.info(f"Migrated {migrated_count} prediction records from history")
        return migrated_count
    
    def migrate_prediction_models(self, legacy_data: Dict[str, Any]) -> int:
        """Migrate prediction models data to SQLite database."""
        logger.info("Starting prediction models migration")
        
        prediction_models = legacy_data.get('prediction_models', {})
        if not prediction_models:
            logger.warning("No prediction models found in legacy data")
            return 0
        
        migrated_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for model_name, model_data in prediction_models.items():
                logger.info(f"Migrating model data for: {model_name}")
                
                if isinstance(model_data, dict):
                    for set_id, prediction_data in model_data.items():
                        migrated_count += self._migrate_prediction_set(
                            cursor, model_name, set_id, prediction_data, "prediction_models"
                        )
                elif isinstance(model_data, list):
                    # Handle list format
                    for i, prediction_data in enumerate(model_data):
                        set_id = f"{model_name}_models_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        migrated_count += self._migrate_prediction_set(
                            cursor, model_name, set_id, prediction_data, "prediction_models"
                        )
            
            conn.commit()
        
        logger.info(f"Migrated {migrated_count} prediction records from models")
        return migrated_count
    
    def _migrate_prediction_set(self, cursor, model_name: str, set_id: str, 
                               prediction_data: Any, source: str) -> int:
        """Migrate a single prediction set."""
        try:
            # Ensure unique set_id by adding source prefix
            unique_set_id = f"{source}_{set_id}"
            
            # Extract metadata
            created_at = datetime.now().isoformat()
            hyperparameters = "{}"
            performance_metrics = "{}"
            notes = f"Migrated from legacy {source}"
            
            # Handle different data formats
            predictions = []
            
            if isinstance(prediction_data, dict):
                # Extract predictions from dict format
                if 'predictions' in prediction_data:
                    predictions = prediction_data['predictions']
                elif 'white_numbers' in prediction_data:
                    # Single prediction format
                    predictions = [prediction_data]
                else:
                    # Dict with multiple predictions
                    for key, value in prediction_data.items():
                        if isinstance(value, dict) and 'white_numbers' in value:
                            predictions.append(value)
                
                # Extract metadata if available
                hyperparameters = json.dumps(self.safe_json_convert(
                    prediction_data.get('hyperparameters', {})
                ))
                performance_metrics = json.dumps(self.safe_json_convert(
                    prediction_data.get('performance_metrics', {})
                ))
                if 'created_at' in prediction_data:
                    created_at = str(prediction_data['created_at'])
            
            elif isinstance(prediction_data, list):
                predictions = prediction_data
            
            else:
                # Single prediction
                predictions = [prediction_data]
            
            # Insert prediction set - CRITICAL: is_current set to FALSE for historical data
            cursor.execute("""
                INSERT OR REPLACE INTO prediction_sets 
                (set_id, model_name, created_at, training_duration, notes, is_current, total_predictions)
                VALUES (?, ?, ?, ?, ?, 0, ?)
            """, (
                unique_set_id, model_name, created_at, 0.0, notes, len(predictions)
            ))
            
            # Insert individual predictions
            prediction_count = 0
            for i, pred in enumerate(predictions):
                if self._insert_individual_prediction(cursor, unique_set_id, pred, i + 1, created_at):
                    prediction_count += 1
            
            if prediction_count > 0:
                logger.info(f"Migrated prediction set {unique_set_id} with {prediction_count} predictions")
            
            return prediction_count
            
        except Exception as e:
            error_msg = f"Error migrating prediction set {set_id}: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            return 0
    
    def _insert_individual_prediction(self, cursor, set_id: str, prediction: Any, 
                                    rank: int, created_at: str) -> bool:
        """Insert an individual prediction record."""
        try:
            white_numbers = []
            powerball = 0
            probability = 0.0
            
            if isinstance(prediction, dict):
                white_numbers = prediction.get('white_numbers', [])
                powerball = prediction.get('powerball', 0)
                probability = prediction.get('probability', 0.0)
            elif isinstance(prediction, (list, tuple)) and len(prediction) >= 6:
                # Format: [w1, w2, w3, w4, w5, pb] or similar
                white_numbers = list(prediction[:5])
                powerball = int(prediction[5])
            
            # Validate data
            if not white_numbers or len(white_numbers) != 5:
                return False
            
            # Convert to JSON string for storage
            white_numbers_json = json.dumps([int(n) for n in white_numbers])
            
            # Insert with is_active=FALSE
            cursor.execute("""
                INSERT INTO individual_predictions 
                (prediction_set_id, white_numbers, powerball, probability, 
                 prediction_rank, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 0)
            """, (
                set_id, white_numbers_json, int(powerball), 
                float(probability), rank, created_at
            ))
            
            return True
            
        except Exception as e:
            logger.warning(f"Could not insert individual prediction: {e}")
            return False
    
    def _migrate_metadata_record(self, cursor, model_name: str, set_id: str, 
                                metadata: Any, source: str) -> int:
        """Migrate metadata records (accuracy, feedback) as prediction sets."""
        try:
            # Ensure unique set_id
            unique_set_id = f"{source}_{set_id}"
            created_at = datetime.now().isoformat()
            notes = f"Migrated metadata from legacy {source} - {model_name} data"
            
            # Store metadata as JSON in notes with default handler
            try:
                metadata_json = json.dumps(metadata, default=str)
            except:
                metadata_json = str(metadata)
            detailed_notes = f"{notes}: {metadata_json}"
            
            # Insert as prediction set with no individual predictions
            cursor.execute("""
                INSERT OR REPLACE INTO prediction_sets 
                (set_id, model_name, created_at, training_duration, notes, is_current, total_predictions)
                VALUES (?, ?, ?, ?, ?, 0, 0)
            """, (
                unique_set_id, f"{model_name}_metadata", created_at, 0.0, detailed_notes
            ))
            
            logger.info(f"Migrated metadata record {unique_set_id}")
            return 1
            
        except Exception as e:
            error_msg = f"Error migrating metadata record {set_id}: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            return 0
    
    def run_migration(self) -> Dict[str, Any]:
        """Execute the complete migration process."""
        logger.info("=== LEGACY DATA MIGRATION EXECUTION STARTED ===")
        self.migration_stats['start_time'] = datetime.now().isoformat()
        
        try:
            # Initialize database schema
            self.initialize_database_schema()
            
            # Load legacy data
            legacy_data = self.load_legacy_joblib_data()
            
            if not legacy_data:
                logger.error("No legacy data found to migrate")
                return self.migration_stats
            
            # Migrate prediction history
            history_count = self.migrate_prediction_history(legacy_data)
            self.migration_stats['predictions_migrated'] += history_count
            
            # Migrate prediction models
            models_count = self.migrate_prediction_models(legacy_data)
            self.migration_stats['predictions_migrated'] += models_count
            
            self.migration_stats['end_time'] = datetime.now().isoformat()
            
            logger.info("=== LEGACY DATA MIGRATION EXECUTION COMPLETED ===")
            logger.info(f"Total predictions migrated: {self.migration_stats['predictions_migrated']}")
            logger.info(f"Errors encountered: {len(self.migration_stats['errors'])}")
            
            return self.migration_stats
            
        except Exception as e:
            error_msg = f"Critical migration error: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            self.migration_stats['end_time'] = datetime.now().isoformat()
            return self.migration_stats

def main():
    """Main execution function."""
    print("Starting Legacy Data Migration Execution...")
    
    migrator = LegacyDataMigrationExecutor()
    results = migrator.run_migration()
    
    print("\n=== MIGRATION SUMMARY ===")
    print(f"Start Time: {results['start_time']}")
    print(f"End Time: {results['end_time']}")
    print(f"Predictions Migrated: {results['predictions_migrated']}")
    print(f"Errors: {len(results['errors'])}")
    
    if results['errors']:
        print("\nErrors encountered:")
        for error in results['errors']:
            print(f"  - {error}")
    
    return results

if __name__ == "__main__":
    main()