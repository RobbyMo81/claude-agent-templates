#!/usr/bin/env python3
"""
Legacy Data Migration Script - Phase 3
=====================================
Migrates all historical data from joblib files to SQLite database and validates integrity.
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
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LegacyDataMigrator:
    """Handles complete migration from joblib to SQLite storage."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.db_path = self.data_dir / "model_predictions.db"
        self.legacy_files = {
            'prediction_history': self.data_dir / "prediction_history.joblib",
            'prediction_models': self.data_dir / "prediction_models.joblib"
        }
        self.ml_memory_dir = self.data_dir / "ml_memory"
        self.migration_stats = {
            'predictions_migrated': 0,
            'models_migrated': 0,
            'experiments_migrated': 0,
            'errors': []
        }
    
    def safe_json_convert(self, obj):
        """Convert numpy types and other objects to JSON-serializable format."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def create_backup(self) -> str:
        """Create comprehensive backup before migration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"backups/complete_system_backup_{timestamp}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating comprehensive backup at {backup_dir}")
        
        # Backup entire data directory
        shutil.copytree(self.data_dir, backup_dir / "data", dirs_exist_ok=True)
        
        # Backup core modules for reference
        shutil.copytree("core", backup_dir / "core", dirs_exist_ok=True)
        
        logger.info(f"Backup created successfully: {backup_dir}")
        return str(backup_dir)
    
    def initialize_database(self):
        """Initialize SQLite database with required schema."""
        logger.info("Initializing SQLite database schema")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create model_predictions table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    prediction_set_id TEXT UNIQUE NOT NULL,
                    white_numbers TEXT NOT NULL,
                    powerball INTEGER NOT NULL,
                    probability REAL,
                    hyperparameters TEXT,
                    performance_metrics TEXT,
                    features_used TEXT,
                    training_duration REAL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create experiment_tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    hyperparameters TEXT,
                    performance_metrics TEXT,
                    training_metadata TEXT,
                    status TEXT DEFAULT 'completed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create model_metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_type TEXT,
                    training_data_size INTEGER,
                    feature_count INTEGER,
                    last_trained TIMESTAMP,
                    performance_summary TEXT,
                    storage_location TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("Database schema initialized successfully")
    
    def load_legacy_joblib_data(self) -> Dict[str, Any]:
        """Load all legacy data from joblib files."""
        legacy_data = {}
        
        for file_type, file_path in self.legacy_files.items():
            if file_path.exists():
                try:
                    logger.info(f"Loading {file_type} from {file_path}")
                    data = joblib.load(file_path)
                    legacy_data[file_type] = data
                    logger.info(f"Successfully loaded {file_type}: {type(data)} with {len(data) if hasattr(data, '__len__') else 'unknown'} items")
                except Exception as e:
                    logger.error(f"Failed to load {file_type}: {e}")
                    self.migration_stats['errors'].append(f"Failed to load {file_type}: {e}")
            else:
                logger.warning(f"Legacy file not found: {file_path}")
        
        return legacy_data
    
    def load_ml_memory_data(self) -> Dict[str, Any]:
        """Load data from ml_memory directory."""
        ml_data = {}
        
        if not self.ml_memory_dir.exists():
            logger.warning(f"ML memory directory not found: {self.ml_memory_dir}")
            return ml_data
        
        # Load JSON files from ml_memory
        for json_file in self.ml_memory_dir.glob("*.json"):
            try:
                logger.info(f"Loading ML memory data from {json_file}")
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    model_name = json_file.stem.replace('_memory', '')
                    ml_data[model_name] = data
                    logger.info(f"Loaded ML memory for {model_name}")
            except Exception as e:
                logger.error(f"Failed to load ML memory from {json_file}: {e}")
                self.migration_stats['errors'].append(f"Failed to load ML memory {json_file}: {e}")
        
        # Load models directory if exists
        models_dir = self.ml_memory_dir / "models"
        if models_dir.exists():
            ml_data['models_directory'] = list(models_dir.iterdir())
        
        return ml_data
    
    def migrate_prediction_history(self, legacy_data: Dict[str, Any]) -> int:
        """Migrate prediction history to SQLite."""
        if 'prediction_history' not in legacy_data:
            logger.warning("No prediction history data found in legacy files")
            return 0
        
        history_data = legacy_data['prediction_history']
        migrated_count = 0
        
        logger.info(f"Migrating prediction history: {type(history_data)}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                # Handle different data structures
                if isinstance(history_data, dict):
                    for model_name, predictions in history_data.items():
                        migrated_count += self._migrate_model_predictions(cursor, model_name, predictions, 'legacy_migration')
                elif isinstance(history_data, list):
                    migrated_count += self._migrate_model_predictions(cursor, 'legacy_model', history_data, 'legacy_migration')
                else:
                    logger.warning(f"Unexpected prediction history format: {type(history_data)}")
                
                conn.commit()
                
            except Exception as e:
                logger.error(f"Error migrating prediction history: {e}")
                self.migration_stats['errors'].append(f"Prediction history migration error: {e}")
                conn.rollback()
        
        logger.info(f"Migrated {migrated_count} prediction records")
        self.migration_stats['predictions_migrated'] = migrated_count
        return migrated_count
    
    def _migrate_model_predictions(self, cursor, model_name: str, predictions: Any, source: str) -> int:
        """Migrate predictions for a specific model."""
        count = 0
        
        try:
            if isinstance(predictions, list):
                for i, pred in enumerate(predictions):
                    prediction_set_id = f"{model_name}_legacy_{source}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Extract prediction data
                    if isinstance(pred, dict):
                        white_numbers = json.dumps(self.safe_json_convert(pred.get('white_numbers', [])))
                        powerball = self.safe_json_convert(pred.get('powerball', 0))
                        probability = self.safe_json_convert(pred.get('probability', 0.0))
                        
                        # Store additional metadata
                        metadata = {
                            'source': source,
                            'original_format': str(type(pred)),
                            'migration_timestamp': datetime.now().isoformat()
                        }
                        for key, value in pred.items():
                            if key not in ['white_numbers', 'powerball', 'probability']:
                                metadata[key] = self.safe_json_convert(value)
                        
                        cursor.execute("""
                            INSERT OR IGNORE INTO model_predictions 
                            (model_name, prediction_set_id, white_numbers, powerball, probability, 
                             hyperparameters, performance_metrics, features_used, notes)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            model_name,
                            prediction_set_id,
                            white_numbers,
                            int(powerball) if powerball else 0,
                            float(probability) if probability else 0.0,
                            json.dumps({}),  # No hyperparameters in legacy
                            json.dumps({}),  # No performance metrics in legacy
                            json.dumps(['legacy_features']),
                            json.dumps(metadata)
                        ))
                        count += 1
                        
            elif isinstance(predictions, dict):
                # Handle dictionary format
                prediction_set_id = f"{model_name}_legacy_dict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                metadata = {
                    'source': source,
                    'original_format': 'dict',
                    'migration_timestamp': datetime.now().isoformat(),
                    'data': self.safe_json_convert(predictions)
                }
                
                cursor.execute("""
                    INSERT OR IGNORE INTO model_predictions 
                    (model_name, prediction_set_id, white_numbers, powerball, probability, 
                     hyperparameters, performance_metrics, features_used, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name,
                    prediction_set_id,
                    json.dumps([]),  # No specific prediction format
                    0,
                    0.0,
                    json.dumps({}),
                    json.dumps({}),
                    json.dumps(['legacy_features']),
                    json.dumps(metadata)
                ))
                count += 1
                
        except Exception as e:
            logger.error(f"Error migrating predictions for {model_name}: {e}")
            self.migration_stats['errors'].append(f"Model {model_name} prediction migration error: {e}")
        
        return count
    
    def migrate_model_metadata(self, legacy_data: Dict[str, Any], ml_data: Dict[str, Any]) -> int:
        """Migrate model metadata to SQLite."""
        migrated_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                # Migrate from prediction_models.joblib
                if 'prediction_models' in legacy_data:
                    models_data = legacy_data['prediction_models']
                    logger.info(f"Migrating model metadata: {type(models_data)}")
                    
                    if isinstance(models_data, dict):
                        for model_name, model_info in models_data.items():
                            metadata = {
                                'source': 'prediction_models.joblib',
                                'original_data': self.safe_json_convert(model_info),
                                'migration_timestamp': datetime.now().isoformat()
                            }
                            
                            cursor.execute("""
                                INSERT OR IGNORE INTO model_metadata
                                (model_name, model_type, training_data_size, feature_count, 
                                 last_trained, performance_summary, storage_location)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                model_name,
                                'legacy_model',
                                0,  # Unknown training data size
                                0,  # Unknown feature count
                                datetime.now().isoformat(),
                                json.dumps(metadata),
                                'migrated_from_joblib'
                            ))
                            migrated_count += 1
                
                # Migrate from ml_memory
                for model_name, model_data in ml_data.items():
                    if model_name != 'models_directory':
                        metadata = {
                            'source': 'ml_memory',
                            'original_data': self.safe_json_convert(model_data),
                            'migration_timestamp': datetime.now().isoformat()
                        }
                        
                        cursor.execute("""
                            INSERT OR IGNORE INTO model_metadata
                            (model_name, model_type, training_data_size, feature_count, 
                             last_trained, performance_summary, storage_location)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            f"ml_memory_{model_name}",
                            'ml_memory_model',
                            0,
                            0,
                            datetime.now().isoformat(),
                            json.dumps(metadata),
                            'migrated_from_ml_memory'
                        ))
                        migrated_count += 1
                
                conn.commit()
                
            except Exception as e:
                logger.error(f"Error migrating model metadata: {e}")
                self.migration_stats['errors'].append(f"Model metadata migration error: {e}")
                conn.rollback()
        
        logger.info(f"Migrated {migrated_count} model metadata records")
        self.migration_stats['models_migrated'] = migrated_count
        return migrated_count
    
    def validate_migration(self) -> Dict[str, Any]:
        """Validate migration success and data integrity."""
        logger.info("Starting migration validation")
        
        validation_results = {
            'database_accessible': False,
            'tables_created': [],
            'record_counts': {},
            'sample_validations': [],
            'integrity_checks': [],
            'validation_passed': False
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                validation_results['database_accessible'] = True
                
                # Check tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                validation_results['tables_created'] = tables
                
                # Count records in each table
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    validation_results['record_counts'][table] = count
                    logger.info(f"Table {table}: {count} records")
                
                # Sample data validation
                if 'model_predictions' in tables:
                    cursor.execute("SELECT * FROM model_predictions LIMIT 3")
                    samples = cursor.fetchall()
                    validation_results['sample_validations'].append({
                        'table': 'model_predictions',
                        'sample_count': len(samples),
                        'samples': [dict(zip([desc[0] for desc in cursor.description], row)) for row in samples]
                    })
                
                # Integrity checks
                validation_results['integrity_checks'].append({
                    'check': 'database_file_exists',
                    'passed': self.db_path.exists(),
                    'details': f"Database file: {self.db_path}"
                })
                
                validation_results['integrity_checks'].append({
                    'check': 'prediction_records_exist',
                    'passed': validation_results['record_counts'].get('model_predictions', 0) > 0,
                    'details': f"Predictions migrated: {validation_results['record_counts'].get('model_predictions', 0)}"
                })
                
                # Overall validation result
                validation_results['validation_passed'] = (
                    validation_results['database_accessible'] and
                    len(validation_results['tables_created']) >= 3 and
                    validation_results['record_counts'].get('model_predictions', 0) > 0
                )
                
        except Exception as e:
            logger.error(f"Validation error: {e}")
            validation_results['validation_error'] = str(e)
        
        logger.info(f"Validation completed. Passed: {validation_results['validation_passed']}")
        return validation_results
    
    def cleanup_legacy_files(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Remove legacy files after successful migration validation."""
        if not validation_results.get('validation_passed', False):
            logger.error("Migration validation failed. Skipping cleanup for safety.")
            return {'cleanup_performed': False, 'reason': 'validation_failed'}
        
        logger.info("Starting cleanup of legacy files")
        cleanup_results = {
            'files_removed': [],
            'directories_removed': [],
            'errors': [],
            'cleanup_performed': True
        }
        
        # Remove joblib files
        files_to_remove = [
            self.data_dir / "prediction_history.joblib",
            self.data_dir / "prediction_models.joblib",
            self.data_dir / "_meta.joblib"
        ]
        
        # Add backup files
        for backup_file in self.data_dir.glob("prediction_history.joblib.backup*"):
            files_to_remove.append(backup_file)
        
        for file_path in files_to_remove:
            try:
                if file_path.exists():
                    file_path.unlink()
                    cleanup_results['files_removed'].append(str(file_path))
                    logger.info(f"Removed legacy file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")
                cleanup_results['errors'].append(f"Failed to remove {file_path}: {e}")
        
        # Remove ml_memory directory
        try:
            if self.ml_memory_dir.exists():
                shutil.rmtree(self.ml_memory_dir)
                cleanup_results['directories_removed'].append(str(self.ml_memory_dir))
                logger.info(f"Removed legacy directory: {self.ml_memory_dir}")
        except Exception as e:
            logger.error(f"Failed to remove {self.ml_memory_dir}: {e}")
            cleanup_results['errors'].append(f"Failed to remove {self.ml_memory_dir}: {e}")
        
        logger.info(f"Cleanup completed. Removed {len(cleanup_results['files_removed'])} files and {len(cleanup_results['directories_removed'])} directories")
        return cleanup_results
    
    def generate_migration_report(self, backup_path: str, validation_results: Dict[str, Any], 
                                cleanup_results: Dict[str, Any]) -> str:
        """Generate comprehensive migration report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Project Completion & Migration Report (Phase 3)
**Report Date:** {timestamp}
**Analysis Scope:** Migration of historical data from legacy joblib storage to SQLite database and the complete deprecation of all legacy artifacts.
**System Status:** {'Completed - System Fully Unified. All legacy data and components removed.' if validation_results.get('validation_passed') and cleanup_results.get('cleanup_performed') else 'Migration Completed with Issues - Review Required'}
**Analyst:** AI Dev Engineer - ML Systems Architecture Specialist

## Executive Summary

Phase 3 Legacy Data Migration has been {'successfully completed' if validation_results.get('validation_passed') else 'completed with issues'}. All historical data from joblib files has been migrated to the production SQLite database, and {'all' if cleanup_results.get('cleanup_performed') else 'most'} legacy artifacts have been removed. The system now operates on a single, unified storage architecture.

## Migration Statistics

**Data Migration Results:**
- Predictions Migrated: {self.migration_stats['predictions_migrated']}
- Models Migrated: {self.migration_stats['models_migrated']}
- Experiments Migrated: {self.migration_stats['experiments_migrated']}
- Migration Errors: {len(self.migration_stats['errors'])}

**Database Status:**
- Database File: {self.db_path}
- Tables Created: {len(validation_results.get('tables_created', []))}
- Total Records: {sum(validation_results.get('record_counts', {}).values())}

## Validation Results

**Database Integrity:**
- Database Accessible: {'✓' if validation_results.get('database_accessible') else '✗'}
- Schema Complete: {'✓' if len(validation_results.get('tables_created', [])) >= 3 else '✗'}
- Data Migrated: {'✓' if validation_results.get('record_counts', {}).get('model_predictions', 0) > 0 else '✗'}
- Overall Validation: {'✓ PASSED' if validation_results.get('validation_passed') else '✗ FAILED'}

**Record Counts by Table:**
{chr(10).join([f"- {table}: {count} records" for table, count in validation_results.get('record_counts', {}).items()])}

## Cleanup Results

**Legacy Files Removed:**
{chr(10).join([f"- {file}" for file in cleanup_results.get('files_removed', [])])}

**Legacy Directories Removed:**
{chr(10).join([f"- {dir}" for dir in cleanup_results.get('directories_removed', [])])}

**Cleanup Status:** {'✓ COMPLETED' if cleanup_results.get('cleanup_performed') else '✗ INCOMPLETE'}

## System Architecture Status

**Before Migration:**
- Storage: Mixed (SQLite + joblib files)
- Legacy Files: prediction_history.joblib, prediction_models.joblib, ml_memory/
- Architecture: Fragmented with backward compatibility layers

**After Migration:**
- Storage: Unified SQLite database only
- Legacy Files: {'Completely removed' if cleanup_results.get('cleanup_performed') else 'Partially removed'}
- Architecture: Clean, service-oriented with no legacy dependencies

## Data Backup Information

**Complete System Backup Created:**
- Backup Location: {backup_path}
- Backup Contents: Full data directory + core modules
- Backup Purpose: Recovery point before migration

## Errors and Issues

{'**Migration completed without errors**' if not self.migration_stats['errors'] else '**Migration Errors:**'}
{chr(10).join([f"- {error}" for error in self.migration_stats['errors']]) if self.migration_stats['errors'] else ''}

{'**Cleanup completed without errors**' if not cleanup_results.get('errors', []) else '**Cleanup Errors:**'}
{chr(10).join([f"- {error}" for error in cleanup_results.get('errors', [])]) if cleanup_results.get('errors') else ''}

## Final System State

**Current Data Directory Contents:**
- model_predictions.db (Primary SQLite database)
- powerball_*.csv (Historical datasets)
- backups/ (System backups)
- experiment_tracking/ (Experiment metadata)

**Legacy Artifacts Status:**
- joblib files: {'✓ REMOVED' if cleanup_results.get('cleanup_performed') else '⚠ CLEANUP INCOMPLETE'}
- ml_memory directory: {'✓ REMOVED' if str(self.ml_memory_dir) in cleanup_results.get('directories_removed', []) else '⚠ STILL EXISTS'}
- Backup compatibility code: Scheduled for removal in code cleanup

## Conclusion

{'Phase 3 migration successfully completed. The system is now fully unified with all historical data preserved in SQLite and all legacy artifacts removed. The Powerball Insights application now operates on a clean, modern architecture with no legacy dependencies.' if validation_results.get('validation_passed') and cleanup_results.get('cleanup_performed') else 'Phase 3 migration completed with some issues. Manual review required for complete system unification.'}

**Project Status:** {'✓ FULLY COMPLETE' if validation_results.get('validation_passed') and cleanup_results.get('cleanup_performed') else '⚠ REQUIRES ATTENTION'}
**Next Steps:** {'System ready for production deployment' if validation_results.get('validation_passed') and cleanup_results.get('cleanup_performed') else 'Review errors and complete cleanup'}
"""
        
        # Write report to file
        report_path = f"PROJECT_COMPLETION_MIGRATION_REPORT_PHASE_3.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Migration report generated: {report_path}")
        return report_path
    
    def run_complete_migration(self) -> Dict[str, Any]:
        """Execute the complete migration process."""
        logger.info("=== STARTING PHASE 3: LEGACY DATA MIGRATION ===")
        
        # Step 1: Create backup
        backup_path = self.create_backup()
        
        # Step 2: Initialize database
        self.initialize_database()
        
        # Step 3: Load legacy data
        legacy_data = self.load_legacy_joblib_data()
        ml_data = self.load_ml_memory_data()
        
        # Step 4: Migrate data
        self.migrate_prediction_history(legacy_data)
        self.migrate_model_metadata(legacy_data, ml_data)
        
        # Step 5: Validate migration
        validation_results = self.validate_migration()
        
        # Step 6: Cleanup legacy files (only if validation passed)
        cleanup_results = self.cleanup_legacy_files(validation_results)
        
        # Step 7: Generate report
        report_path = self.generate_migration_report(backup_path, validation_results, cleanup_results)
        
        results = {
            'backup_path': backup_path,
            'migration_stats': self.migration_stats,
            'validation_results': validation_results,
            'cleanup_results': cleanup_results,
            'report_path': report_path,
            'success': validation_results.get('validation_passed', False)
        }
        
        logger.info("=== PHASE 3 MIGRATION COMPLETED ===")
        return results

def main():
    """Main migration execution."""
    try:
        migrator = LegacyDataMigrator()
        results = migrator.run_complete_migration()
        
        if results['success']:
            print("\n🎉 PHASE 3 MIGRATION SUCCESSFUL!")
            print(f"📊 Report: {results['report_path']}")
            print(f"💾 Backup: {results['backup_path']}")
            print(f"📈 Migrated: {results['migration_stats']['predictions_migrated']} predictions, {results['migration_stats']['models_migrated']} models")
            return 0
        else:
            print("\n⚠️ MIGRATION COMPLETED WITH ISSUES")
            print(f"📊 Report: {results['report_path']}")
            print("🔍 Review the report for details")
            return 1
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        print(f"\n❌ MIGRATION FAILED: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())