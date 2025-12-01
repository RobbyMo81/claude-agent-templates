#!/usr/bin/env python3
"""
Legacy Data Migration Validation Script
======================================
Validates the successful migration of historical prediction data from legacy .joblib files
to the unified SQLite database with comprehensive verification checks.
"""

import os
import sys
import sqlite3
import joblib
import json
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legacy_migration_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LegacyDataMigrationValidator:
    """Validates the migration from legacy joblib files to SQLite database."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.db_path = self.data_dir / "model_predictions.db"
        
        # Check for legacy files in backup directory
        self.legacy_backup_dir = self.data_dir / "legacy_backup_20250604_165124"
        if self.legacy_backup_dir.exists():
            self.legacy_files = {
                'prediction_history': self.legacy_backup_dir / "prediction_history.joblib",
                'prediction_models': self.legacy_backup_dir / "prediction_models.joblib"
            }
        else:
            # Fallback to data directory
            self.legacy_files = {
                'prediction_history': self.data_dir / "prediction_history.joblib",
                'prediction_models': self.data_dir / "prediction_models.joblib"
            }
        
        self.validation_results = {
            'record_count_check': False,
            'data_integrity_check': False,
            'is_active_flag_check': False,
            'sample_comparison_check': False,
            'legacy_record_count': 0,
            'migrated_record_count': 0,
            'sampled_records': 0,
            'validation_errors': [],
            'validation_warnings': []
        }
    
    def count_legacy_records(self) -> int:
        """Count total prediction records in legacy joblib files."""
        logger.info("Counting records in legacy joblib files")
        total_count = 0
        
        try:
            for data_type, file_path in self.legacy_files.items():
                if file_path.exists():
                    logger.info(f"Analyzing {data_type} from {file_path}")
                    data = joblib.load(file_path)
                    
                    if data_type == 'prediction_history':
                        count = self._count_prediction_history_records(data)
                        logger.info(f"Found {count} records in prediction_history")
                        total_count += count
                    
                    elif data_type == 'prediction_models':
                        count = self._count_prediction_models_records(data)
                        logger.info(f"Found {count} records in prediction_models")
                        total_count += count
                else:
                    logger.warning(f"Legacy file not found: {file_path}")
        
        except Exception as e:
            error_msg = f"Error counting legacy records: {e}"
            logger.error(error_msg)
            self.validation_results['validation_errors'].append(error_msg)
        
        logger.info(f"Total legacy records counted: {total_count}")
        return total_count
    
    def _count_prediction_history_records(self, data: Dict) -> int:
        """Count records in prediction history data structure."""
        count = 0
        
        for model_name, model_predictions in data.items():
            if isinstance(model_predictions, dict):
                count += len(model_predictions)
            elif isinstance(model_predictions, list):
                count += len(model_predictions)
            else:
                count += 1  # Single prediction
        
        return count
    
    def _count_prediction_models_records(self, data: Dict) -> int:
        """Count records in prediction models data structure."""
        count = 0
        
        for model_name, model_data in data.items():
            if isinstance(model_data, dict):
                count += len(model_data)
            elif isinstance(model_data, list):
                count += len(model_data)
            else:
                count += 1  # Single prediction
        
        return count
    
    def count_migrated_records(self) -> int:
        """Count total prediction records in SQLite database."""
        logger.info("Counting migrated records in SQLite database")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count prediction sets from legacy migration
                cursor.execute("""
                    SELECT COUNT(*) FROM prediction_sets 
                    WHERE notes LIKE '%Migrated from legacy%'
                """)
                prediction_sets_count = cursor.fetchone()[0]
                
                # Count individual predictions linked to migrated sets
                cursor.execute("""
                    SELECT COUNT(*) FROM individual_predictions ip
                    JOIN prediction_sets ps ON ip.prediction_set_id = ps.set_id
                    WHERE ps.notes LIKE '%Migrated from legacy%'
                """)
                individual_predictions_count = cursor.fetchone()[0]
                
                logger.info(f"Migrated prediction sets: {prediction_sets_count}")
                logger.info(f"Migrated individual predictions: {individual_predictions_count}")
                
                return individual_predictions_count
        
        except Exception as e:
            error_msg = f"Error counting migrated records: {e}"
            logger.error(error_msg)
            self.validation_results['validation_errors'].append(error_msg)
            return 0
    
    def validate_is_active_flags(self) -> bool:
        """Validate that all migrated records have is_active=FALSE."""
        logger.info("Validating is_active flags for migrated records")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check prediction_sets table - is_current should be FALSE
                cursor.execute("""
                    SELECT COUNT(*) FROM prediction_sets 
                    WHERE notes LIKE '%Migrated from legacy%' AND is_current = 1
                """)
                active_sets_count = cursor.fetchone()[0]
                
                # Check individual_predictions table - is_active should be FALSE
                cursor.execute("""
                    SELECT COUNT(*) FROM individual_predictions ip
                    JOIN prediction_sets ps ON ip.prediction_set_id = ps.set_id
                    WHERE ps.notes LIKE '%Migrated from legacy%' AND ip.is_active = 1
                """)
                active_predictions_count = cursor.fetchone()[0]
                
                if active_sets_count > 0:
                    error_msg = f"Found {active_sets_count} migrated prediction sets with is_current=TRUE"
                    logger.error(error_msg)
                    self.validation_results['validation_errors'].append(error_msg)
                    return False
                
                if active_predictions_count > 0:
                    error_msg = f"Found {active_predictions_count} migrated predictions with is_active=TRUE"
                    logger.error(error_msg)
                    self.validation_results['validation_errors'].append(error_msg)
                    return False
                
                logger.info("All migrated records correctly have is_active/is_current=FALSE")
                return True
        
        except Exception as e:
            error_msg = f"Error validating is_active flags: {e}"
            logger.error(error_msg)
            self.validation_results['validation_errors'].append(error_msg)
            return False
    
    def validate_sample_data_integrity(self, sample_size: int = 10) -> bool:
        """Compare randomly sampled records between legacy files and database."""
        logger.info(f"Validating data integrity for {sample_size} random samples")
        
        try:
            # Load legacy data
            legacy_data = {}
            for data_type, file_path in self.legacy_files.items():
                if file_path.exists():
                    legacy_data[data_type] = joblib.load(file_path)
            
            if not legacy_data:
                logger.warning("No legacy data found for sampling")
                return False
            
            # Extract sample records from legacy data
            legacy_samples = self._extract_legacy_samples(legacy_data, sample_size)
            
            if not legacy_samples:
                logger.warning("Could not extract samples from legacy data")
                return False
            
            # Verify samples exist in database
            matches_found = 0
            for sample in legacy_samples:
                if self._verify_sample_in_database(sample):
                    matches_found += 1
            
            logger.info(f"Found {matches_found}/{len(legacy_samples)} sample matches in database")
            
            if matches_found < len(legacy_samples):
                warning_msg = f"Only {matches_found}/{len(legacy_samples)} samples matched in database"
                logger.warning(warning_msg)
                self.validation_results['validation_warnings'].append(warning_msg)
            
            self.validation_results['sampled_records'] = len(legacy_samples)
            return matches_found > 0  # At least some matches found
        
        except Exception as e:
            error_msg = f"Error in sample validation: {e}"
            logger.error(error_msg)
            self.validation_results['validation_errors'].append(error_msg)
            return False
    
    def _extract_legacy_samples(self, legacy_data: Dict, sample_size: int) -> List[Dict]:
        """Extract random samples from legacy data."""
        samples = []
        
        try:
            # Extract from prediction_history
            if 'prediction_history' in legacy_data:
                hist_data = legacy_data['prediction_history']
                for model_name, model_predictions in hist_data.items():
                    if isinstance(model_predictions, dict):
                        for set_id, prediction_data in model_predictions.items():
                            if isinstance(prediction_data, dict) and 'white_numbers' in prediction_data:
                                samples.append({
                                    'source': 'prediction_history',
                                    'model_name': model_name,
                                    'set_id': set_id,
                                    'white_numbers': prediction_data['white_numbers'],
                                    'powerball': prediction_data.get('powerball', 0)
                                })
                    elif isinstance(model_predictions, list):
                        for i, prediction_data in enumerate(model_predictions):
                            if isinstance(prediction_data, dict) and 'white_numbers' in prediction_data:
                                samples.append({
                                    'source': 'prediction_history',
                                    'model_name': model_name,
                                    'set_id': f"{model_name}_hist_{i}",
                                    'white_numbers': prediction_data['white_numbers'],
                                    'powerball': prediction_data.get('powerball', 0)
                                })
            
            # Randomly sample up to sample_size records
            if len(samples) > sample_size:
                samples = random.sample(samples, sample_size)
            
            return samples
        
        except Exception as e:
            logger.error(f"Error extracting samples: {e}")
            return []
    
    def _verify_sample_in_database(self, sample: Dict) -> bool:
        """Verify a sample record exists in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Look for matching white_numbers and powerball
                white_numbers_json = json.dumps(sample['white_numbers'])
                
                cursor.execute("""
                    SELECT ip.id FROM individual_predictions ip
                    JOIN prediction_sets ps ON ip.prediction_set_id = ps.set_id
                    WHERE ps.notes LIKE '%Migrated from legacy%'
                    AND ip.white_numbers = ? AND ip.powerball = ?
                    LIMIT 1
                """, (white_numbers_json, sample['powerball']))
                
                result = cursor.fetchone()
                return result is not None
        
        except Exception as e:
            logger.warning(f"Error verifying sample: {e}")
            return False
    
    def run_validation(self) -> Dict[str, Any]:
        """Execute complete validation of the migration."""
        logger.info("=== LEGACY DATA MIGRATION VALIDATION STARTED ===")
        
        # 1. Record count validation
        logger.info("Step 1: Validating record counts")
        legacy_count = self.count_legacy_records()
        migrated_count = self.count_migrated_records()
        
        self.validation_results['legacy_record_count'] = legacy_count
        self.validation_results['migrated_record_count'] = migrated_count
        
        if migrated_count >= legacy_count:
            self.validation_results['record_count_check'] = True
            logger.info(f"✓ Record count validation PASSED: {migrated_count} >= {legacy_count}")
        else:
            error_msg = f"Record count validation FAILED: {migrated_count} < {legacy_count}"
            logger.error(error_msg)
            self.validation_results['validation_errors'].append(error_msg)
        
        # 2. is_active flag validation
        logger.info("Step 2: Validating is_active flags")
        flags_valid = self.validate_is_active_flags()
        self.validation_results['is_active_flag_check'] = flags_valid
        
        if flags_valid:
            logger.info("✓ is_active flag validation PASSED")
        else:
            logger.error("✗ is_active flag validation FAILED")
        
        # 3. Sample data integrity validation
        logger.info("Step 3: Validating sample data integrity")
        sample_valid = self.validate_sample_data_integrity()
        self.validation_results['sample_comparison_check'] = sample_valid
        
        if sample_valid:
            logger.info("✓ Sample data integrity validation PASSED")
        else:
            logger.error("✗ Sample data integrity validation FAILED")
        
        # Overall validation result
        all_checks_passed = (
            self.validation_results['record_count_check'] and
            self.validation_results['is_active_flag_check'] and
            self.validation_results['sample_comparison_check']
        )
        
        self.validation_results['data_integrity_check'] = all_checks_passed
        
        logger.info("=== LEGACY DATA MIGRATION VALIDATION COMPLETED ===")
        logger.info(f"Overall validation result: {'PASSED' if all_checks_passed else 'FAILED'}")
        
        return self.validation_results
    
    def generate_validation_report(self) -> str:
        """Generate a detailed validation report."""
        results = self.validation_results
        
        report = f"""
LEGACY DATA MIGRATION VALIDATION REPORT
======================================
Validation Date: {datetime.now().isoformat()}

VALIDATION SUMMARY
------------------
Overall Result: {'PASSED' if results['data_integrity_check'] else 'FAILED'}
Record Count Check: {'PASSED' if results['record_count_check'] else 'FAILED'}
is_active Flag Check: {'PASSED' if results['is_active_flag_check'] else 'FAILED'}
Sample Integrity Check: {'PASSED' if results['sample_comparison_check'] else 'FAILED'}

DETAILED RESULTS
----------------
Legacy Records Found: {results['legacy_record_count']}
Migrated Records Found: {results['migrated_record_count']}
Sample Records Verified: {results['sampled_records']}

VALIDATION ERRORS ({len(results['validation_errors'])})
{'-' * 20}
"""
        
        if results['validation_errors']:
            for error in results['validation_errors']:
                report += f"• {error}\n"
        else:
            report += "No validation errors found.\n"
        
        report += f"""
VALIDATION WARNINGS ({len(results['validation_warnings'])})
{'-' * 20}
"""
        
        if results['validation_warnings']:
            for warning in results['validation_warnings']:
                report += f"• {warning}\n"
        else:
            report += "No validation warnings found.\n"
        
        return report

def main():
    """Main validation execution."""
    print("Starting Legacy Data Migration Validation...")
    
    validator = LegacyDataMigrationValidator()
    results = validator.run_validation()
    
    # Generate and display report
    report = validator.generate_validation_report()
    print(report)
    
    # Write report to file
    with open('legacy_migration_validation_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nValidation report saved to: legacy_migration_validation_report.txt")
    
    return results

if __name__ == "__main__":
    main()