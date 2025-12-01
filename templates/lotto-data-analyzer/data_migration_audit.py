#!/usr/bin/env python3
"""
Data Migration Audit Script - Phase 4
=====================================
Audits the current state of migrated prediction data in the SQLite database
to verify historical predictions exist and document their is_active status.
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def audit_migrated_predictions():
    """
    Comprehensive audit of migrated prediction data in the SQLite database.
    
    This audit will:
    1. Verify database exists and is accessible
    2. Check for migrated historical predictions
    3. Document is_active status of all records
    4. Analyze prediction distribution by model
    5. Generate audit report
    """
    
    db_path = Path("data/model_predictions.db")
    
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return
    
    print("=" * 60)
    print("DATA MIGRATION AUDIT - PHASE 4")
    print("=" * 60)
    print(f"Database Path: {db_path}")
    print(f"Audit Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Verify database schema
            print("1. DATABASE SCHEMA VERIFICATION")
            print("-" * 40)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"Available tables: {tables}")
            
            if 'model_predictions' not in tables:
                print("ERROR: model_predictions table not found!")
                return
            
            # 2. Get table schema
            cursor.execute("PRAGMA table_info(model_predictions)")
            columns = cursor.fetchall()
            print("\nmodel_predictions table schema:")
            for col in columns:
                print(f"  {col[1]} ({col[2]}) - NOT NULL: {bool(col[3])}")
            print()
            
            # 3. Count all predictions
            print("2. PREDICTION DATA ANALYSIS")
            print("-" * 40)
            cursor.execute("SELECT COUNT(*) FROM model_predictions")
            total_predictions = cursor.fetchone()[0]
            print(f"Total predictions in database: {total_predictions}")
            
            # 4. Analyze is_active status
            cursor.execute("SELECT is_active, COUNT(*) FROM model_predictions GROUP BY is_active")
            active_status = cursor.fetchall()
            print("\nPredictions by is_active status:")
            for status, count in active_status:
                status_text = "ACTIVE" if status else "INACTIVE"
                print(f"  {status_text}: {count} predictions")
            
            # 5. Analyze by model name
            cursor.execute("SELECT model_name, COUNT(*) FROM model_predictions GROUP BY model_name ORDER BY COUNT(*) DESC")
            model_distribution = cursor.fetchall()
            print(f"\nPredictions by model (Total models: {len(model_distribution)}):")
            for model, count in model_distribution:
                print(f"  {model}: {count} predictions")
            
            # 6. Check for legacy migration markers
            print("\n3. LEGACY MIGRATION VERIFICATION")
            print("-" * 40)
            cursor.execute("SELECT COUNT(*) FROM model_predictions WHERE prediction_set_id LIKE '%legacy%'")
            legacy_count = cursor.fetchone()[0]
            print(f"Predictions with 'legacy' in prediction_set_id: {legacy_count}")
            
            # Check for migration patterns in prediction_id or model_name
            cursor.execute("SELECT COUNT(*) FROM model_predictions WHERE prediction_id LIKE '%legacy%'")
            legacy_pred_id_count = cursor.fetchone()[0]
            print(f"Predictions with 'legacy' in prediction_id: {legacy_pred_id_count}")
            
            # 7. Sample predictions analysis
            print("\n4. SAMPLE PREDICTIONS ANALYSIS")
            print("-" * 40)
            cursor.execute("""
                SELECT model_name, prediction_id, prediction_set_id, white_numbers, powerball, is_active, created_at
                FROM model_predictions 
                ORDER BY created_at DESC
                LIMIT 10
            """)
            
            samples = cursor.fetchall()
            if samples:
                print("Most recent 10 predictions:")
                for i, (model, pred_id, set_id, white_nums, pb, active, created) in enumerate(samples, 1):
                    print(f"\n  Sample {i}:")
                    print(f"    Model: {model}")
                    print(f"    Prediction ID: {pred_id}")
                    print(f"    Set ID: {set_id}")
                    print(f"    White Numbers: {white_nums}")
                    print(f"    Powerball: {pb}")
                    print(f"    Is Active: {active}")
                    print(f"    Created: {created}")
            else:
                print("No predictions found")
                
            # Check oldest predictions
            print("\n5. OLDEST PREDICTIONS ANALYSIS")
            print("-" * 40)
            cursor.execute("""
                SELECT model_name, prediction_id, prediction_set_id, white_numbers, powerball, is_active, created_at
                FROM model_predictions 
                ORDER BY created_at ASC
                LIMIT 5
            """)
            
            oldest = cursor.fetchall()
            if oldest:
                print("Oldest 5 predictions (potentially migrated):")
                for i, (model, pred_id, set_id, white_nums, pb, active, created) in enumerate(oldest, 1):
                    print(f"\n  Record {i}:")
                    print(f"    Model: {model}")
                    print(f"    Prediction ID: {pred_id}")
                    print(f"    Set ID: {set_id}")
                    print(f"    White Numbers: {white_nums}")
                    print(f"    Powerball: {pb}")
                    print(f"    Is Active: {active}")
                    print(f"    Created: {created}")
            else:
                print("No oldest predictions found")
            
            # 8. Check current predictions vs legacy predictions
            print("\n5. CURRENT VS LEGACY ANALYSIS")
            print("-" * 40)
            cursor.execute("""
                SELECT 
                    CASE WHEN prediction_set_id LIKE '%legacy%' THEN 'Legacy' ELSE 'Current' END as type,
                    COUNT(*),
                    AVG(CASE WHEN is_active THEN 1.0 ELSE 0.0 END) * 100 as active_percentage
                FROM model_predictions 
                GROUP BY CASE WHEN prediction_set_id LIKE '%legacy%' THEN 'Legacy' ELSE 'Current' END
            """)
            
            type_analysis = cursor.fetchall()
            for pred_type, count, active_pct in type_analysis:
                print(f"  {pred_type} predictions: {count} (Active: {active_pct:.1f}%)")
            
            # 9. Recent activity check
            print("\n6. RECENT ACTIVITY CHECK")
            print("-" * 40)
            cursor.execute("""
                SELECT created_at, COUNT(*) 
                FROM model_predictions 
                WHERE created_at IS NOT NULL 
                GROUP BY DATE(created_at) 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            
            recent_activity = cursor.fetchall()
            if recent_activity:
                print("Recent prediction activity by date:")
                for date, count in recent_activity:
                    print(f"  {date}: {count} predictions")
            else:
                print("No recent activity data found")
            
            print("\n" + "=" * 60)
            print("AUDIT COMPLETED SUCCESSFULLY")
            print("=" * 60)
            
    except Exception as e:
        print(f"ERROR during audit: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    audit_migrated_predictions()