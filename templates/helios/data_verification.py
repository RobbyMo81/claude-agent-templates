"""
Data Verification Module for Helios PowerballNet
Comprehensive validation and quality checks for training data.
This module should be run before any training to ensure data integrity.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple, Optional

class PowerballDataVerifier:
    """Comprehensive data verification for Powerball training data."""
    
    def __init__(self, data_path: str):
        """
        Initialize data verifier.
        
        Args:
            data_path (str): Path to the Powerball CSV data file
        """
        self.data_path = data_path
        self.df = None
        self.verification_report = {}
        self.errors = []
        self.warnings = []
        
    def load_and_verify(self) -> bool:
        """
        Complete data verification workflow.
        
        Returns:
            bool: True if data passes all critical checks, False otherwise
        """
        print("🔍 HELIOS POWERBALL DATA VERIFICATION")
        print("=" * 60)
        
        # Step 1: Load data
        if not self._load_data():
            return False
            
        # Step 2: Basic validation
        if not self._basic_validation():
            return False
            
        # Step 3: Data quality checks
        self._data_quality_checks()
        
        # Step 4: Training suitability checks
        self._training_suitability_checks()
        
        # Step 5: Generate report
        self._generate_verification_report()
        
        # Step 6: Display summary
        self._display_summary()
        
        return len(self.errors) == 0
    
    def _load_data(self) -> bool:
        """Load and perform initial data validation."""
        print(f"📂 Loading data from: {self.data_path}")
        
        try:
            if not os.path.exists(self.data_path):
                self.errors.append(f"Data file not found: {self.data_path}")
                return False
                
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Successfully loaded {len(self.df)} records")
            return True
            
        except Exception as e:
            self.errors.append(f"Failed to load data: {str(e)}")
            return False
    
    def _basic_validation(self) -> bool:
        """Perform basic data structure validation."""
        print("\n🔎 Basic Data Structure Validation")
        print("-" * 40)
        
        # Check required columns
        required_columns = ['draw_date', 'wb1', 'wb2', 'wb3', 'wb4', 'wb5', 'pb']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            self.errors.append(f"Missing required columns: {missing_columns}")
            print(f"❌ Missing columns: {missing_columns}")
            return False
        
        print(f"✅ All required columns present: {required_columns}")
        
        # Check data types and convert
        try:
            self.df['draw_date'] = pd.to_datetime(self.df['draw_date'])
            for col in ['wb1', 'wb2', 'wb3', 'wb4', 'wb5', 'pb']:
                self.df[col] = pd.to_numeric(self.df[col])
            print("✅ Data types validated and converted")
            
        except Exception as e:
            self.errors.append(f"Data type conversion failed: {str(e)}")
            return False
            
        return True
    
    def _data_quality_checks(self):
        """Perform comprehensive data quality checks."""
        print("\n🔍 Data Quality Analysis")
        print("-" * 40)
        
        # Check for missing values
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            self.warnings.append(f"Found {missing_count} missing values")
            print(f"⚠️  Missing values: {missing_count}")
        else:
            print("✅ No missing values found")
        
        # Check for duplicates
        duplicate_dates = self.df['draw_date'].duplicated().sum()
        if duplicate_dates > 0:
            self.errors.append(f"Found {duplicate_dates} duplicate dates")
            print(f"❌ Duplicate dates: {duplicate_dates}")
        else:
            print("✅ No duplicate dates found")
        
        # Check date ordering
        is_sorted = self.df['draw_date'].is_monotonic_decreasing
        if not is_sorted:
            self.warnings.append("Data is not sorted by date (newest first)")
            print("⚠️  Data not sorted by date - will be sorted automatically")
        else:
            print("✅ Data properly sorted (newest first)")
        
        # Check number ranges
        self._validate_number_ranges()
        
        # Check data recency
        self._check_data_recency()
        
        # Check data span
        self._check_data_span()
    
    def _validate_number_ranges(self):
        """Validate Powerball number ranges."""
        print("\n🎯 Number Range Validation")
        print("-" * 30)
        
        # White balls should be 1-69
        white_ball_cols = ['wb1', 'wb2', 'wb3', 'wb4', 'wb5']
        white_min = self.df[white_ball_cols].min().min()
        white_max = self.df[white_ball_cols].max().max()
        
        if white_min < 1 or white_max > 69:
            self.errors.append(f"White ball numbers out of range: {white_min}-{white_max} (should be 1-69)")
            print(f"❌ White balls out of range: {white_min}-{white_max}")
        else:
            print(f"✅ White balls in valid range: {white_min}-{white_max}")
        
        # Powerball should be 1-26
        pb_min = self.df['pb'].min()
        pb_max = self.df['pb'].max()
        
        if pb_min < 1 or pb_max > 26:
            self.errors.append(f"Powerball numbers out of range: {pb_min}-{pb_max} (should be 1-26)")
            print(f"❌ Powerball out of range: {pb_min}-{pb_max}")
        else:
            print(f"✅ Powerball in valid range: {pb_min}-{pb_max}")
        
        # Check for duplicate white balls in same draw
        duplicate_whites = 0
        for idx, row in self.df.iterrows():
            white_numbers = [row['wb1'], row['wb2'], row['wb3'], row['wb4'], row['wb5']]
            if len(white_numbers) != len(set(white_numbers)):
                duplicate_whites += 1
        
        if duplicate_whites > 0:
            self.errors.append(f"Found {duplicate_whites} draws with duplicate white balls")
            print(f"❌ Duplicate white balls in {duplicate_whites} draws")
        else:
            print("✅ No duplicate white balls within draws")
    
    def _check_data_recency(self):
        """Check how recent the data is."""
        print("\n📅 Data Recency Check")
        print("-" * 25)
        
        latest_date = self.df['draw_date'].max()
        days_old = (datetime.now() - latest_date).days
        
        self.verification_report['latest_draw'] = latest_date.strftime('%Y-%m-%d')
        self.verification_report['days_old'] = days_old
        
        if days_old <= 7:
            print(f"✅ Data is current (latest: {latest_date.strftime('%Y-%m-%d')}, {days_old} days old)")
        elif days_old <= 30:
            print(f"⚠️  Data is somewhat outdated (latest: {latest_date.strftime('%Y-%m-%d')}, {days_old} days old)")
            self.warnings.append(f"Data is {days_old} days old")
        else:
            print(f"❌ Data is significantly outdated (latest: {latest_date.strftime('%Y-%m-%d')}, {days_old} days old)")
            self.errors.append(f"Data is {days_old} days old - may not be suitable for current predictions")
    
    def _check_data_span(self):
        """Check the span and coverage of data."""
        print("\n📊 Data Coverage Analysis")
        print("-" * 30)
        
        earliest_date = self.df['draw_date'].min()
        latest_date = self.df['draw_date'].max()
        span_days = (latest_date - earliest_date).days
        span_years = span_days / 365.25
        
        self.verification_report.update({
            'earliest_draw': earliest_date.strftime('%Y-%m-%d'),
            'latest_draw': latest_date.strftime('%Y-%m-%d'),
            'span_days': span_days,
            'span_years': round(span_years, 1),
            'total_draws': len(self.df)
        })
        
        print(f"📈 Date range: {earliest_date.strftime('%Y-%m-%d')} → {latest_date.strftime('%Y-%m-%d')}")
        print(f"📏 Time span: {span_years:.1f} years ({span_days} days)")
        print(f"🎯 Total draws: {len(self.df)}")
        
        # Expected draws (roughly 3 per week)
        expected_draws = span_days * 3 / 7
        coverage_ratio = len(self.df) / expected_draws
        
        print(f"📊 Coverage: {coverage_ratio:.1%} of expected draws")
        
        if coverage_ratio < 0.8:
            self.warnings.append(f"Low data coverage: {coverage_ratio:.1%} of expected draws")
        
        # Check for sufficient training data
        if len(self.df) < 100:
            self.errors.append("Insufficient data for training (< 100 draws)")
        elif len(self.df) < 500:
            self.warnings.append("Limited training data (< 500 draws)")
    
    def _training_suitability_checks(self):
        """Check data suitability for neural network training."""
        print("\n🧠 Training Suitability Analysis")
        print("-" * 35)
        
        # Calculate training sequences for different sequence lengths
        sequence_lengths = [15, 20, 30, 50]
        
        for seq_len in sequence_lengths:
            total_sequences = max(0, len(self.df) - seq_len)
            val_split = 0.2
            train_sequences = int(total_sequences * (1 - val_split))
            val_sequences = total_sequences - train_sequences
            
            status = "✅" if train_sequences >= 500 else "⚠️" if train_sequences >= 200 else "❌"
            print(f"{status} Seq Length {seq_len:2d}: {train_sequences:4d} train, {val_sequences:3d} val sequences")
            
            if seq_len == 30:  # Default sequence length
                self.verification_report['training_sequences'] = {
                    'sequence_length': seq_len,
                    'total_sequences': total_sequences,
                    'train_sequences': train_sequences,
                    'val_sequences': val_sequences
                }
    
    def _generate_verification_report(self):
        """Generate comprehensive verification report."""
        self.verification_report.update({
            'verification_timestamp': datetime.now().isoformat(),
            'data_path': self.data_path,
            'errors': self.errors,
            'warnings': self.warnings,
            'status': 'PASSED' if len(self.errors) == 0 else 'FAILED'
        })
        
        # Save report
        report_path = os.path.join(os.path.dirname(self.data_path), 'data_verification_report.json')
        try:
            with open(report_path, 'w') as f:
                json.dump(self.verification_report, f, indent=2, default=str)
            print(f"\n📄 Verification report saved: {report_path}")
        except Exception as e:
            self.warnings.append(f"Could not save verification report: {e}")
    
    def _display_summary(self):
        """Display verification summary."""
        print("\n" + "="*60)
        print("📋 DATA VERIFICATION SUMMARY")
        print("="*60)
        
        status_emoji = "✅" if len(self.errors) == 0 else "❌"
        status_text = "PASSED" if len(self.errors) == 0 else "FAILED"
        
        print(f"{status_emoji} Overall Status: {status_text}")
        print(f"📊 Total Records: {len(self.df) if self.df is not None else 0}")
        print(f"❌ Errors: {len(self.errors)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        if self.errors:
            print(f"\n❌ CRITICAL ERRORS:")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        if len(self.errors) == 0:
            print(f"\n🚀 Data is ready for training!")
            if 'training_sequences' in self.verification_report:
                ts = self.verification_report['training_sequences']
                print(f"   • Training sequences: {ts['train_sequences']}")
                print(f"   • Validation sequences: {ts['val_sequences']}")
        else:
            print(f"\n🛑 Please resolve errors before proceeding with training.")

def verify_powerball_data(data_path: str) -> bool:
    """
    Convenience function to verify Powerball data.
    
    Args:
        data_path (str): Path to the CSV data file
        
    Returns:
        bool: True if data passes verification, False otherwise
    """
    verifier = PowerballDataVerifier(data_path)
    return verifier.load_and_verify()

def main():
    """Main function for standalone execution."""
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Default path
        data_path = "C:/Users/RobMo/OneDrive/Documents/PowerBall/downloads/powerball2015_Aug2025.csv"
    
    success = verify_powerball_data(data_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
