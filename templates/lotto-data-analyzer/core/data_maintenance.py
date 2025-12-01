"""
Data Maintenance Agent
---------------------
Provides tools for cleaning, maintaining, and optimizing Powerball datasets.
Features:
- Duplicate detection and removal
- Date format standardization
- Missing value handling
- Data validation (range checks)
- Dataset optimization
- Automatic backups
"""

import pandas as pd
import numpy as np
import os
import datetime
import logging
from typing import Optional, Dict, List, Tuple
import joblib
import shutil
from pathlib import Path
import streamlit as st

# Constants
DATA_DIR = Path("data")
BACKUP_DIR = DATA_DIR / "backups"
DEFAULT_CSV = DATA_DIR / "powerball_history.csv"
MAX_WHITE_BALL = 69
MIN_WHITE_BALL = 1
MAX_POWERBALL = 26
MIN_POWERBALL = 1

# Ensure backup directory exists
BACKUP_DIR.mkdir(exist_ok=True, parents=True)

class DataMaintenanceAgent:
    """
    Data Maintenance Agent for Powerball datasets.
    Provides tools for data cleaning, validation, and optimization.
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None, file_path: Optional[str] = None):
        """
        Initialize the data maintenance agent.
        
        Args:
            df: DataFrame to maintain, or None to load from file_path
            file_path: Path to CSV file to load, or None to use default
        """
        # Always initialize self.df as a DataFrame to keep return types consistent
        self.df: pd.DataFrame = pd.DataFrame()
        self.file_path = file_path or str(DEFAULT_CSV)
        self.issues = {
            'duplicates': 0,
            'missing_values': 0,
            'out_of_range': 0,
            'date_format': 0,
            'total': 0,
        }
        # If a DataFrame was provided, use it. Otherwise, try loading from file.
        if df is not None:
            self.df = df
        else:
            if os.path.exists(self.file_path):
                # load_data sets self.df and handles errors
                self.load_data()
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from the file path.
        
        Returns:
            DataFrame with loaded data
        """
        try:
            self.df = pd.read_csv(self.file_path)
            return self.df
        except Exception as e:
            logging.error(f"Error loading data from {self.file_path}: {e}")
            self.df = pd.DataFrame()
            return self.df
    
    def backup_data(self) -> str:
        """
        Create a backup of the current dataset.
        
        Returns:
            Path to the backup file
        """
        if self.df is None or self.df.empty:
            return "No data to backup"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = BACKUP_DIR / f"powerball_backup_{timestamp}.csv"
        
        try:
            self.df.to_csv(backup_file, index=False)
            return str(backup_file)
        except Exception as e:
            logging.error(f"Error creating backup: {e}")
            return f"Backup failed: {e}"
    
    def detect_duplicates(self) -> pd.DataFrame:
        """
        Detect duplicate records in the dataset.
        
        Returns:
            DataFrame containing duplicate records
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Check for exact duplicates
        duplicates = self.df[self.df.duplicated(keep='first')]
        self.issues['duplicates'] = len(duplicates)
        self.issues['total'] += len(duplicates)
        
        # Ensure we return a DataFrame, not a Series
        if isinstance(duplicates, pd.Series):
            duplicates = duplicates.to_frame()
            
        return duplicates
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate records from the dataset.
        
        Returns:
            DataFrame with duplicates removed
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Count duplicates first
        dup_count = sum(self.df.duplicated())
        self.issues['duplicates'] = dup_count
        self.issues['total'] += dup_count
        
        # Remove duplicates
        self.df = self.df.drop_duplicates(keep='first')
        
        return self.df
    
    def standardize_date_format(self) -> pd.DataFrame:
        """
        Standardize date formats in the dataset.
        
        Returns:
            DataFrame with standardized date formats
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Identify date column
        date_col = None
        for col in self.df.columns:
            col_lower = col.lower()
            if 'date' in col_lower or 'day' in col_lower or 'time' in col_lower:
                date_col = col
                break
        
        if date_col:
            try:
                # Convert to datetime and standardize format
                # Keep track of how many records were modified
                orig_vals = self.df[date_col].copy()
                self.df[date_col] = pd.to_datetime(self.df[date_col])
                modified = (orig_vals != self.df[date_col].astype(str)).sum()
                self.issues['date_format'] = modified
                self.issues['total'] += modified
            except Exception as e:
                logging.error(f"Error standardizing date format: {e}")
        
        return self.df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Returns:
            DataFrame with missing values handled
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Count missing values
        missing_count = self.df.isna().sum().sum()
        self.issues['missing_values'] = missing_count
        self.issues['total'] += missing_count
        
        # Handle missing values based on column type
        for col in self.df.columns:
            col_lower = col.lower()
            
            # Numeric columns (like ball numbers)
            if any(term in col_lower for term in ['n1', 'n2', 'n3', 'n4', 'n5', 'powerball', 'ball']):
                # For numeric lottery data, we can't truly impute missing values
                # The best approach is to drop those rows as they represent incomplete data
                # But we'll flag them so the user can decide
                missing_rows = self.df[self.df[col].isna()]
                self.df = self.df.dropna(subset=[col])
            
            # Date columns
            elif any(term in col_lower for term in ['date', 'day', 'time']):
                # Can't reliably impute missing dates
                self.df = self.df.dropna(subset=[col])
            
            # Other columns
            else:
                # For non-critical columns, fill with mode (most common value)
                if self.df[col].isna().any():
                    if self.df[col].dtype == 'object':
                        fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
                        self.df[col] = self.df[col].fillna(fill_value)
                    else:
                        fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                        self.df[col] = self.df[col].fillna(fill_value)
        
        return self.df
    
    def validate_range(self) -> Dict[str, List[int]]:
        """
        Validate that Powerball numbers are within allowed ranges.
        
        Returns:
            Dictionary with lists of row indices that are out of range
        """
        if self.df is None or self.df.empty:
            return {'white_balls': [], 'powerball': []}
        
        # Check white ball columns (n1-n5)
        white_cols = [col for col in self.df.columns if col in ['n1', 'n2', 'n3', 'n4', 'n5']]
        
        out_of_range = {
            'white_balls': [],
            'powerball': []
        }
        
        # Check white balls (1-69)
        for col in white_cols:
            invalid_idx = self.df[
                (self.df[col] < MIN_WHITE_BALL) | 
                (self.df[col] > MAX_WHITE_BALL)
            ].index.tolist()
            out_of_range['white_balls'].extend(invalid_idx)
        
        # Check Powerball (1-26)
        if 'powerball' in self.df.columns:
            invalid_idx = self.df[
                (self.df['powerball'] < MIN_POWERBALL) | 
                (self.df['powerball'] > MAX_POWERBALL)
            ].index.tolist()
            out_of_range['powerball'].extend(invalid_idx)
        
        # Remove duplicates
        out_of_range['white_balls'] = list(set(out_of_range['white_balls']))
        out_of_range['powerball'] = list(set(out_of_range['powerball']))
        
        # Update issue count
        self.issues['out_of_range'] = len(out_of_range['white_balls']) + len(out_of_range['powerball'])
        self.issues['total'] += self.issues['out_of_range']
        
        return out_of_range
    
    def fix_out_of_range(self) -> pd.DataFrame:
        """
        Fix out-of-range values in the dataset.
        Removes rows with invalid values.
        
        Returns:
            DataFrame with out-of-range values fixed
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Get out-of-range indices
        out_of_range = self.validate_range()
        
        # Combine all invalid indices
        all_invalid = list(set(out_of_range['white_balls'] + out_of_range['powerball']))
        
        # Remove invalid rows
        if all_invalid:
            self.df = self.df.drop(all_invalid)
        
        return self.df
    
    def optimize_dataset(self) -> pd.DataFrame:
        """
        Optimize the dataset for memory efficiency.
        
        Returns:
            Optimized DataFrame
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Optimize numeric columns
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Check if column contains only integers
                if (self.df[col] % 1 == 0).all():
                    # Check range to determine best integer type
                    col_min = self.df[col].min()
                    col_max = self.df[col].max()
                    
                    if col_min >= 0 and col_max <= 255:
                        self.df[col] = self.df[col].astype('uint8')
                    elif col_min >= 0 and col_max <= 65535:
                        self.df[col] = self.df[col].astype('uint16')
                    elif col_min >= -128 and col_max <= 127:
                        self.df[col] = self.df[col].astype('int8')
                    elif col_min >= -32768 and col_max <= 32767:
                        self.df[col] = self.df[col].astype('int16')
        
        return self.df
    
    def sort_by_date(self) -> pd.DataFrame:
        """
        Sort the dataset by draw date.
        
        Returns:
            DataFrame sorted by date
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Find date column
        date_col = None
        for col in self.df.columns:
            col_lower = col.lower()
            if 'date' in col_lower or 'day' in col_lower or 'time' in col_lower:
                date_col = col
                break
        
        if date_col:
            # Ensure date column is in datetime format
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            
            # Sort by date (ascending so oldest is first)
            self.df = self.df.sort_values(date_col, ascending=True)
        
        return self.df
    
    def clean_and_validate(self) -> pd.DataFrame:
        """
        Run all cleaning and validation steps.
        
        Returns:
            Cleaned and validated DataFrame
        """
        # Reset issue counts
        self.issues = {
            'duplicates': 0,
            'missing_values': 0,
            'out_of_range': 0,
            'date_format': 0,
            'total': 0
        }
        
        # Run all steps
        self.backup_data()
        self.remove_duplicates()
        self.standardize_date_format()
        self.handle_missing_values()
        self.fix_out_of_range()
        self.optimize_dataset()
        self.sort_by_date()
        
        return self.df
    
    def save_data(self, file_path: Optional[str] = None) -> str:
        """
        Save the cleaned dataset.
        
        Args:
            file_path: Path to save the cleaned dataset. If None, uses the original path.
            
        Returns:
            Path to the saved file
        """
        if self.df is None or self.df.empty:
            return "No data to save"
        
        save_path = file_path or self.file_path
        
        try:
            self.df.to_csv(save_path, index=False)
            return save_path
        except Exception as e:
            logging.error(f"Error saving cleaned data: {e}")
            return f"Save failed: {e}"
    
    def cleanup_old_backups(self, keep_count: int = 5) -> Tuple[int, int]:
        """
        Delete old backup files, keeping only the most recent ones.
        
        Args:
            keep_count: Number of most recent backups to keep
            
        Returns:
            Tuple of (deleted_count, deleted_size_bytes)
        """
        deleted_count = 0
        deleted_size = 0
        
        try:
            if BACKUP_DIR.exists():
                # Get all backup files sorted by modification time (newest first)
                # Using st_mtime instead of st_birthtime for Windows compatibility
                backups = list(BACKUP_DIR.glob("*.csv"))
                backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                # Keep only the most recent 'keep_count' backups
                for backup in backups[keep_count:]:
                    deleted_size += backup.stat().st_size
                    backup.unlink()
                    deleted_count += 1
        except Exception as e:
            logging.error(f"Error deleting old backups: {e}")
            
        return (deleted_count, deleted_size)


def render(df: pd.DataFrame):
    """Render the data maintenance Streamlit UI for the given dataframe."""
    st.title("Data Maintenance Dashboard")

    # Display dataset info
    st.subheader("Dataset Information")

    if df is None or df.empty:
        st.warning("No data available. Please upload or ingest data first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        if 'draw_date' in df.columns:
            st.metric("Date Range", f"{df['draw_date'].min()} to {df['draw_date'].max()}")
        else:
            st.metric("Date Range", "N/A - No draw_date column found")
    with col3:
        st.metric("Dataset Size", f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")

    # Create tabs for different maintenance tasks
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Quality",
        "Date Format Standardization",
        "Range Validation",
        "Backup Management",
    ])

    # Tab 1: Data Quality
    with tab1:
        st.header("Data Quality Analysis")
        st.write("Identify and fix quality issues in the dataset")

        # Check for duplicates
        if 'draw_date' in df.columns:
            duplicate_dates = df['draw_date'].duplicated().sum()
            if duplicate_dates > 0:
                st.error(f"⚠️ Found {duplicate_dates} duplicate draw dates!")

                if st.button("Remove Duplicates"):
                    # Implementation would go here
                    st.info("Duplicate removal functionality to be implemented")
            else:
                st.success("✅ No duplicate draw dates found")
        else:
            st.warning("Draw date column not found - cannot check for duplicates")

        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            st.error(f"⚠️ Found {missing_count} missing values in the dataset!")
            st.write("Missing values by column:")
            st.write(df.isnull().sum())

            if st.button("Handle Missing Values"):
                # Implementation would go here
                st.info("Missing value handling to be implemented")
        else:
            st.success("✅ No missing values found")

    # Tab 2: Date Format Standardization
    with tab2:
        st.header("Date Format Standardization")
        st.write("Ensure consistent date formats across the dataset")

        # Date format analysis would go here
        st.info("Date format standardization to be implemented")

    # Tab 3: Range Validation
    with tab3:
        st.header("Data Range Validation")
        st.write("Check that all values are within expected ranges")

        # Check white ball ranges
        invalid_white = 0
        for col in ['n1', 'n2', 'n3', 'n4', 'n5']:
            if col in df.columns:
                invalid_in_col = ((df[col] < MIN_WHITE_BALL) | (df[col] > MAX_WHITE_BALL)).sum()
                invalid_white += invalid_in_col

        # Check powerball ranges
        invalid_pb = 0
        if 'powerball' in df.columns:
            invalid_pb = ((df['powerball'] < MIN_POWERBALL) | (df['powerball'] > MAX_POWERBALL)).sum()

        if invalid_white > 0 or invalid_pb > 0:
            st.error(f"⚠️ Found {invalid_white} invalid white ball values and {invalid_pb} invalid powerball values!")

            if st.button("Fix Range Issues"):
                # Implementation would go here
                st.info("Range fixing functionality to be implemented")
        else:
            st.success("✅ All values are within expected ranges")

    # Tab 4: Backup Management
    with tab4:
        st.header("Backup Management")
        st.write("Create and manage dataset backups")

        # Display existing backups
        try:
            if BACKUP_DIR.exists():
                backups = list(BACKUP_DIR.glob("*.csv"))
                backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                if backups:
                    st.write(f"Found {len(backups)} backup files:")
                    for backup in backups[:5]:  # Show most recent 5
                        modified_time = datetime.datetime.fromtimestamp(backup.stat().st_mtime)
                        file_size = backup.stat().st_size / 1024  # KB
                        st.write(f"- {backup.name} ({modified_time.strftime('%Y-%m-%d %H:%M:%S')}, {file_size:.1f} KB)")

                    if len(backups) > 5:
                        st.write(f"... and {len(backups) - 5} more")
                else:
                    st.info("No backups found")
            else:
                st.info("Backup directory does not exist yet")
        except Exception as e:
            st.error(f"Error listing backups: {e}")

        # Backup creation controls
        if st.button("Create New Backup"):
            agent = DataMaintenanceAgent(df)
            backup_path = agent.backup_data()
            st.success(f"Backup created at: {backup_path}")

        # Backup cleanup controls
        keep_count = st.slider("Keep how many recent backups?", 1, 20, 5)
        if st.button("Clean Up Old Backups"):
            agent = DataMaintenanceAgent(df)
            deleted_count, deleted_size = agent.cleanup_old_backups(keep_count)
            if deleted_count > 0:
                st.success(f"Removed {deleted_count} old backups (freed {deleted_size / (1024*1024):.2f} MB)")
            else:
                st.info("No backups were deleted")
