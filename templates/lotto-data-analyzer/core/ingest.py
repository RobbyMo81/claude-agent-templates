# core/ingest.py
import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from pathlib import Path
from datetime import datetime
from core.storage import get_store, DATASET_PATHS

# Priority order for datasets (newest/most complete first)
MAX_MB = 8
DATA_PATH = DATASET_PATHS[0]

from typing import Union

def parse_flexible_date(date_string: str) -> pd.Timestamp:
    """
    Tries to parse a date string using a list of known formats.
    Returns a pandas Timestamp on success, or pd.NaT on failure.
    """
    if not isinstance(date_string, str) or not date_string.strip():
        return pd.NaT

    # Comprehensive list of formats to try, ordered by likelihood
    formats_to_try = [
        '%Y-%m-%d',        # 2025-06-22 (Standard)
        '%m/%d/%Y',        # 05/04/2016 (Handles MM/DD/YYYY and M/D/YYYY)
        '%m-%d-%Y',        # 05-04-2016 (Handles MM-DD-YYYY)
        '%m/%d/%y',        # 5/4/16
        '%A, %B %d, %Y',   # Wednesday, May 04, 2016
        '%B %d, %Y',       # May 04, 2016 (with comma)
        '%B %d %Y',        # May 4 2016
    ]

    for fmt in formats_to_try:
        try:
            # Using pd.to_datetime is a robust choice as it returns a Timestamp
            return pd.to_datetime(date_string, format=fmt)
        except (ValueError, TypeError):
            # If this format fails, continue to the next one
            continue
    
    # If the loop finishes without returning, no format matched.
    return pd.NaT

# ----------------------------------------------------------------------
def render() -> None:
    st.header("📤 Upload / Data")
    st.caption("Drag in a Powerball CSV, use the default file, or manually add new draw results.")

    # Create tabs for different data input methods
    tab1, tab2 = st.tabs(["File Upload", "Manual Entry"])
    
    with tab1:
        st.subheader("📁 File Upload")
        # -------- 1. File uploader (always visible) -----------------------
        file = st.file_uploader("Upload Powerball CSV",
                                type="csv",
                                help=f"Limit {MAX_MB} MB per file • CSV only",
                                accept_multiple_files=False)

        # -------- 2. Decide which dataframe to show ----------------------
        df: pd.DataFrame | None = None

        if file:                              # user just uploaded something
            try:
                df = pd.read_csv(file)
                
                # Standardize date format to YYYY-MM-DD
                if 'draw_date' in df.columns:
                    try:
                        original_count = len(df)

                        # Apply the new flexible date parsing function
                        df['draw_date'] = df['draw_date'].apply(parse_flexible_date)
                        
                        # Remove any rows where date conversion failed
                        df = df.dropna(subset=['draw_date'])
                        converted_count = len(df)
                        
                        if converted_count < original_count:
                            st.warning(f"Removed {original_count - converted_count} rows with invalid dates")
                        
                        # Convert to standardized string format AFTER parsing and cleaning
                        df['draw_date'] = df['draw_date'].dt.strftime('%Y-%m-%d')
                        st.success(f"Loaded {converted_count:,} rows from upload (dates standardized to YYYY-MM-DD format)")
                    except Exception as date_error:
                        st.warning(f"Date format standardization failed: {date_error}")
                        st.success(f"Loaded {len(df):,} rows from upload")
                else:
                    st.success(f"Loaded {len(df):,} rows from upload")
                    
            except Exception as e:
                st.error(f"❌ Cannot read that CSV: {e}")
                return

        else:                                 # fall back to default on disk
            df = get_store().latest()
            if df is not None:
                st.success(f"Loaded {len(df):,} rows from the default dataset.")
            else:
                st.info("No valid default dataset yet — upload one to begin.")
                return
        
        # -------- 3. Preview + persist (File Upload Tab) -----------------------------------
        st.dataframe(df.head(), use_container_width=True)

        # save upload as new default
        if file and st.button("💾  Save as default dataset"):
            get_store().set_latest(df)
            get_store().reload_data()
            st.toast("Saved as new default dataset! The app is now using the new data.", icon="✅")
            st.rerun()

        # offer to delete the bad file
        if (not file) and st.button("🗑️ Delete invalid default file"): # DATA_PATH
            DATA_PATH.unlink(missing_ok=True)
            st.toast("Deleted. Upload a new dataset to continue.", icon="🗑️")
            st.rerun()           # refresh the page

    with tab2:
        st.subheader("✏️ Manual Entry")
        st.info("Add individual Powerball draw results with authentic data from official sources.")
        
        # Load current dataset to add to
        current_df = get_store().latest()
        if current_df is None:
            st.warning("No existing dataset found. Please upload a base dataset first.")
            return
            
        # Manual entry form
        with st.form("manual_entry_form"):
            st.markdown("**Enter New Draw Result**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Date input
                draw_date = st.date_input("Draw Date", help="Enter the official draw date")
                
                # White balls input
                st.markdown("**White Balls (5 numbers from 1-69):**")
                white_cols = st.columns(5)
                white_balls = []
                for i, col in enumerate(white_cols):
                    with col:
                        num = st.number_input(f"Ball {i+1}", min_value=1, max_value=69, 
                                            value=1, key=f"white_{i}")
                        white_balls.append(num)
                
                # Powerball input
                powerball = st.number_input("Powerball (1-26)", min_value=1, max_value=26, value=1)
            
            with col2:
                st.markdown("**Validation:**")
                # Check for duplicates in white balls
                white_set = set(white_balls)
                if len(white_set) != 5:
                    st.error("❌ White balls must be unique")
                    valid_entry = False
                else:
                    st.success("✅ White balls are unique")
                    valid_entry = True
                
                # Check if date already exists
                if current_df is not None and 'draw_date' in current_df.columns:
                    date_str = str(draw_date)
                    if date_str in current_df['draw_date'].astype(str).values:
                        st.warning("⚠️ Date already exists")
                        valid_entry = False
                    else:
                        st.success("✅ New date")
            
            submitted = st.form_submit_button("Add Draw Result", disabled=not valid_entry)
            
            if submitted and valid_entry:
                # Create new row
                new_row = {
                    'draw_date': str(draw_date),
                    'n1': white_balls[0],
                    'n2': white_balls[1], 
                    'n3': white_balls[2],
                    'n4': white_balls[3],
                    'n5': white_balls[4],
                    'powerball': powerball
                }
                
                # Add to dataframe
                new_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
                
                # Sort by date (newest first)
                new_df['draw_date'] = pd.to_datetime(new_df['draw_date'])
                new_df = new_df.sort_values('draw_date', ascending=False)
                new_df['draw_date'] = new_df['draw_date'].dt.strftime('%Y-%m-%d')
                
                # Save to file
                # Update storage
                get_store().set_latest(new_df)
                get_store().reload_data()
                
                st.success(f"✅ Added draw result for {draw_date}")
                st.success(f"**Numbers:** {', '.join(map(str, sorted(white_balls)))} | **Powerball:** {powerball}")
                st.toast("Draw result added successfully!", icon="🎯")
                st.rerun()
        
        # Show recent entries
        if current_df is not None and not current_df.empty:
            st.markdown("---")
            st.markdown("**Recent Draw Results:**")
            # Show last 10 entries
            recent_df = current_df.head(10)
            st.dataframe(recent_df, use_container_width=True)
