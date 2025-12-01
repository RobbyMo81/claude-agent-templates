import streamlit as st
import pandas as pd
from typing import Optional, Tuple

"""
Reusable data selector UI for selecting/combining CSV sources into a canonical DataFrame.
Provides:
- central dataset inclusion
- multiple file upload
- selection of CSVs in data/ directory
- "Build combined" button
- optional "Save as default" action

Returns (combined_df, save_as_default_flag)
"""


def data_selector_ui() -> Tuple[pd.DataFrame, bool]:
    save_as_default = False
    combined_df = pd.DataFrame()

    st.sidebar.subheader("Training Data Sources")
    st.sidebar.caption("Select CSV sources to build the training dataset used by ML services.")

    use_central = st.sidebar.checkbox("Use central dataset (default)", value=True)
    uploaded_files = st.sidebar.file_uploader("Upload additional CSVs (optional)", accept_multiple_files=True, type="csv")

    # Discover CSVs in data/ lazily
    data_dir_files = []
    try:
        from ..storage import DATA_PATH
        data_dir_files = [p.name for p in DATA_PATH.glob("*.csv")]
    except Exception:
        data_dir_files = []

    selected_disk_files = []
    if data_dir_files:
        selected_disk_files = st.sidebar.multiselect("Select CSVs from data/ to include", options=data_dir_files, default=[])

    # Build combined dataset
    if st.sidebar.button("Build combined training dataset"):
        parts = []
        from ..storage import load_and_normalize_csv, get_store, DATA_PATH as _DATA_PATH

        if use_central:
            try:
                central = get_store().latest()
            except Exception:
                central = pd.DataFrame()
            if not central.empty:
                parts.append(central.copy())

        # Uploaded
        if uploaded_files:
            for uf in uploaded_files:
                df_part = load_and_normalize_csv(uf)
                if not df_part.empty:
                    parts.append(df_part)
                else:
                    st.warning(f"Uploaded file {getattr(uf,'name','uploaded')} skipped (could not normalize)")

        # Disk files
        for fname in selected_disk_files:
            p = _DATA_PATH / fname
            df_part = load_and_normalize_csv(p)
            if not df_part.empty:
                parts.append(df_part)
            else:
                st.warning(f"File {fname} skipped (could not normalize)")

        if not parts:
            st.error("No valid data sources selected or files could not be normalized")
            st.stop()

        combined_df = pd.concat(parts, ignore_index=True)
        if 'draw_date' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['draw_date'], keep='first')

        st.success(f"Built combined dataset with {len(combined_df)} rows from {len(parts)} sources")
        st.dataframe(combined_df.head(), use_container_width=True)

        if st.sidebar.button("Save combined as default dataset"):
            try:
                # Persist to default dataset path
                get_store().set_latest(combined_df)
                get_store().reload_data()
                save_as_default = True
                st.toast("Saved combined dataset as default.")
            except Exception as e:
                st.error(f"Failed to save combined dataset: {e}")

    return combined_df, save_as_default
