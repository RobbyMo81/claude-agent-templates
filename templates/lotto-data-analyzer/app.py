"""
Powerball Insights
-----------------
A comprehensive data analysis and visualization tool for Powerball lottery data.
"""
import streamlit as st

# App configuration - MUST be first
st.set_page_config(
    page_title="Powerball Insights",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

import importlib
import logging
from pathlib import Path
from core.storage import get_store
from core.model_training_service import ModelTrainingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model training service on startup to load persisted models
@st.cache_resource
def initialize_model_training_service():
    """Initialize and cache the model training service with persisted models."""
    try:
        logger.info("Initializing model training service...")
        service = ModelTrainingService()
        logger.info("Model training service initialized successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize model training service: {e}")
        st.error(f"Fatal: Could not initialize model service. {e}")
        return None

# Load models on startup
model_training_service = initialize_model_training_service()

# Load data centrally
df = get_store().latest()

# Core pages to include in navigation
PAGES = {
    "Upload / Data": "ingest",
    "CSV Formatter": "csv_formatter",
    "Data Maintenance": "data_maintenance",
    "Prediction Storage Migration": "prediction_storage_migration_ui",
    "Number Frequency": "frequency",
    "Day of Week Analysis": "dow_analysis",
    "Time Trends": "time_trends", 
    "Inter-Draw Gaps": "inter_draw",
    "Combinatorial Analysis": "combos",
    "Sum Analysis": "sums",
    "ML Experimental": "ml_experimental",
    "AutoML Tuning": "automl_simple",
    "System Architecture": "system_relationship_visualizer",
}

# Include AI analysis module
PAGES["Ask the Numbers (AI)"] = "llm_query"

# App title and description
st.sidebar.title("🎯 Powerball Insights")
st.sidebar.caption("Statistical analysis and visualization tool")

# Navigation
page_name = st.sidebar.radio("Navigate", list(PAGES.keys()))
page_module = PAGES[page_name]

# Documentation expander
with st.sidebar.expander("About & Documentation"):
    st.markdown("""
    **Powerball Insights** provides statistical analysis of lottery data. 
    
    Features:
    - Upload & format lottery data
    - Visualize number frequencies
    - Analyze day-of-week patterns
    - Explore time-based trends
    - Calculate combinatorial statistics
    - Run experimental ML analyses
    
    **Note**: No prediction can improve your odds of winning. This app is for educational and entertainment purposes.
    """)

# Display version at the bottom of sidebar
st.sidebar.divider()
from core import __version__
st.sidebar.caption(f"v{__version__}")

# Import and render the selected page
try:
    page = importlib.import_module(f"core.{page_module}")
    
    # Pages that do not require the main DataFrame
    no_df_pages = ["ingest", "csv_formatter", "system_relationship_visualizer", "prediction_storage_migration_ui"]
    # Pages that require the model training service
    service_pages = ["ml_experimental", "automl_simple"]

    # If the service failed to load, show an error and stop.
    if model_training_service is None and page_module in service_pages:
        st.error("The Model Training Service could not be initialized. This page cannot be loaded.")
    else:
        render_args = {}
        if page_module not in no_df_pages:
            render_args['df'] = df
        
        if page_module in service_pages:
            render_args['service'] = model_training_service
            
        # Call render with keyword arguments, which is more robust
        page.render(**render_args)

except Exception as e:
    st.error(f"Error loading page: {e}")
    logger.error(f"Page loading error for {page_module}: {e}", exc_info=True)
