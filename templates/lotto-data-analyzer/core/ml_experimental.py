# core/ml_experimental.py  ────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
import time
import datetime
from itertools import combinations
from .model_training_service import ModelTrainingService
from .persistent_model_predictions import get_prediction_manager
from .ui.data_selector import data_selector_ui

# Constants
WHITE_MIN, WHITE_MAX = 1, 69
PB_MIN, PB_MAX = 1, 26
DISCLAIMER = "This is an experimental feature. Predictions are not guaranteed."

# Supported models
MODELS = {
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "Ridge Regression": Ridge,
}


def _sanitize_numbers(nums):
    """Ensure 5 unique white balls w/in 1-69 and one PB 1-26; return (white, pb)."""
    try:
        *white_floats, pb_float = nums
        white = [int(round(x)) for x in white_floats]
        white = [min(max(WHITE_MIN, w), WHITE_MAX) for w in white]
        pb = int(round(pb_float))
        pb = min(max(PB_MIN, pb), PB_MAX)
        
        # Deduplicate keeping original order as much as possible
        seen = set()
        unique_white = []
        for w in white:
            while w in seen:
                w = (w % WHITE_MAX) + 1  # Wrap around more elegantly
            seen.add(w)
            unique_white.append(w)
        
        # Ensure exactly 5 white balls
        while len(unique_white) < 5:
            for i in range(WHITE_MIN, WHITE_MAX + 1):
                if i not in seen:
                    unique_white.append(i)
                    seen.add(i)
                    break
                    
        unique_white.sort()  # Convention is to present sorted white balls
        return unique_white, pb
        
    except Exception as e:
        st.error(f"Error sanitizing numbers: {str(e)}")
        return [1, 2, 3, 4, 5], 1  # Fallback to default values


def render(df: pd.DataFrame, service: ModelTrainingService):
    st.header("🧪 Experimental ML Analysis")
    st.info(DISCLAIMER)

    if df.empty:
        st.warning("No draw history available. Please ingest or upload data first.")
        return

    # --- Train & Predict + Post-Analysis ---
    # Model selection and hyperparameters in sidebar
    st.sidebar.subheader("Model Configuration")
    model_name = st.sidebar.selectbox("Select model", list(MODELS.keys()))
    
    # Dynamic hyperparameters based on selected model
    hyperparams = {}
    if model_name == "Random Forest":
        hyperparams["n_estimators"] = st.sidebar.slider("Trees (n_estimators)", 100, 1000, 400, step=100)
        hyperparams["max_depth"] = st.sidebar.slider("Max depth", 5, 50, 20)
    elif model_name == "Gradient Boosting":
        hyperparams["n_estimators"] = st.sidebar.slider("Boosting stages", 50, 500, 100, step=50)
        hyperparams["learning_rate"] = st.sidebar.slider("Learning rate", 0.01, 0.2, 0.1, step=0.01)
    elif model_name == "Ridge Regression":
        hyperparams["alpha"] = st.sidebar.slider("Regularization strength (alpha)", 0.1, 10.0, 1.0, step=0.1)
    
    # New Draw Results section for updating prediction history
    st.sidebar.divider()
    st.sidebar.subheader("New Draw Results")

    # Data selector UI (reusable)
    combined_df, saved = data_selector_ui()
    if combined_df is not None and not combined_df.empty:
        df = combined_df
    
    # Create expander to keep sidebar clean
    with st.sidebar.expander("Enter actual draw results", expanded=False):
        st.caption("Enter recent draw results to update prediction history")
        
        # White balls input (5 numbers)
        white_cols = st.columns(5)
        white_balls = []
        for i, col in enumerate(white_cols):
            with col:
                num = st.number_input(f"White #{i+1}", min_value=1, max_value=69, 
                                     value=1, step=1, key=f"actual_white_{i}")
                white_balls.append(num)
        
        # Powerball input
        pb = st.number_input("Powerball", min_value=1, max_value=26, value=1, step=1)
        
        # Draw date input
        draw_date = st.date_input("Draw Date", value=datetime.datetime.now().date())
        
        # Update button
        if st.button("Save Results & Update History"):
            try:
                # Format the draw result
                new_draw = {
                    'white_numbers': sorted(white_balls),
                    'powerball': pb,
                    'draw_date': draw_date.strftime('%Y-%m-%d')
                }
                
                # UNIFIED DATA UPDATE: Update both main dataset AND prediction system
                from .storage import get_store
                
                # Step 1: Update main dataset (same as manual entry in ingest.py)
                current_df = get_store().latest()
                if current_df.empty:
                    # Load from CSV if store is empty (use storage helper)
                    try:
                        from .storage import load_and_normalize_csv, DATA_PATH as _DATA_PATH
                        current_df = load_and_normalize_csv(_DATA_PATH)
                    except Exception:
                        current_df = pd.DataFrame(columns=['draw_date', 'n1', 'n2', 'n3', 'n4', 'n5', 'powerball'])
                
                # Check if date already exists
                date_str = str(draw_date)
                date_exists = False
                if not current_df.empty and 'draw_date' in current_df.columns:
                    if date_str in current_df['draw_date'].astype(str).values:
                        date_exists = True
                
                dataset_updated = False
                prediction_updated = False
                
                if not date_exists:
                    # Add new row to main dataset
                    new_row = {
                        'draw_date': date_str,
                        'n1': white_balls[0],
                        'n2': white_balls[1], 
                        'n3': white_balls[2],
                        'n4': white_balls[3],
                        'n5': white_balls[4],
                        'powerball': pb
                    }
                    
                    # Add to dataframe
                    new_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # Sort by date (newest first)
                    new_df['draw_date'] = pd.to_datetime(new_df['draw_date'])
                    new_df = new_df.sort_values('draw_date', ascending=False)
                    new_df['draw_date'] = new_df['draw_date'].dt.strftime('%Y-%m-%d')
                    
                    # Save to CSV and update storage
                    from .ingest import DATA_PATH
                    DATA_PATH.parent.mkdir(exist_ok=True, parents=True)
                    new_df.to_csv(DATA_PATH, index=False)
                    get_store().set_latest(new_df)
                    
                    # Update df_history for immediate use
                    df = new_df.copy()
                    dataset_updated = True
                
                # Step 2: Evaluate accuracy of previous predictions
                from .prediction_accuracy_evaluator import get_accuracy_evaluator
                try:
                    accuracy_evaluator = get_accuracy_evaluator()
                    accuracy_results = accuracy_evaluator.evaluate_predictions_against_draw(
                        draw_date=draw_date.strftime('%Y-%m-%d'),
                        white_numbers=sorted(white_balls),
                        powerball=pb
                    )
                    if accuracy_results:
                        prediction_updated = True
                        st.info(f"Accuracy evaluation completed for {len(accuracy_results)} models.")
                except Exception as e:
                    st.error(f"Failed to evaluate prediction accuracy: {e}")
                
                # Clear any cached data to ensure UI refreshes
                if 'df_history' in st.session_state:
                    del st.session_state['df_history']
                
                # Display comprehensive success/failure message
                if dataset_updated and prediction_updated:
                    st.success(f"""
                    ✅ **Complete Data Update Successful!**
                    
                    **Main Dataset Updated:**
                    - Added new draw result to CSV file
                    - Updated storage system
                    - All UI displays will now show the new data
                    
                    **Prediction System Updated:**
                    - ML models updated with new draw result
                    - Prediction accuracy tracking updated
                    - Model performance metrics recalculated
                    
                    **Draw Details:**
                    - White Balls: {sorted(white_balls)}
                    - Powerball: {pb}
                    - Date: {draw_date.strftime('%Y-%m-%d')}
                    
                    Navigate to other tabs to see the updated data reflected throughout the application.
                    """)
                elif dataset_updated:
                    st.success(f"""
                    ✅ **Main Dataset Updated Successfully**
                    
                    The new draw result has been added to the main dataset. 
                    All UI displays including "Last 5 Actual Draws" will now show the updated data.
                    
                    ⚠️ Prediction system update incomplete - some ML features may not reflect the new data until next model training.
                    """)
                elif prediction_updated:
                    st.success(f"""
                    ✅ **Prediction System Updated**
                    
                    The ML prediction system has been updated with the new draw result.
                    
                    ⚠️ Main dataset not updated - the "Last 5 Actual Draws" display may not show this new result.
                    """)
                elif date_exists:
                    st.warning(f"""
                    ⚠️ **Draw Date Already Exists**
                    
                    A draw result for {draw_date.strftime('%Y-%m-%d')} already exists in the dataset.
                    Please choose a different date or use the Upload/Data tab to modify existing entries.
                    """)
                else:
                    st.error(f"""
                    ❌ **Update Failed**
                    
                    Neither the main dataset nor the prediction system could be updated.
                    This may indicate a data storage issue or permission problem.
                    
                    Try using the Upload/Data → Manual Entry tab as an alternative.
                    """)
            except Exception as e:
                st.error(f"Error updating prediction history: {str(e)}")

    # Tab view for different aspects of ML analysis
    ml_tabs = st.tabs([
        "Train & Predict", 
        "Model Evaluation", 
        "Feature Importance", 
        "Prediction Management", 
        "Prediction History"
    ])
    
    # Data preparation is now handled by the ModelTrainingService.
    # The _prep function and its call have been removed.
    
    # Tab 1: Train & Predict
    with ml_tabs[0]:
        if st.button("Train & predict next draw"):
            with st.spinner("Training model and generating predictions..."):
                try:
                    # Use the unified service to train and predict
                    results = service.train_and_predict(
                        df=df,
                        model_name=model_name,
                        hyperparameters=hyperparams
                    )
                    
                    if results and results.get('success'):
                        st.success(f"✅ {model_name} training completed and predictions stored!")
                        st.info(f"Prediction Set ID: {results.get('set_id')}")
                        st.info(f"Training and prediction completed in {results.get('training_duration', 0):.2f} seconds")
                        
                        # Display predictions
                        st.subheader("Generated Predictions")
                        if 'predictions' in results and results['predictions']:
                            for pred in results['predictions']:
                                white_nums = pred.get('white_numbers', [])
                                pb = pred.get('powerball', 0)
                                
                                balls_html = ""
                                for num in white_nums:
                                    balls_html += f'<span style="display:inline-block; background-color:white; color:black; border:2px solid blue; border-radius:50%; width:30px; height:30px; text-align:center; line-height:30px; margin-right:5px;">{num}</span>'
                                balls_html += f'<span style="display:inline-block; background-color:red; color:white; border-radius:50%; width:30px; height:30px; text-align:center; line-height:30px;">{pb}</span>'
                                st.markdown(balls_html, unsafe_allow_html=True)
                        else:
                            st.info("No predictions were generated.")
                            
                        # Display performance metrics
                        st.subheader("Performance Metrics")
                        perf_metrics = results.get('training_results', {}).get('performance_metrics', {})
                        if perf_metrics:
                            st.json(perf_metrics)
                        else:
                            st.info("No performance metrics available.")
                            
                    else:
                        st.error(f"Training failed: {results.get('error', 'Unknown error')}")

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    
    # Tab 2: Model Evaluation
    with ml_tabs[1]:
        st.subheader("Model Performance")
        st.write("View performance metrics from the last training session for each model.")
        
        # Get performance for the selected model
        performance = service.get_model_performance(model_name)
        
        if performance:
            st.metric("White Ball MAE", f"{performance.get('white_mae', 0):.3f}")
            st.metric("Powerball MAE", f"{performance.get('powerball_mae', 0):.3f}")
            st.json(performance)
        else:
            st.info(f"No performance data available for {model_name}. Train the model to see its performance.")

    # Tab 3: Feature Importance
    with ml_tabs[2]:
        st.subheader("Feature Importance")
        st.write("This feature is under development and will be available in a future version.")

    # Tab 4: Prediction Management
    with ml_tabs[3]:
        st.subheader("Prediction Management")
        st.write("This feature is under development and will be available in a future version.")

    # Tab 5: Prediction History
    with ml_tabs[4]:
        st.subheader(f"Prediction History for {model_name}")
        
        try:
            pm = get_prediction_manager()
            history = pm.get_prediction_history(model_name, limit=10)
            
            if not history:
                st.info("No prediction history found for this model.")
            else:
                for item in history:
                    with st.expander(f"Set ID: {item['set_id']} ({item['created_at']})"):
                        st.json(item)
        except Exception as e:
            st.error(f"Error loading prediction history: {e}")
