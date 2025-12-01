"""
Simple AutoML Tuning Demo
------------------------
Clean, working demonstration of integrated hyperparameter tuning and experiment tracking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

from .ml_tuning_simplified import SimplifiedTuningWizard, create_tuning_config
from .experiment_tracker import get_experiment_tracker
from .storage import get_store
# Lazy import of reusable data selector UI
try:
    from .ui.data_selector import data_selector_ui
except Exception:
    data_selector_ui = None
from .automl_workflow_automation import get_automl_workflow_automator

def render_header():
    """Render the header for the AutoML demo."""
    st.header("Simple AutoML Tuning Demo")

def render(df: pd.DataFrame, service=None):
    """Main render function for the AutoML demo."""
    render_header()
    render_sidebar(df, service)
    
    st.title("AutoML Model Tuning")

def render_sidebar(df: pd.DataFrame, service=None):
    """Sidebar for AutoML options (placeholder)."""
    st.sidebar.header("AutoML Options")
    st.sidebar.write(f"Data shape: {df.shape}")

    # Offer the reusable data selector UI if available (show regardless of df emptiness)
    combined_df = None
    if data_selector_ui is not None:
        try:
            sel_df, saved = data_selector_ui()
            if sel_df is not None and not sel_df.empty:
                combined_df = sel_df
                # use combined dataset for downstream preparation
                df = combined_df
        except Exception:
            combined_df = None

    if df.empty:
        st.warning("No data available for AutoML tuning.")
        return


    # Prepare data for ML using service if available, otherwise fallback to direct preparation
    if service:
        try:
            features_df = service.feature_service.engineer_features(df)
            X = features_df.values
            feature_cols = features_df.columns.tolist()
            y = df['powerball'].values
        except Exception as e:
            st.error(f"Error using feature engineering service: {e}")
            X, y, feature_cols = prepare_ml_data(df)
    else:
        X, y, feature_cols = prepare_ml_data(df)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Quick Tuning", "📊 Dashboard", "📈 Analysis", "🚀 Deployment"])
    
    with tab1:
        render_tuning_tab(X, y, feature_cols)
    
    with tab2:
        render_dashboard_tab()
    
    with tab3:
        render_analysis_tab()
    
    with tab4:
        render_deployment_tab()

def prepare_ml_data(df_history):
    """Prepare data for machine learning."""
    try:
        # Basic data preparation
        feature_cols = ['n1', 'n2', 'n3', 'n4', 'n5']

        if df_history is None or df_history.empty:
            return np.empty((0, len(feature_cols))), np.empty((0,)), feature_cols

        if not all(col in df_history.columns for col in feature_cols):
            return np.empty((0, len(feature_cols))), np.empty((0,)), feature_cols

        X = df_history[feature_cols].to_numpy()
        y = df_history['powerball'].to_numpy() if 'powerball' in df_history.columns else df_history['n1'].to_numpy()

        return X, y, feature_cols
        
    except Exception as e:
        st.error(f"Data preparation error: {str(e)}")
    return np.empty((0,5)), np.empty((0,)), ['n1','n2','n3','n4','n5']

def render_tuning_tab(X, y, feature_cols):
    """Render the hyperparameter tuning tab."""
    st.subheader("Quick Hyperparameter Tuning")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        experiment_name = st.text_input(
            "Experiment Name", 
            value=f"Demo_{datetime.now().strftime('%H%M%S')}"
        )
        
        model_name = st.selectbox(
            "Select Model",
            ["RandomForest", "GradientBoosting", "Ridge", "Lasso"]
        )
    
    with col2:
        search_strategy = st.selectbox(
            "Search Strategy",
            ["random_search", "grid_search"]
        )
        
        n_trials = st.slider("Number of Trials", 5, 20, 10)
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", X.shape[0])
    with col2:
        st.metric("Features", X.shape[1])
    with col3:
        st.metric("Target Range", f"{y.min():.1f} - {y.max():.1f}")
    
    # Execute tuning
    if st.button("🚀 Start Tuning", type="primary", use_container_width=True):
        run_tuning_demo(experiment_name, model_name, search_strategy, n_trials, X, y)

def run_tuning_demo(experiment_name, model_name, search_strategy, n_trials, X, y):
    """Execute tuning demonstration."""
    
    # Progress indicator
    progress_placeholder = st.empty()
    
    with progress_placeholder:
        st.info("Setting up hyperparameter tuning...")
    
    try:
        # Get tracker and wizard
        tracker = get_experiment_tracker()
        wizard = SimplifiedTuningWizard(experiment_tracker=tracker)
        
        # Create config
        config = create_tuning_config(
            experiment_name=experiment_name,
            model_name=model_name,
            search_strategy=search_strategy,
            n_trials=n_trials,
            cv_folds=3
        )
        
        with progress_placeholder:
            st.info("Running optimization...")
        
        # Run tuning
        result = wizard.run_tuning_job(config, X, y)
        
        # Show success
        with progress_placeholder:
            st.success(f"Tuning completed in {result.total_time:.2f} seconds!")
        
        # Display results
        show_results(result)
        
    except Exception as e:
        with progress_placeholder:
            st.error(f"Tuning failed: {str(e)}")

def show_results(result):
    """Display tuning results."""
    st.subheader("Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trials", len(result.all_trials))
    with col2:
        st.metric("Best Score", f"{result.best_trial.metrics.get('objective_value', 0):.4f}")
    with col3:
        st.metric("Total Time", f"{result.total_time:.2f}s")
    with col4:
        success_rate = len([t for t in result.all_trials if t.cv_scores]) / len(result.all_trials) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Best configuration
    st.subheader("Best Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Hyperparameters:**")
        for param, value in result.best_trial.hyperparameters.items():
            st.write(f"- {param}: {value}")
    
    with col2:
        st.write("**Performance:**")
        st.write(f"- CV Mean: {result.best_trial.cv_mean:.4f}")
        st.write(f"- CV Std: {result.best_trial.cv_std:.4f}")
        st.write(f"- Training Time: {result.best_trial.training_time:.2f}s")
    
    # Convergence plot
    if result.convergence_history:
        st.subheader("Optimization Progress")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(result.convergence_history) + 1)),
            y=result.convergence_history,
            mode='lines+markers',
            name='Best Score'
        ))
        
        fig.update_layout(
            title="Convergence History",
            xaxis_title="Trial Number",
            yaxis_title="Best Score"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Trials table
    st.subheader("All Trials")
    
    trials_data = []
    for trial in result.all_trials:
        row = {
            "Trial": trial.trial_number,
            "Score": trial.metrics.get('objective_value', 0) if hasattr(trial, 'metrics') else 0,
            "CV Mean": trial.cv_mean if hasattr(trial, 'cv_mean') else 0,
            "Time": trial.training_time
        }
        row.update(trial.hyperparameters)
        trials_data.append(row)
    
    trials_df = pd.DataFrame(trials_data)
    trials_df = trials_df.sort_values("Score", ascending=False)
    
    st.dataframe(trials_df, use_container_width=True)

def render_dashboard_tab():
    """Render experiment dashboard."""
    st.subheader("Experiment Dashboard")
    
    tracker = get_experiment_tracker()
    
    # Get data
    all_runs = tracker.get_experiment_runs(limit=100)
    all_jobs = tracker.get_experiment_jobs(limit=50)
    
    if not all_runs:
        st.info("No experiments found. Run the Quick Tuning to create experiments.")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Experiments", len(set(run.experiment_name for run in all_runs)))
    with col2:
        st.metric("Jobs", len(all_jobs))
    with col3:
        st.metric("Trials", len(all_runs))
    with col4:
        avg_duration = np.mean([run.duration_seconds for run in all_runs if run.duration_seconds])
        st.metric("Avg Time", f"{avg_duration:.2f}s")
    
    # Recent experiments
    st.subheader("Recent Experiments")
    
    if all_jobs:
        job_data = []
        for job in all_jobs[:10]:
            job_data.append({
                "Experiment": job.experiment_name,
                "Status": job.status,
                "Start Time": job.start_time[:19] if job.start_time else "",
                "Trials": job.total_trials
            })
        
        jobs_df = pd.DataFrame(job_data)
        st.dataframe(jobs_df, use_container_width=True)

def render_deployment_tab():
    """Render the AutoML model deployment tab."""
    st.subheader("🚀 Deploy Optimized Models")
    st.write("Train a production-ready model using the best hyperparameters found during tuning.")

    automator = get_automl_workflow_automator()
    candidates = automator.get_deployment_candidates()

    if not candidates:
        st.info("No completed tuning jobs available for deployment. Run a tuning job first.")
        return

    st.write(f"Found **{len(candidates)}** completed tuning jobs ready for deployment.")

    for job in candidates:
        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**Experiment:** `{job.experiment_name}`")
                best_trial = automator.tracker.get_run_by_id(job.best_trial_id) if job.best_trial_id is not None else None
                if best_trial and 'objective_value' in best_trial.metrics:
                    st.write(f"**Best Score:** `{best_trial.metrics['objective_value']:.4f}`")
                st.caption(f"Job ID: {job.job_id}")

            with col2:
                if best_trial:
                    st.write("**Best Hyperparameters:**")
                    st.json(best_trial.hyperparameters, expanded=False)

            with col3:
                if st.button("Deploy this Model", key=job.job_id, use_container_width=True):
                    with st.spinner(f"Training model from job `{job.experiment_name}`..."):
                        result = automator.trigger_automated_training(job.job_id)

                    if result.training_successful:
                        st.success(f"✅ Model '{result.model_name}' trained successfully! New Prediction Set ID: `{result.prediction_set_id}`")
                    else:
                        st.error(f"❌ Deployment failed: {result.error_message}")

def render_analysis_tab():
    """Render analysis tab."""
    st.subheader("Results Analysis")
    
    tracker = get_experiment_tracker()
    all_runs = tracker.get_experiment_runs(limit=1000)
    
    if not all_runs:
        st.info("No experiment results available.")
        return
    
    # Experiment selection
    experiments = list(set(run.experiment_name for run in all_runs))
    
    if len(experiments) > 1:
        selected_experiment = st.selectbox("Select Experiment", experiments)
        filtered_runs = [r for r in all_runs if r.experiment_name == selected_experiment]
    else:
        selected_experiment = experiments[0] if experiments else "No experiments"
        filtered_runs = all_runs
    
    if not filtered_runs:
        st.warning("No runs found.")
        return
    
    # Analysis metrics
    objectives = [r.metrics.get('objective_value', 0) for r in filtered_runs if r.metrics.get('objective_value')]
    
    if objectives:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Runs", len(filtered_runs))
        with col2:
            st.metric("Best Score", f"{max(objectives):.4f}")
        with col3:
            st.metric("Mean Score", f"{np.mean(objectives):.4f}")
        with col4:
            st.metric("Std Score", f"{np.std(objectives):.4f}")
        
        # Distribution plot
        st.subheader("Score Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=objectives, nbinsx=15))
        fig.update_layout(
            title="Distribution of Objective Values",
            xaxis_title="Objective Value",
            yaxis_title="Count"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top runs
        st.subheader("Top Performing Runs")
        
        top_runs = sorted(
            [r for r in filtered_runs if r.metrics.get('objective_value')], 
            key=lambda r: r.metrics['objective_value'], 
            reverse=True
        )[:5]
        
        if top_runs:
            comparison_data = []
            for i, run in enumerate(top_runs):
                row = {
                    "Rank": i + 1,
                    "Run ID": run.run_id[:8],
                    "Score": run.metrics.get('objective_value', 0),
                    "Duration": f"{run.duration_seconds:.2f}s"
                }
                row.update(run.hyperparameters)
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    else:
        st.warning("No objective values found in experiment runs.")