Title: Architectural Review of `core/ml_experimental.py`
Report Date: June 21, 2025
Analysis Scope: Functional and architectural analysis of the `core/ml_experimental.py` module.
System Status: Analysis Complete - Recommendation Provided.
Analyst: Gemini Code Assist, AI DevOps Engineer

---

## 1. Executive Summary

This report provides a comprehensive architectural review of the `core/ml_experimental.py` module. The analysis reveals that this module, while offering a wide range of user-facing features, operates as a parallel, inconsistent, and partially deprecated system within our newly unified architecture.

It contains significant code duplication, implements its own non-standard feature engineering and model training pipelines, and maintains direct dependencies on the legacy `joblib`-based `PredictionSystem`. These factors introduce significant risks to data integrity, system stability, and long-term maintainability.

**The final recommendation is to completely remove the `core/ml_experimental.py` module.** Its valuable UI components should be integrated into a new, clean interface that exclusively uses the unified `ModelTrainingService` and `FeatureEngineeringService`.

---

## 2. Core Functionality Analysis

The `ml_experimental.py` module provides a user-facing UI page titled "🧪 Experimental ML Analysis" with the following features organized into tabs:

-   **Model Configuration (Sidebar):** Allows users to select a model (`Random Forest`, `Gradient Boosting`, `Ridge Regression`) and configure basic hyperparameters in real-time.
-   **New Draw Results (Sidebar):** A form for manually entering new draw results. This is a duplicate of the primary functionality in the `ingest.py` module.
-   **Train & Predict Tab:** The main action tab, which triggers a custom model training and prediction workflow. It displays the generated predictions alongside the last 5 actual draws and provides some basic match statistics.
-   **Model Evaluation Tab:**
    -   *Model Performance:* Runs a custom cross-validation loop and displays the Mean Absolute Error (MAE) for the selected model.
    -   *Training History:* Displays training session information from a separate, module-specific persistence layer (`ml_memory`).
    -   *Prediction History:* Displays historical predictions (this section has been a source of `datetime` parsing bugs).
-   **Feature Importance Tab:**
    -   *Basic:* Calculates and displays feature importance from a custom-trained model.
    -   *Enhanced:* Calls the legacy `PredictionSystem` to get feature importance.
-   **Prediction Management & History Tabs:** These tabs act as containers that render UI components from `core/ml_prediction_interface.py`.

---

## 3. Architectural Integration and Dependencies

The module is deeply integrated with numerous, and often conflicting, parts of the `core` system.

-   **`core.storage`**: Correctly used to fetch the primary historical dataset.
-   **`core.modernized_prediction_system`**: Used to fetch "enhanced" predictions. This is a compatibility layer that should ideally be replaced by direct service calls.
-   **`core.ml_memory`**: **(Red Flag)** Imports a separate, parallel persistence system for tracking training sessions, which is entirely redundant with the main `PersistentModelPredictionManager`.
-   **`core.persistent_model_predictions`**: Correctly used to get the manager for the main SQLite database.
-   **`core.ml_prediction_interface`**: Correctly used to render shared UI components for managing and viewing predictions.
-   **`core.datetime_manager`**: Correctly used for standardized date/time parsing.
-   **`core.prediction_accuracy_evaluator`**: Correctly used to evaluate accuracy when new draw results are submitted.
-   **`core.prediction_system`**: **(CRITICAL FLAW)** Directly imports and instantiates the legacy, `joblib`-based `PredictionSystem`.

---

## 4. Redundancy Analysis

This module duplicates a significant amount of functionality that has been centralized in our unified services.

### 4.1. Feature Engineering Redundancy

The module implements its own feature engineering in the `_prep` function, which is completely redundant with the `FeatureEngineeringService`.

**`ml_experimental.py` Implementation:**
```python
def _prep(df: pd.DataFrame):
    # ...
    # Identifies top co-occurring white-ball pairs
    # ...
    # Create binary features for each top pair
    feature_cols = ["ordinal"]
    for a, b in top_pairs:
        col = f"pair_{a}_{b}"
        # ...
        df[col] = (contains_a & contains_b).astype(int)
        feature_cols.append(col)
    # ...
```
**Analysis:** This is a basic, standalone feature engineering pipeline. The `FeatureEngineeringService` provides a much more comprehensive and standardized set of over 100 features and is the single source of truth for this functionality.

### 4.2. Model Training Redundancy

The module's "Train & predict" button triggers a completely separate and non-standard training workflow.

**`ml_experimental.py` Implementation:**
```python
if st.button("Train & predict next draw"):
    # ...
    # Starts its own ML memory session via get_ml_tracker()
    # ...
    # Performs its own cross-validation using KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, ...)
    # ...
    # Trains on the full dataset
    model.fit(X, y)
    # ...
    # Persists the model to a separate directory using joblib
    joblib.dump(model, model_path)
    # ...
    # Stores predictions in the main SQLite DB, but with its own logic
    pm.store_model_predictions(...)
```
**Analysis:** This entire workflow is redundant. The `ModelTrainingService` is designed to handle all of these steps (training, cross-validation, persistence, prediction storage) in a standardized, robust, and centralized manner. This module's implementation bypasses the unified service entirely.

### 4.3. Data Ingestion Redundancy

The "New Draw Results" form in the sidebar duplicates the functionality of the "Manual Entry" tab in `core/ingest.py`. While both ultimately write to the same CSV file, the implementation in `ml_experimental.py` is far more complex and attempts to update multiple systems at once, creating a high risk of inconsistency.

---

## 5. Legacy Dependency Analysis

The most critical architectural flaw is the module's direct dependency on the deprecated, `joblib`-based `PredictionSystem`.

**Code Evidence:**
```python
# Near the top of the file, this import should not exist
from .prediction_system import PredictionSystem

# In the "Train & Predict" tab, a section explicitly uses the legacy system
try:
    from .prediction_system import PredictionSystem
    prediction_system = PredictionSystem(df_history)
    
    # Displays data directly from the legacy system's joblib file
    last_predictions = prediction_system.prediction_history['predictions'][-5:]
    last_accuracy = prediction_system.prediction_history['accuracy'][-5:]
    # ...
except Exception as e:
    st.error(f"Error loading comparison data: {str(e)}")
```
**Analysis:** This direct usage of the legacy system completely undermines the project's goal of a unified SQLite-based architecture. It re-introduces the risk of file-based data corruption and creates a confusing user experience where data from two different storage systems (legacy `joblib` and modern `SQLite`) is displayed side-by-side.

---

## 6. Risk Assessment

-   **Stability Risk (High):** The module's complex, non-standard logic and mixed dependencies make it brittle and a frequent source of bugs. Its large size and duplicated code make it difficult to debug and maintain.
-   **Data Integrity Risk (High):** By implementing its own data ingestion and persistence logic (`ml_memory`), it creates parallel data stores that can easily become out of sync with the primary database, leading to inconsistent system behavior. The use of the legacy `PredictionSystem` actively pulls potentially stale or conflicting data into the UI.
-   **Maintainability Risk (Very High):** This module represents a significant source of technical debt. Any future changes to core services (`ModelTrainingService`, `FeatureEngineeringService`) would require parallel, manual updates here. Its existence forces developers to understand and maintain two separate, competing architectures.

---

## 7. Final Recommendation

The `core/ml_experimental.py` module provides a feature-rich UI, but its underlying implementation is architecturally unsound, highly redundant, and reliant on deprecated components. The cost of refactoring this module to properly use the unified services would be equivalent to a complete rewrite. The value it provides is overshadowed by the stability and maintainability risks it introduces.

Therefore, the final recommendation is:

**Option B: Remove**

The `core/ml_experimental.py` file should be **completely removed** from the project.

**Justification:**
1.  **Extreme Redundancy:** Its core functionalities (feature engineering, model training, data ingestion) are duplicates of the robust, centralized services we have built.
2.  **Legacy Dependencies:** Its direct use of the `joblib`-based `PredictionSystem` is a critical architectural violation that cannot be justified.
3.  **High Maintenance Cost:** The effort required to maintain this parallel system and keep it in sync with the primary architecture is unsustainable.
4.  **Low Refactoring ROI:** The valuable parts of the module are the UI components it calls from `core/ml_prediction_interface.py`. These can be easily reused in a new, clean, and much simpler UI page that correctly interacts with the `ModelTrainingService`.

**Proposed Next Step:** Create a new, streamlined UI module (e.g., `core/model_workbench.py`) that provides a simple interface for triggering the `ModelTrainingService` and then calls the existing `render_prediction_display_interface` and `render_prediction_history_interface` to display the results. This approach will provide the same user value without the associated technical debt and risk.