"""
Unified Model Training Service
=============================
Consolidated training pipeline for all ML models across the system.
Eliminates training logic duplication and provides standardized cross-validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
import joblib
import os
from .feature_engineering_service import FeatureEngineeringService
from .persistent_model_predictions import PersistentModelPredictionManager, get_current_model_paths

logger = logging.getLogger(__name__)

# Column name constants (canonical names used across the project)
WHITE_BALL_COLS = ["n1", "n2", "n3", "n4", "n5"]
POWERBALL_COL = "powerball"

class ModelTrainingService:
    """
    Unified service for ML model training across all prediction systems.
    Provides standardized training, validation, and storage operations.
    """
    
    def __init__(self):
        """Initialize the model training service."""
        self.feature_service = FeatureEngineeringService()
        self.storage_manager = PersistentModelPredictionManager()
        self.models = {}
        self.model_performance = {}
        self.models_directory = "models"
        self.supported_models = {
            'Ridge Regression': Ridge,
            'Random Forest': RandomForestRegressor,
            'Gradient Boosting': GradientBoostingRegressor
        }
        self._ensure_models_directory()
        self._load_persisted_models()
    
    def _ensure_models_directory(self):
        """Ensure the models directory exists."""
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)
            logger.info(f"Created models directory: {self.models_directory}")
    
    def _generate_model_filename(self, model_name: str, timestamp: Optional[str] = None) -> str:
        """Generate a unique filename for a model artifact."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.lower().replace(' ', '_').replace('/', '_')
        return f"{safe_name}_{timestamp}.joblib"
    
    def _save_model_to_disk(self, model_name: str, pipeline) -> str:
        """
        Save a trained model pipeline to disk using joblib.
        
        Args:
            model_name: Name of the model
            pipeline: Trained scikit-learn pipeline
            
        Returns:
            Path to the saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self._generate_model_filename(model_name, timestamp)
        filepath = os.path.join(self.models_directory, filename)
        
        try:
            joblib.dump(pipeline, filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            raise
    
    def _load_persisted_models(self):
        """Load all persisted models from disk on startup."""
        try:
            model_paths = get_current_model_paths()
            loaded_count = 0
            
            for model_name, model_path in model_paths.items():
                if os.path.exists(model_path):
                    try:
                        combined_pipeline = joblib.load(model_path)
                        if isinstance(combined_pipeline, dict):
                            if 'white_pipeline' in combined_pipeline and 'powerball_pipeline' in combined_pipeline:
                                self.models[f"{model_name}_white"] = combined_pipeline['white_pipeline']
                                self.models[f"{model_name}_powerball"] = combined_pipeline['powerball_pipeline']
                                loaded_count += 1
                                logger.info(f"Loaded persisted model: {model_name} from {model_path}")
                            else:
                                logger.warning(f"Skipping invalid or legacy model structure in {model_path}")
                        else:
                            # Legacy single pipeline - store as is
                            self.models[model_name] = combined_pipeline
                            loaded_count += 1
                            logger.info(f"Loaded legacy model: {model_name} from {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to load model {model_name} from {model_path}: {e}")
                else:
                    logger.warning(f"Model file not found: {model_path} for {model_name}")
            
            if loaded_count > 0:
                logger.info(f"Successfully loaded {loaded_count} persisted models on startup")
            else:
                logger.info("No persisted models found to load on startup")
                
        except Exception as e:
            logger.error(f"Error loading persisted models: {e}")
    
    def train_models(self, df: pd.DataFrame, model_names: Optional[List[str]] = None, 
                    hyperparameters: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Train multiple ML models with standardized pipeline.
        
        Args:
            df: Historical lottery data
            model_names: List of models to train (default: all supported)
            hyperparameters: Custom hyperparameters for each model
            
        Returns:
            Dictionary with training results and performance metrics
        """
        if model_names is None:
            model_names = list(self.supported_models.keys())
        
        if hyperparameters is None:
            hyperparameters = self._get_default_hyperparameters()
        
        training_results = {}
        
        try:
            # Prepare features and targets
            X, y_white, y_powerball, feature_names = self._prepare_training_data(df)

            # _prepare_training_data returns empty arrays on failure; check size
            if X.size == 0:
                raise ValueError("No training data available")
            
            # Train each requested model
            for model_name in model_names:
                if model_name in self.supported_models:
                    result = self._train_single_model(
                        model_name, X, y_white, y_powerball, 
                        hyperparameters.get(model_name, {})
                    )
                    training_results[model_name] = result
                    
                    logger.info(f"Trained {model_name} - White MAE: {result['white_mae']:.2f}, "
                              f"Powerball MAE: {result['powerball_mae']:.2f}")
                else:
                    logger.warning(f"Unsupported model: {model_name}")
            
            return {
                'models_trained': list(training_results.keys()),
                'results': training_results,
                'training_data_size': len(X),
                'feature_count': X.shape[1],
                'feature_names': feature_names,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return {'error': str(e), 'models_trained': []}
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare features and targets for training.

        This method always returns numpy arrays and a list of feature names.
        On error or when data is insufficient it returns empty arrays (shape[0]==0)
        rather than None so callers can rely on consistent types.
        """
        try:
            # Engineer features using centralized service
            features_df = self.feature_service.engineer_features(df)

            if features_df is None or features_df.empty:
                logger.warning("Feature engineering returned no features")
                return np.empty((0, 0)), np.empty((0, len(WHITE_BALL_COLS))), np.empty((0,)), []

            X = features_df.values
            feature_names = features_df.columns.tolist()

            # Prepare targets using canonical column constants and validate presence
            white_cols = WHITE_BALL_COLS
            missing_white = [c for c in white_cols if c not in df.columns]
            if missing_white:
                logger.warning(f"Missing white-ball target columns: {missing_white}")
                return np.empty((0, X.shape[1])), np.empty((0, len(white_cols))), np.empty((0,)), feature_names

            if POWERBALL_COL not in df.columns:
                logger.warning(f"Missing powerball target column: {POWERBALL_COL}")
                return np.empty((0, X.shape[1])), np.empty((0, len(white_cols))), np.empty((0,)), feature_names

            # Ensure targets are numpy ndarrays (avoid pandas ExtensionArray types)
            y_white = np.asarray(df[white_cols].to_numpy())
            y_powerball = np.asarray(df[POWERBALL_COL].to_numpy())

            # Ensure we have enough data
            if X.shape[0] < 10:
                logger.warning(f"Insufficient training data: {X.shape[0]} samples")
                return np.empty((0, X.shape[1])), np.empty((0, len(white_cols))), np.empty((0,)), feature_names

            return X, y_white, y_powerball, feature_names

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.empty((0, 0)), np.empty((0, len(WHITE_BALL_COLS))), np.empty((0,)), []
    
    def _train_single_model(self, model_name: str, X: np.ndarray, y_white: np.ndarray, 
                           y_powerball: np.ndarray, hyperparams: Dict) -> Dict[str, Any]:
        """Train a single model with cross-validation."""
        try:
            # Get model class and create instances
            model_class = self.supported_models[model_name]
            
            # Create white ball model (multi-output)
            white_model = model_class(**hyperparams, random_state=42)
            white_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', MultiOutputRegressor(white_model))
            ])
            
            # Create powerball model
            powerball_model = model_class(**hyperparams, random_state=42)
            powerball_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', powerball_model)
            ])
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Cross-validate white ball model
            white_scores = cross_val_score(
                white_pipeline, X, y_white, 
                cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1
            )
            white_mae = -white_scores.mean()
            
            # Cross-validate powerball model
            powerball_scores = cross_val_score(
                powerball_pipeline, X, y_powerball,
                cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1
            )
            powerball_mae = -powerball_scores.mean()
            
            # Train final models on full dataset
            white_pipeline.fit(X, y_white)
            powerball_pipeline.fit(X, y_powerball)
            
            # Create combined pipeline for persistence
            combined_pipeline = {
                'white_pipeline': white_pipeline,
                'powerball_pipeline': powerball_pipeline,
                'model_type': model_name,
                'trained_at': datetime.now().isoformat()
            }
            
            # Save model to disk
            model_artifact_path = self._save_model_to_disk(model_name, combined_pipeline)
            
            # Store models in memory
            self.models[f"{model_name}_white"] = white_pipeline
            self.models[f"{model_name}_powerball"] = powerball_pipeline
            
            # Performance metrics
            performance_metrics = {
                'white_mae': float(white_mae),
                'powerball_mae': float(powerball_mae),
                'white_std': float(white_scores.std()),
                'powerball_std': float(powerball_scores.std()),
                'cv_splits': 5,
                'training_samples': len(X),
                'feature_count': X.shape[1]
            }
            
            self.model_performance[model_name] = performance_metrics
            
            return {
                'model_name': model_name,
                'white_mae': white_mae,
                'powerball_mae': powerball_mae,
                'white_pipeline': white_pipeline,
                'powerball_pipeline': powerball_pipeline,
                'performance_metrics': performance_metrics,
                'hyperparameters': hyperparams,
                'model_artifact_path': model_artifact_path,
                'training_completed': True
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'training_completed': False
            }
    
    def generate_predictions(self, model_name: str, X_latest: np.ndarray, 
                           prediction_count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate predictions using trained models.
        
        Args:
            model_name: Name of the model to use
            X_latest: The latest feature vector for prediction
            prediction_count: Number of predictions to generate
            
        Returns:
            List of prediction dictionaries
        """
        try:
            # Check if model is trained
            white_model_key = f"{model_name}_white"
            powerball_model_key = f"{model_name}_powerball"
            
            if white_model_key not in self.models or powerball_model_key not in self.models:
                raise ValueError(f"Model {model_name} not trained")
            
            if X_latest is None or X_latest.size == 0:
                raise ValueError("No feature data available for prediction")
            
            # Ensure correct shape if needed
            if X_latest.ndim == 1:
                X_latest = X_latest.reshape(1, -1)
            
            predictions = []
            
            for i in range(prediction_count):
                # Add small random noise for variation
                X_pred = X_latest + np.random.normal(0, 0.01, X_latest.shape)
                
                # Predict white balls
                white_pred = self.models[white_model_key].predict(X_pred)[0]
                white_numbers = np.clip(np.round(white_pred), 1, 69).astype(int)
                white_numbers = self._ensure_unique_numbers(white_numbers, 1, 69, 5)
                
                # Predict powerball
                powerball_pred = self.models[powerball_model_key].predict(X_pred)[0]
                powerball = int(np.clip(np.round(powerball_pred), 1, 26))
                
                # Calculate probability estimate
                probability = self._calculate_probability_estimate(white_numbers, powerball, model_name)
                
                prediction = {
                    'white_numbers': sorted(white_numbers.tolist()),
                    'powerball': powerball,
                    'probability': probability,
                    'model_used': model_name,
                    'prediction_index': i + 1
                }
                
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions with {model_name}: {e}")
            return []
    
    def store_model_predictions(self, model_name: str, predictions: List[Dict], 
                              performance_metrics: Dict, hyperparameters: Dict,
                              features_used: List[str], model_artifact_path: Optional[str] = None, 
                              training_duration: float = 0.0, notes: str = "") -> str:
        """
        Store model predictions using unified storage interface.
        
        Args:
            model_name: Name of the model
            predictions: List of predictions
            performance_metrics: Model performance metrics
            hyperparameters: Model hyperparameters
            features_used: List of feature names used in training
            model_artifact_path: Path to the persisted model artifact
            training_duration: Time taken for training
            notes: Additional notes
            
        Returns:
            Prediction set ID
        """
        try:
            set_id = self.storage_manager.store_model_predictions(
                model_name=model_name,
                predictions=predictions,
                hyperparameters=hyperparameters,
                performance_metrics=performance_metrics,
                features_used=features_used,
                model_artifact_path=model_artifact_path,
                training_duration=training_duration,
                notes=notes
            )
            logger.info(f"Stored {len(predictions)} predictions for {model_name} (set: {set_id})")
            return set_id
        except Exception as e:
            logger.error(f"Error storing predictions for {model_name}: {e}")
            return ""
    
    def train_and_predict(self, df: pd.DataFrame, model_name: str, 
                         prediction_count: int = 5, hyperparameters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete training and prediction pipeline for a single model.
        
        Args:
            df: Historical data
            model_name: Model to train and use for predictions
            prediction_count: Number of predictions to generate
            hyperparameters: Custom hyperparameters
            
        Returns:
            Complete results including training metrics and predictions
        """
        start_time = datetime.now()
        
        try:
            # 1. Prepare data
            X, y_white, y_powerball, feature_names = self._prepare_training_data(df)
            if X.size == 0:
                raise ValueError("Data preparation failed.")
            
            # 2. Train the model
            model_result = self._train_single_model(
                model_name, X, y_white, y_powerball, 
                hyperparameters or self._get_default_hyperparameters().get(model_name, {})
            )
            if not model_result.get('training_completed'):
                raise ValueError(f"Training failed: {model_result.get('error', 'Unknown error')}")
            
            # 3. Generate predictions using the latest feature set
            X_latest = X[-1:] if len(X) > 0 else X
            predictions = self.generate_predictions(model_name, X_latest, prediction_count)
            if not predictions:
                return {'error': 'Prediction generation failed', 'predictions': []}
            
            # 4. Calculate training duration and store predictions
            training_duration = (datetime.now() - start_time).total_seconds()
            
            set_id = self.store_model_predictions(
                model_name=model_name,
                predictions=predictions,
                performance_metrics=model_result['performance_metrics'],
                hyperparameters=hyperparameters or {},
                features_used=feature_names,
                model_artifact_path=model_result.get('model_artifact_path'),
                training_duration=training_duration,
                notes="Trained and predicted via unified service"
            )
            
            return {
                'model_name': model_name,
                'training_results': model_result,
                'predictions': predictions,
                'set_id': set_id,
                'training_duration': training_duration,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in train_and_predict for {model_name}: {e}")
            return {'error': str(e), 'predictions': [], 'success': False}
    
    def _ensure_unique_numbers(self, numbers: np.ndarray, min_val: int, 
                              max_val: int, target_count: int) -> np.ndarray:
        """Ensure predicted numbers are unique within valid range."""
        unique_numbers = []
        used_numbers = set()
        
        for num in numbers:
            num = int(np.clip(num, min_val, max_val))
            if num not in used_numbers:
                unique_numbers.append(num)
                used_numbers.add(num)
        
        # Fill remaining slots if needed
        while len(unique_numbers) < target_count:
            num = np.random.randint(min_val, max_val + 1)
            if num not in used_numbers:
                unique_numbers.append(num)
                used_numbers.add(num)
        
        return np.array(unique_numbers[:target_count])
    
    def _calculate_probability_estimate(self, white_numbers: np.ndarray, 
                                       powerball: int, model_name: str) -> float:
        """Calculate probability estimate for predictions."""
        try:
            # Base probability for any specific combination
            base_prob = 1.0 / (292201338)  # Approximate Powerball odds
            
            # Adjust based on model performance
            if model_name in self.model_performance:
                perf = self.model_performance[model_name]
                white_confidence = max(0.1, 1.0 - (perf['white_mae'] / 10.0))
                powerball_confidence = max(0.1, 1.0 - (perf['powerball_mae'] / 10.0))
                adjustment = (white_confidence + powerball_confidence) / 2.0
            else:
                adjustment = 0.5
            
            return base_prob * adjustment * 1000000  # Scale for readability
            
        except Exception:
            return 0.001  # Fallback probability
    
    def _get_default_hyperparameters(self) -> Dict[str, Dict]:
        """Get default hyperparameters for all models."""
        return {
            'Ridge Regression': {
                'alpha': 1.0
            },
            'Random Forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            'Gradient Boosting': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        }
    
    def get_model_performance(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for trained models."""
        if model_name:
            return self.model_performance.get(model_name, {})
        return self.model_performance.copy()
    
    def get_trained_models(self) -> List[str]:
        """Get list of currently trained models."""
        trained = set()
        for key in self.models.keys():
            if key.endswith('_white'):
                mname = key.replace('_white', '')
                if f"{mname}_powerball" in self.models:
                    trained.add(mname)
        return list(trained)
    
    def clear_models(self) -> None:
        """Clear all trained models from memory."""
        self.models.clear()
        self.model_performance.clear()
        logger.info("Cleared all trained models from memory")