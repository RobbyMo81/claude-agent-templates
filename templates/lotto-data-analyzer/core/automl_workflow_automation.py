"""
AutoML Workflow Automation
=========================
Implements the final step of the AutoML workflow by automating the training
process using optimal hyperparameters discovered during tuning.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd

from .model_training_service import ModelTrainingService
from .experiment_tracker import get_experiment_tracker, ExperimentJob
from .storage import get_store

logger = logging.getLogger(__name__)

@dataclass
class AutoMLTrainingResult:
    """Data class for automated training results."""
    model_name: str
    best_params: Dict[str, Any]
    training_successful: bool
    prediction_set_id: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    training_duration: Optional[float] = None
    error_message: Optional[str] = None

class AutoMLWorkflowAutomator:
    """Automates the final step of the AutoML workflow."""

    def __init__(self):
        self.tracker = get_experiment_tracker()
        self.training_service = ModelTrainingService()
        self.data_store = get_store()

    def get_deployment_candidates(self) -> List[ExperimentJob]:
        """Get completed tuning jobs that are candidates for deployment."""
        try:
            all_jobs = self.tracker.get_experiment_jobs(limit=100)
            candidates = [job for job in all_jobs if job.status == 'completed' and job.best_trial_id]
            candidates.sort(key=lambda x: x.start_time, reverse=True)
            return candidates
        except Exception as e:
            logger.error(f"Error getting deployment candidates: {e}")
            return []

    def trigger_automated_training(self, job_id: str) -> AutoMLTrainingResult:
        """
        Trigger a new training run using the best parameters from a tuning job.
        """
        try:
            job = next((j for j in self.tracker.get_experiment_jobs(limit=1000) if j.job_id == job_id), None)
            if not job:
                raise ValueError(f"Tuning job with ID {job_id} not found.")

            best_trial_run = self.tracker.get_run_by_id(job.best_trial_id)
            if not best_trial_run:
                 raise ValueError(f"Best trial run {job.best_trial_id} not found for job {job_id}.")

            # The model name in the tuning config might be "RandomForest", but the service expects "Random Forest"
            model_map = {
                "RandomForest": "Random Forest",
                "GradientBoosting": "Gradient Boosting",
                "Ridge": "Ridge Regression",
                "Lasso": "Lasso Regression",
            }
            config_model_name = job.configuration.get("model_name")
            service_model_name = model_map.get(config_model_name)
            
            if not service_model_name:
                 raise ValueError(f"Model name '{config_model_name}' not supported by ModelTrainingService.")

            best_params = best_trial_run.hyperparameters
            
            logger.info(f"Triggering training for {service_model_name} with params: {best_params}")

            df = self.data_store.latest()
            if df.empty:
                raise ValueError("No data available for training.")

            # Trigger training using the ModelTrainingService
            training_result = self.training_service.train_and_predict(
                df=df,
                model_name=service_model_name,
                hyperparameters=best_params
            )

            if training_result.get('success'):
                return AutoMLTrainingResult(
                    model_name=service_model_name,
                    best_params=best_params,
                    training_successful=True,
                    prediction_set_id=training_result.get('set_id'),
                    performance_metrics=training_result.get('training_results', {}).get('performance_metrics'),
                    training_duration=training_result.get('training_duration')
                )
            else:
                return AutoMLTrainingResult(
                    model_name=service_model_name,
                    best_params=best_params,
                    training_successful=False,
                    error_message=training_result.get('error', 'Unknown training error')
                )

        except Exception as e:
            logger.error(f"Automated training failed: {e}")
            return AutoMLTrainingResult(
                model_name="Unknown",
                best_params={},
                training_successful=False,
                error_message=str(e)
            )

# Global instance for easy access
_automator = None

def get_automl_workflow_automator():
    """Get the global workflow automator instance."""
    global _automator
    if _automator is None:
        _automator = AutoMLWorkflowAutomator()
    return _automator