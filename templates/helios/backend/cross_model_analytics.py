#!/usr/bin/env python3
"""
Cross-Model Analytics Engine
===========================

Phase 4.1 Implementation: Advanced cross-model comparison and analysis
Provides comprehensive analytics across multiple models including:
- Performance comparison matrices
- Training efficiency analysis
- Model evolution tracking
- Ensemble recommendation algorithms
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive performance metrics for a single model"""

    model_name: str
    training_time: float
    final_loss: float
    best_loss: float
    total_epochs: int
    convergence_epoch: Optional[int]
    stability_score: float
    efficiency_score: float
    last_updated: datetime


@dataclass
class CrossModelComparison:
    """Comparison analysis between multiple models"""

    compared_models: List[str]
    performance_ranking: List[Tuple[str, float]]
    efficiency_ranking: List[Tuple[str, float]]
    convergence_analysis: Dict[str, int]
    recommendation_score: Dict[str, float]
    ensemble_potential: float
    analysis_timestamp: datetime


@dataclass
class EnsembleRecommendation:
    """Recommendation for model ensemble configuration"""

    recommended_models: List[str]
    weights: List[float]
    expected_performance: float
    confidence_score: float
    reasoning: str
    risk_assessment: str


class CrossModelAnalytics:
    """
    Advanced analytics engine for cross-model analysis and comparison
    """

    def __init__(self, memory_store, models_dir: str = "models"):
        self.memory_store = memory_store
        self.models_dir = Path(models_dir)
        self.performance_cache = {}
        self.analysis_history = []

    def analyze_model_performance(
        self, model_name: str, days_back: int = 30
    ) -> ModelPerformanceMetrics:
        """
        Comprehensive performance analysis for a single model

        Args:
            model_name: Name of the model to analyze
            days_back: Number of days of historical data to consider

        Returns:
            ModelPerformanceMetrics object with comprehensive analysis
        """
        logger.info(f"Analyzing performance for model: {model_name}")

        # Get training history
        training_history = self._get_training_history(model_name, days_back)

        if not training_history:
            logger.warning(f"No training history found for model: {model_name}")
            return ModelPerformanceMetrics(
                model_name=model_name,
                training_time=0.0,
                final_loss=float("inf"),
                best_loss=float("inf"),
                total_epochs=0,
                convergence_epoch=None,
                stability_score=0.0,
                efficiency_score=0.0,
                last_updated=datetime.now(),
            )

        # Calculate performance metrics
        final_loss = training_history[-1].get("final_loss", float("inf"))
        best_loss = min([h.get("final_loss", float("inf")) for h in training_history])
        total_epochs = sum([h.get("total_epochs", 0) for h in training_history])
        training_time = sum([h.get("training_duration", 0) for h in training_history])

        # Analyze convergence
        convergence_epoch = self._analyze_convergence(training_history)

        # Calculate stability score (consistency of performance)
        stability_score = self._calculate_stability_score(training_history)

        # Calculate efficiency score (performance per unit time)
        efficiency_score = self._calculate_efficiency_score(training_history)

        return ModelPerformanceMetrics(
            model_name=model_name,
            training_time=training_time,
            final_loss=final_loss,
            best_loss=best_loss,
            total_epochs=total_epochs,
            convergence_epoch=convergence_epoch,
            stability_score=stability_score,
            efficiency_score=efficiency_score,
            last_updated=datetime.now(),
        )

    def compare_models(
        self, model_names: List[str], comparison_type: str = "comprehensive"
    ) -> CrossModelComparison:
        """
        Compare multiple models across various metrics

        Args:
            model_names: List of model names to compare
            comparison_type: Type of comparison ("performance", "efficiency", "comprehensive")

        Returns:
            CrossModelComparison object with detailed analysis
        """
        logger.info(f"Comparing models: {model_names} ({comparison_type})")

        # Analyze each model
        model_metrics = {}
        for model_name in model_names:
            model_metrics[model_name] = self.analyze_model_performance(model_name)

        # Performance ranking (lower loss is better)
        performance_ranking = sorted(
            [(name, metrics.best_loss) for name, metrics in model_metrics.items()],
            key=lambda x: x[1],
        )

        # Efficiency ranking (higher efficiency score is better)
        efficiency_ranking = sorted(
            [
                (name, metrics.efficiency_score)
                for name, metrics in model_metrics.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        # Convergence analysis
        convergence_analysis = {
            name: metrics.convergence_epoch or metrics.total_epochs
            for name, metrics in model_metrics.items()
        }

        # Calculate recommendation scores
        recommendation_score = self._calculate_recommendation_scores(model_metrics)

        # Assess ensemble potential
        ensemble_potential = self._assess_ensemble_potential(model_metrics)

        return CrossModelComparison(
            compared_models=model_names,
            performance_ranking=performance_ranking,
            efficiency_ranking=efficiency_ranking,
            convergence_analysis=convergence_analysis,
            recommendation_score=recommendation_score,
            ensemble_potential=ensemble_potential,
            analysis_timestamp=datetime.now(),
        )

    def generate_ensemble_recommendations(
        self, model_names: List[str], target_metric: str = "loss"
    ) -> List[EnsembleRecommendation]:
        """
        Generate ensemble recommendations for given models

        Args:
            model_names: List of available models
            target_metric: Metric to optimize for ("loss", "stability", "efficiency")

        Returns:
            List of EnsembleRecommendation objects ranked by expected performance
        """
        logger.info(
            f"Generating ensemble recommendations for {len(model_names)} models"
        )

        # Analyze all models
        model_metrics = {
            name: self.analyze_model_performance(name) for name in model_names
        }

        recommendations = []

        # Strategy 1: Best performers ensemble
        best_performers = self._select_best_performers(model_metrics, top_n=3)
        if len(best_performers) >= 2:
            weights = self._calculate_optimal_weights(best_performers)
            rec = EnsembleRecommendation(
                recommended_models=list(best_performers.keys()),
                weights=weights,
                expected_performance=self._estimate_ensemble_performance(
                    best_performers, weights
                ),
                confidence_score=0.8,
                reasoning="Top performing models with balanced weights",
                risk_assessment="Low risk - proven performers",
            )
            recommendations.append(rec)

        # Strategy 2: Diverse models ensemble
        diverse_models = self._select_diverse_models(model_metrics)
        if len(diverse_models) >= 2:
            weights = self._calculate_diversity_weights(diverse_models)
            rec = EnsembleRecommendation(
                recommended_models=list(diverse_models.keys()),
                weights=weights,
                expected_performance=self._estimate_ensemble_performance(
                    diverse_models, weights
                ),
                confidence_score=0.6,
                reasoning="Diverse models for robust predictions",
                risk_assessment="Medium risk - experimental combination",
            )
            recommendations.append(rec)

        # Strategy 3: Stability-focused ensemble
        stable_models = self._select_stable_models(model_metrics)
        if len(stable_models) >= 2:
            weights = self._calculate_stability_weights(stable_models)
            rec = EnsembleRecommendation(
                recommended_models=list(stable_models.keys()),
                weights=weights,
                expected_performance=self._estimate_ensemble_performance(
                    stable_models, weights
                ),
                confidence_score=0.9,
                reasoning="Stable models for consistent performance",
                risk_assessment="Very low risk - consistent performers",
            )
            recommendations.append(rec)

        # Sort by expected performance
        recommendations.sort(key=lambda x: x.expected_performance)

        return recommendations

    def analyze_historical_trends(self, days_back: int = 90) -> Dict[str, Any]:
        """
        Analyze historical trends across all models

        Args:
            days_back: Number of days to analyze

        Returns:
            Dictionary with trend analysis results
        """
        logger.info(f"Analyzing historical trends for {days_back} days")

        # Get all models with activity in the time period
        models_with_activity = self._get_active_models(days_back)

        trend_analysis = {
            "time_period": f"{days_back} days",
            "active_models": len(models_with_activity),
            "model_trends": {},
            "overall_trends": {},
            "insights": [],
        }

        # Analyze trends for each model
        for model_name in models_with_activity:
            model_trend = self._analyze_model_trend(model_name, days_back)
            trend_analysis["model_trends"][model_name] = model_trend

        # Calculate overall trends
        trend_analysis["overall_trends"] = self._calculate_overall_trends(
            trend_analysis["model_trends"]
        )

        # Generate insights
        trend_analysis["insights"] = self._generate_trend_insights(
            trend_analysis["model_trends"], trend_analysis["overall_trends"]
        )

        return trend_analysis

    def get_performance_matrix(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Generate a comprehensive performance comparison matrix

        Args:
            model_names: List of models to include in matrix

        Returns:
            Dictionary with matrix data suitable for visualization
        """
        logger.info(f"Generating performance matrix for {len(model_names)} models")

        matrix_data = {
            "models": model_names,
            "metrics": [
                "final_loss",
                "best_loss",
                "stability_score",
                "efficiency_score",
            ],
            "data": [],
            "rankings": {},
            "correlations": {},
            "generated_at": datetime.now().isoformat(),
        }

        # Collect metrics for all models
        model_metrics = {}
        for model_name in model_names:
            metrics = self.analyze_model_performance(model_name)
            model_metrics[model_name] = {
                "final_loss": metrics.final_loss,
                "best_loss": metrics.best_loss,
                "stability_score": metrics.stability_score,
                "efficiency_score": metrics.efficiency_score,
            }

        # Build matrix data
        for metric in matrix_data["metrics"]:
            row = []
            for model_name in model_names:
                value = model_metrics[model_name][metric]
                row.append(value if value != float("inf") else None)
            matrix_data["data"].append(row)

        # Calculate rankings for each metric
        for i, metric in enumerate(matrix_data["metrics"]):
            values = [
                (model_names[j], matrix_data["data"][i][j])
                for j in range(len(model_names))
                if matrix_data["data"][i][j] is not None
            ]

            # Sort (ascending for loss metrics, descending for scores)
            reverse = metric in ["stability_score", "efficiency_score"]
            sorted_values = sorted(values, key=lambda x: x[1], reverse=reverse)
            matrix_data["rankings"][metric] = sorted_values

        return matrix_data

    # Helper methods

    def _get_training_history(
        self, model_name: str, days_back: int
    ) -> List[Dict[str, Any]]:
        """Get training history for a model from memory store"""
        try:
            with self.memory_store._get_connection() as conn:
                cursor = conn.cursor()

                since_date = (datetime.now() - timedelta(days=days_back)).isoformat()

                # Get training sessions and join with training logs to get final loss
                cursor.execute(
                    """
                    SELECT ts.job_id, ts.status, ts.start_time, ts.end_time, ts.config,
                           tl.epoch as max_epoch, tl.loss as final_loss
                    FROM training_sessions ts
                    LEFT JOIN (
                        SELECT job_id, MAX(epoch) as max_epoch, epoch, loss
                        FROM training_logs
                        GROUP BY job_id
                        HAVING epoch = MAX(epoch)
                    ) tl ON ts.job_id = tl.job_id
                    WHERE ts.model_name = ? AND ts.start_time >= ? AND ts.status = 'completed'
                    ORDER BY ts.start_time DESC
                """,
                    (model_name, since_date),
                )

                rows = cursor.fetchall()

                history = []
                for row in rows:
                    config = json.loads(row[4]) if row[4] else {}

                    # Calculate training duration
                    training_duration = 0
                    if row[2] and row[3]:  # start_time and end_time
                        start = datetime.fromisoformat(row[2])
                        end = datetime.fromisoformat(row[3])
                        training_duration = (end - start).total_seconds()

                    # Extract total epochs from config or use max_epoch from logs
                    total_epochs = config.get("epochs", row[5] or 0)

                    history.append(
                        {
                            "job_id": row[0],
                            "status": row[1],
                            "start_time": row[2],
                            "end_time": row[3],
                            "total_epochs": total_epochs,
                            "final_loss": row[6] or float("inf"),
                            "training_duration": training_duration,
                            "config": config,
                        }
                    )

                return history

        except Exception as e:
            logger.error(f"Error getting training history for {model_name}: {e}")
            return []

    def _analyze_convergence(
        self, training_history: List[Dict[str, Any]]
    ) -> Optional[int]:
        """Analyze at what epoch the model typically converges"""
        if not training_history:
            return None

        # For now, use a simple heuristic - convergence is when loss improvement
        # becomes minimal. More sophisticated analysis could use loss curves.
        total_epochs = [h.get("total_epochs", 0) for h in training_history]
        if total_epochs:
            return int(
                np.median(total_epochs) * 0.7
            )  # Assume convergence at 70% of training

        return None

    def _calculate_stability_score(
        self, training_history: List[Dict[str, Any]]
    ) -> float:
        """Calculate stability score based on loss variance across runs"""
        if len(training_history) < 2:
            return 0.5  # Neutral score for insufficient data

        losses = [
            h.get("final_loss", float("inf"))
            for h in training_history
            if h.get("final_loss", float("inf")) != float("inf")
        ]

        if not losses:
            return 0.0

        # Higher stability = lower variance
        variance = np.var(losses)
        mean_loss = np.mean(losses)

        if mean_loss == 0:
            return 1.0

        coefficient_of_variation = np.sqrt(variance) / mean_loss
        stability_score = max(0.0, 1.0 - coefficient_of_variation)

        return min(1.0, stability_score)

    def _calculate_efficiency_score(
        self, training_history: List[Dict[str, Any]]
    ) -> float:
        """Calculate efficiency score (performance improvement per unit time)"""
        if not training_history:
            return 0.0

        # Calculate average performance per minute of training
        total_time = sum([h.get("training_duration", 0) for h in training_history])
        if total_time == 0:
            return 0.0

        losses = [
            h.get("final_loss", float("inf"))
            for h in training_history
            if h.get("final_loss", float("inf")) != float("inf")
        ]

        if not losses:
            return 0.0

        avg_loss = np.mean(losses)
        time_hours = total_time / 3600  # Convert to hours

        # Efficiency = 1 / (loss * time_hours)
        # Normalize to 0-1 scale
        if avg_loss == 0 or time_hours == 0:
            return 1.0

        efficiency = 1.0 / (avg_loss * time_hours)
        return min(1.0, float(efficiency / 10.0))  # Scale factor for normalization

    def _calculate_recommendation_scores(
        self, model_metrics: Dict[str, ModelPerformanceMetrics]
    ) -> Dict[str, float]:
        """Calculate overall recommendation scores for models"""
        scores = {}

        for name, metrics in model_metrics.items():
            # Weighted combination of performance, stability, and efficiency
            performance_score = (
                1.0 / (1.0 + metrics.best_loss)
                if metrics.best_loss != float("inf")
                else 0.0
            )
            stability_score = metrics.stability_score
            efficiency_score = metrics.efficiency_score

            # Weighted average (performance weighted highest)
            overall_score = (
                0.5 * performance_score + 0.3 * stability_score + 0.2 * efficiency_score
            )

            scores[name] = overall_score

        return scores

    def _assess_ensemble_potential(
        self, model_metrics: Dict[str, ModelPerformanceMetrics]
    ) -> float:
        """Assess how well the models would work together in an ensemble"""
        if len(model_metrics) < 2:
            return 0.0

        # Diversity factor - models should have different strengths
        performance_scores = [
            1.0 / (1.0 + m.best_loss) if m.best_loss != float("inf") else 0.0
            for m in model_metrics.values()
        ]

        # Ensemble potential is higher when models have diverse but good performance
        diversity = np.std(performance_scores)
        average_performance = np.mean(performance_scores)

        # Balance diversity and performance
        ensemble_potential = 0.7 * average_performance + 0.3 * diversity

        return min(1.0, float(ensemble_potential))

    def _select_best_performers(
        self, model_metrics: Dict[str, ModelPerformanceMetrics], top_n: int = 3
    ) -> Dict[str, ModelPerformanceMetrics]:
        """Select top performing models"""
        sorted_models = sorted(model_metrics.items(), key=lambda x: x[1].best_loss)

        return dict(sorted_models[:top_n])

    def _select_diverse_models(
        self, model_metrics: Dict[str, ModelPerformanceMetrics]
    ) -> Dict[str, ModelPerformanceMetrics]:
        """Select models with diverse characteristics"""
        # Simple diversity selection based on different performance profiles
        models = list(model_metrics.items())

        if len(models) <= 2:
            return dict(models)

        # Select models with different stability/efficiency profiles
        selected = {}

        # Find most stable model
        most_stable = max(models, key=lambda x: x[1].stability_score)
        selected[most_stable[0]] = most_stable[1]

        # Find most efficient model (if different)
        most_efficient = max(models, key=lambda x: x[1].efficiency_score)
        if most_efficient[0] not in selected:
            selected[most_efficient[0]] = most_efficient[1]

        # Find best performer (if different)
        best_performer = min(models, key=lambda x: x[1].best_loss)
        if best_performer[0] not in selected:
            selected[best_performer[0]] = best_performer[1]

        return selected

    def _select_stable_models(
        self, model_metrics: Dict[str, ModelPerformanceMetrics]
    ) -> Dict[str, ModelPerformanceMetrics]:
        """Select models with highest stability scores"""
        stable_threshold = 0.6

        stable_models = {
            name: metrics
            for name, metrics in model_metrics.items()
            if metrics.stability_score >= stable_threshold
        }

        if len(stable_models) < 2:
            # If not enough stable models, select top 2 by stability
            sorted_by_stability = sorted(
                model_metrics.items(), key=lambda x: x[1].stability_score, reverse=True
            )
            stable_models = dict(sorted_by_stability[:2])

        return stable_models

    def _calculate_optimal_weights(
        self, models: Dict[str, ModelPerformanceMetrics]
    ) -> List[float]:
        """Calculate optimal weights for ensemble based on inverse loss"""
        if not models:
            return []

        # Weight based on inverse of loss (better models get higher weight)
        inv_losses = []
        for metrics in models.values():
            if metrics.best_loss == float("inf") or metrics.best_loss == 0:
                inv_losses.append(1.0)
            else:
                inv_losses.append(1.0 / metrics.best_loss)

        # Normalize to sum to 1
        total = sum(inv_losses)
        if total == 0:
            return [1.0 / len(models)] * len(models)

        return [w / total for w in inv_losses]

    def _calculate_diversity_weights(
        self, models: Dict[str, ModelPerformanceMetrics]
    ) -> List[float]:
        """Calculate weights that emphasize diversity"""
        # For diversity ensemble, use more balanced weights
        n_models = len(models)
        if n_models == 0:
            return []

        # Start with equal weights, then adjust based on complementary strengths
        base_weight = 1.0 / n_models
        weights = []

        for metrics in models.values():
            # Slightly favor models with unique characteristics
            unique_factor = 1.0 + 0.1 * (metrics.efficiency_score - 0.5)
            weight = base_weight * unique_factor
            weights.append(weight)

        # Normalize
        total = sum(weights)
        return [w / total for w in weights]

    def _calculate_stability_weights(
        self, models: Dict[str, ModelPerformanceMetrics]
    ) -> List[float]:
        """Calculate weights based on stability scores"""
        if not models:
            return []

        stability_scores = [metrics.stability_score for metrics in models.values()]
        total_stability = sum(stability_scores)

        if total_stability == 0:
            return [1.0 / len(models)] * len(models)

        return [score / total_stability for score in stability_scores]

    def _estimate_ensemble_performance(
        self, models: Dict[str, ModelPerformanceMetrics], weights: List[float]
    ) -> float:
        """Estimate the performance of an ensemble"""
        if not models or not weights:
            return float("inf")

        # Weighted average of losses with ensemble benefit
        weighted_loss = sum(
            w * metrics.best_loss
            for w, metrics in zip(weights, models.values())
            if metrics.best_loss != float("inf")
        )

        # Ensemble typically performs better than weighted average
        ensemble_benefit = 0.95  # 5% improvement factor
        estimated_loss = weighted_loss * ensemble_benefit

        return estimated_loss

    def _get_active_models(self, days_back: int) -> List[str]:
        """Get list of models with activity in the specified time period"""
        try:
            with self.memory_store._get_connection() as conn:
                cursor = conn.cursor()

                since_date = (datetime.now() - timedelta(days=days_back)).isoformat()

                cursor.execute(
                    """
                    SELECT DISTINCT model_name
                    FROM training_sessions
                    WHERE start_time >= ?
                """,
                    (since_date,),
                )

                return [row[0] for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error getting active models: {e}")
            return []

    def _analyze_model_trend(self, model_name: str, days_back: int) -> Dict[str, Any]:
        """Analyze trend for a specific model"""
        training_history = self._get_training_history(model_name, days_back)

        if len(training_history) < 2:
            return {
                "trend_direction": "insufficient_data",
                "improvement_rate": 0.0,
                "consistency": 0.0,
            }

        # Analyze loss trend over time
        losses = [
            h.get("final_loss", float("inf"))
            for h in training_history
            if h.get("final_loss", float("inf")) != float("inf")
        ]

        if len(losses) < 2:
            return {
                "trend_direction": "no_valid_data",
                "improvement_rate": 0.0,
                "consistency": 0.0,
            }

        # Calculate trend direction
        recent_avg = np.mean(losses[: len(losses) // 2])  # More recent half
        older_avg = np.mean(losses[len(losses) // 2 :])  # Older half

        if recent_avg < older_avg:
            trend_direction = "improving"
            improvement_rate = (older_avg - recent_avg) / older_avg
        elif recent_avg > older_avg:
            trend_direction = "declining"
            improvement_rate = (recent_avg - older_avg) / older_avg  # Negative
        else:
            trend_direction = "stable"
            improvement_rate = 0.0

        # Calculate consistency (inverse of variance)
        consistency = 1.0 / (1.0 + np.var(losses))

        return {
            "trend_direction": trend_direction,
            "improvement_rate": improvement_rate,
            "consistency": consistency,
            "data_points": len(losses),
        }

    def _calculate_overall_trends(
        self, model_trends: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall trends across all models"""
        if not model_trends:
            return {}

        # Count trend directions
        trend_counts = defaultdict(int)
        improvement_rates = []
        consistency_scores = []

        for trend_data in model_trends.values():
            trend_counts[trend_data["trend_direction"]] += 1

            if trend_data["trend_direction"] != "insufficient_data":
                improvement_rates.append(trend_data["improvement_rate"])
                consistency_scores.append(trend_data["consistency"])

        return {
            "trend_distribution": dict(trend_counts),
            "average_improvement_rate": np.mean(improvement_rates)
            if improvement_rates
            else 0.0,
            "average_consistency": np.mean(consistency_scores)
            if consistency_scores
            else 0.0,
            "total_models_analyzed": len(model_trends),
        }

    def _generate_trend_insights(
        self, model_trends: Dict[str, Dict[str, Any]], overall_trends: Dict[str, Any]
    ) -> List[str]:
        """Generate human-readable insights from trend analysis"""
        insights = []

        if not model_trends:
            insights.append("No model data available for trend analysis")
            return insights

        # Overall trend insights
        trend_dist = overall_trends.get("trend_distribution", {})
        improving_count = trend_dist.get("improving", 0)
        declining_count = trend_dist.get("declining", 0)
        stable_count = trend_dist.get("stable", 0)

        total_valid = improving_count + declining_count + stable_count

        if total_valid > 0:
            if improving_count > total_valid * 0.6:
                insights.append(
                    "Most models are showing improvement in recent training sessions"
                )
            elif declining_count > total_valid * 0.6:
                insights.append(
                    "Concerning: Most models are showing declining performance"
                )
            elif stable_count > total_valid * 0.6:
                insights.append(
                    "Model performance is generally stable with minimal changes"
                )
            else:
                insights.append(
                    "Mixed trends: Models showing varied performance patterns"
                )

        # Improvement rate insights
        avg_improvement = overall_trends.get("average_improvement_rate", 0.0)
        if avg_improvement > 0.1:
            insights.append(
                f"Strong improvement trend: Average {avg_improvement:.1%} performance gain"
            )
        elif avg_improvement < -0.1:
            insights.append(
                f"Performance decline: Average {abs(avg_improvement):.1%} loss in performance"
            )

        # Consistency insights
        avg_consistency = overall_trends.get("average_consistency", 0.0)
        if avg_consistency > 0.8:
            insights.append(
                "High consistency: Models showing reliable performance patterns"
            )
        elif avg_consistency < 0.5:
            insights.append(
                "Low consistency: Models showing unpredictable performance variations"
            )

        return insights
