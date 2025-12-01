"""
Metacognitive Engine for Helios AI System
Phase 3 Implementation: Self-awareness and learning strategy capabilities

This module provides metacognitive capabilities including:
- Self-assessment of model performance
- Learning strategy optimization
- Confidence estimation
- Knowledge gap identification
- Adaptive learning rate adjustment
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import statistics

# Use absolute import instead of relative import
from memory_store import MemoryStore

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for metacognitive assessments"""

    VERY_LOW = 0.0
    LOW = 0.25
    MEDIUM = 0.5
    HIGH = 0.75
    VERY_HIGH = 1.0


class LearningStrategy(Enum):
    """Available learning strategies"""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"
    ACTIVE_LEARNING = "active_learning"


@dataclass
class MetacognitiveAssessment:
    """Results of a metacognitive self-assessment"""

    confidence_score: float
    predicted_performance: float
    uncertainty_estimate: float
    knowledge_gaps: List[str]
    recommended_strategy: LearningStrategy
    assessment_timestamp: datetime
    context: Dict[str, Any]


@dataclass
class PerformancePattern:
    """Identified pattern in model performance"""

    pattern_type: str
    pattern_strength: float
    conditions: Dict[str, Any]
    impact_score: float
    frequency: int
    last_observed: datetime


class MetacognitiveEngine:
    """
    Core metacognitive engine that provides self-awareness capabilities
    """

    def __init__(self, memory_store: MemoryStore):
        """
        Initialize the metacognitive engine

        Args:
            memory_store: Instance of MemoryStore for persistent data
        """
        self.memory_store = memory_store
        self.confidence_threshold = 0.7
        self.uncertainty_threshold = 0.3
        self.pattern_min_frequency = 3

        # Metacognitive parameters
        self.self_assessment_interval = 100  # Steps between assessments
        self.performance_window = 1000  # Recent steps to consider
        self.confidence_decay = 0.95  # Confidence decay per step

        logger.info("Metacognitive engine initialized")

    def assess_current_state(
        self,
        model_name: str,
        current_metrics: Dict[str, float],
        recent_performance: List[Any],  # Accept both List[float] and List[Dict]
        context: Optional[Dict[str, Any]] = None,
    ) -> MetacognitiveAssessment:
        """
        Perform comprehensive self-assessment of current model state

        Args:
            model_name: Name of the model being assessed
            current_metrics: Current performance metrics
            recent_performance: Recent performance history (can be floats or dicts with performance data)
            context: Additional context information

        Returns:
            MetacognitiveAssessment with detailed self-assessment results
        """
        logger.info(f"Starting metacognitive assessment for model {model_name}")

        # Convert recent_performance to float list if needed
        performance_values = self._extract_performance_values(recent_performance)

        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            current_metrics, performance_values
        )

        # Predict future performance
        predicted_performance = self._predict_performance(
            performance_values, current_metrics
        )

        # Estimate uncertainty
        uncertainty_estimate = self._estimate_uncertainty(
            performance_values, current_metrics
        )

        # Identify knowledge gaps
        knowledge_gaps = self._identify_knowledge_gaps(model_name, current_metrics)

        # Recommend learning strategy
        recommended_strategy = self._recommend_learning_strategy(
            confidence_score, uncertainty_estimate, performance_values
        )

        # Create assessment
        assessment = MetacognitiveAssessment(
            confidence_score=confidence_score,
            predicted_performance=predicted_performance,
            uncertainty_estimate=uncertainty_estimate,
            knowledge_gaps=knowledge_gaps,
            recommended_strategy=recommended_strategy,
            assessment_timestamp=datetime.now(),
            context=context or {},
        )

        # Store assessment in memory
        self._store_assessment(model_name, assessment)

        logger.info(
            f"Metacognitive assessment completed. Confidence: {confidence_score:.3f}, "
            f"Strategy: {recommended_strategy.value}"
        )

        return assessment

    def _extract_performance_values(self, recent_performance: List[Any]) -> List[float]:
        """Extract float performance values from mixed input formats"""
        if not recent_performance:
            return []

        performance_values = []
        for item in recent_performance:
            if isinstance(item, dict):
                # Extract from dict - prioritize common performance metrics
                if "accuracy" in item:
                    performance_values.append(float(item["accuracy"]))
                elif "loss" in item:
                    # Convert loss to performance score (1 - loss)
                    performance_values.append(max(0.0, 1.0 - float(item["loss"])))
                elif "score" in item:
                    performance_values.append(float(item["score"]))
                else:
                    # Use first numeric value found
                    for value in item.values():
                        if isinstance(value, (int, float)):
                            performance_values.append(float(value))
                            break
                    else:
                        # Default if no numeric value found
                        performance_values.append(0.5)
            elif isinstance(item, (int, float)):
                performance_values.append(float(item))
            else:
                # Default for unknown types
                performance_values.append(0.5)

        return performance_values

    def _calculate_confidence(
        self, current_metrics: Dict[str, float], recent_performance: List[float]
    ) -> float:
        """Calculate confidence score based on current state and recent performance"""
        if not recent_performance:
            return 0.5  # Neutral confidence with no history

        # Performance consistency factor
        performance_std = (
            float(np.std(recent_performance)) if len(recent_performance) > 1 else 0.0
        )
        consistency_factor = max(0.0, 1.0 - performance_std)

        # Recent trend factor
        if len(recent_performance) >= 5:
            recent_trend = float(
                np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
            )
            trend_factor = max(0.0, min(1.0, 0.5 + recent_trend))
        else:
            trend_factor = 0.5

        # Current performance level
        current_performance = recent_performance[-1] if recent_performance else 0.5
        performance_factor = max(0.0, min(1.0, current_performance))

        # Combine factors with weights
        confidence = (
            0.4 * performance_factor + 0.3 * consistency_factor + 0.3 * trend_factor
        )

        return max(0.0, min(1.0, confidence))

    def _predict_performance(
        self, recent_performance: List[float], current_metrics: Dict[str, float]
    ) -> float:
        """Predict future performance based on current trends"""
        if not recent_performance:
            return 0.5

        # Simple trend-based prediction
        if len(recent_performance) >= 3:
            # Use polynomial fitting for prediction
            x = np.array(range(len(recent_performance)))
            y = np.array(recent_performance)

            # Fit linear trend
            try:
                coeffs = np.polyfit(x, y, 1)
                predicted = float(coeffs[0]) * len(recent_performance) + float(
                    coeffs[1]
                )
                return max(0.0, min(1.0, predicted))
            except (np.linalg.LinAlgError, Warning):
                pass

        # Fallback to recent average
        return float(np.mean(recent_performance[-5:]))

    def _estimate_uncertainty(
        self, recent_performance: List[float], current_metrics: Dict[str, float]
    ) -> float:
        """Estimate uncertainty in current model state"""
        if not recent_performance:
            return 1.0  # Maximum uncertainty with no data

        # Performance variance
        performance_variance = (
            float(np.var(recent_performance)) if len(recent_performance) > 1 else 0.0
        )

        # Prediction error (if we have enough data)
        prediction_error = 0.0
        if len(recent_performance) >= 10:
            # Calculate rolling prediction errors
            errors = []
            for i in range(5, len(recent_performance)):
                actual = recent_performance[i]
                predicted = float(np.mean(recent_performance[i - 5 : i]))
                errors.append(abs(actual - predicted))
            prediction_error = float(np.mean(errors)) if errors else 0.0

        # Combine uncertainty factors
        uncertainty = min(1.0, performance_variance + prediction_error)

        return uncertainty

    def _identify_knowledge_gaps(
        self, model_name: str, current_metrics: Dict[str, float]
    ) -> List[str]:
        """Identify areas where the model lacks knowledge or performance"""
        gaps = []

        # Check performance metrics for weak areas
        for metric_name, metric_value in current_metrics.items():
            if metric_value < 0.6:  # Threshold for poor performance
                gaps.append(f"Low performance in {metric_name}: {metric_value:.3f}")

        # Analyze historical patterns for recurring issues
        historical_metrics = self.memory_store.get_performance_metrics(
            model_name=model_name, hours=168  # Last week
        )

        # Group by metric name and find consistently poor performers
        metric_groups = {}
        for metric in historical_metrics:
            name = metric["metric_name"]
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(metric["metric_value"])

        for metric_name, values in metric_groups.items():
            if len(values) >= 5:  # Need sufficient data
                avg_value = statistics.mean(values)
                if avg_value < 0.5:
                    gaps.append(f"Consistently poor {metric_name}: avg {avg_value:.3f}")

        return gaps

    def _recommend_learning_strategy(
        self, confidence: float, uncertainty: float, recent_performance: List[float]
    ) -> LearningStrategy:
        """Recommend optimal learning strategy based on current state"""

        # High confidence, low uncertainty -> Conservative
        if confidence > 0.8 and uncertainty < 0.2:
            return LearningStrategy.CONSERVATIVE

        # Low confidence, high uncertainty -> Aggressive
        if confidence < 0.4 and uncertainty > 0.6:
            return LearningStrategy.AGGRESSIVE

        # Check for performance plateau
        if len(recent_performance) >= 10:
            recent_trend = np.polyfit(
                range(len(recent_performance[-10:])), recent_performance[-10:], 1
            )[0]
            if abs(recent_trend) < 0.001:  # Flat trend
                return LearningStrategy.AGGRESSIVE

        # Check for instability (high variance)
        if len(recent_performance) > 5:
            recent_variance = np.var(recent_performance[-10:])
            if recent_variance > 0.1:
                return LearningStrategy.CONSERVATIVE

        # Default to balanced approach
        return LearningStrategy.BALANCED

    def _store_assessment(self, model_name: str, assessment: MetacognitiveAssessment):
        """Store metacognitive assessment in memory"""
        assessment_data = {
            "confidence_score": assessment.confidence_score,
            "predicted_performance": assessment.predicted_performance,
            "uncertainty_estimate": assessment.uncertainty_estimate,
            "knowledge_gaps": assessment.knowledge_gaps,
            "recommended_strategy": assessment.recommended_strategy.value,
            "context": assessment.context,
        }

        self.memory_store.store_enhanced_journal_entry(
            model_name=model_name,
            session_id="metacognitive_assessment",
            event_type="self_assessment",
            event_data=assessment_data,
            confidence_score=assessment.confidence_score,
        )

    def analyze_performance_patterns(
        self, model_name: str, days: int = 7
    ) -> List[PerformancePattern]:
        """Analyze historical performance to identify patterns"""
        logger.info(f"Analyzing performance patterns for {model_name} over {days} days")

        # Get historical data
        metrics = self.memory_store.get_performance_metrics(
            model_name=model_name, hours=days * 24
        )

        if len(metrics) < 10:
            logger.warning("Insufficient data for pattern analysis")
            return []

        patterns = []

        # Group metrics by type
        metric_groups = {}
        for metric in metrics:
            name = metric["metric_name"]
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(metric)

        # Analyze each metric type
        for metric_name, metric_data in metric_groups.items():
            if len(metric_data) < 5:
                continue

            values = [m["metric_value"] for m in metric_data]
            timestamps = [datetime.fromisoformat(m["timestamp"]) for m in metric_data]

            # Detect trends
            trend_pattern = self._detect_trend_pattern(metric_name, values, timestamps)
            if trend_pattern:
                patterns.append(trend_pattern)

            # Detect cyclical patterns
            cyclical_pattern = self._detect_cyclical_pattern(
                metric_name, values, timestamps
            )
            if cyclical_pattern:
                patterns.append(cyclical_pattern)

            # Detect anomalies
            anomaly_patterns = self._detect_anomaly_patterns(
                metric_name, values, timestamps
            )
            patterns.extend(anomaly_patterns)

        # Store patterns for future reference
        for pattern in patterns:
            self._store_pattern(model_name, pattern)

        logger.info(f"Identified {len(patterns)} performance patterns")
        return patterns

    def _detect_trend_pattern(
        self, metric_name: str, values: List[float], timestamps: List[datetime]
    ) -> Optional[PerformancePattern]:
        """Detect trend patterns in performance data"""
        if len(values) < 5:
            return None

        # Calculate trend using linear regression
        x = np.array(range(len(values)))
        y = np.array(values)

        try:
            slope, intercept = np.polyfit(x, y, 1)

            # Significant trend threshold
            if abs(float(slope)) > 0.01:
                pattern_type = "increasing_trend" if slope > 0 else "decreasing_trend"
                pattern_strength = min(1.0, abs(float(slope)) * 10)  # Scale to 0-1

                return PerformancePattern(
                    pattern_type=pattern_type,
                    pattern_strength=pattern_strength,
                    conditions={"metric_name": metric_name, "slope": float(slope)},
                    impact_score=pattern_strength,
                    frequency=len(values),
                    last_observed=timestamps[-1],
                )
        except (np.linalg.LinAlgError, Warning):
            pass

        return None

    def _detect_cyclical_pattern(
        self, metric_name: str, values: List[float], timestamps: List[datetime]
    ) -> Optional[PerformancePattern]:
        """Detect cyclical patterns in performance data"""
        if len(values) < 10:
            return None

        # Simple autocorrelation-based cycle detection
        # This is a simplified implementation - could be enhanced with FFT

        autocorr_scores = []
        for lag in range(1, min(len(values) // 2, 10)):
            correlation = np.corrcoef(values[:-lag], values[lag:])[0, 1]
            if not np.isnan(correlation):
                autocorr_scores.append((lag, abs(correlation)))

        if autocorr_scores:
            best_lag, best_correlation = max(autocorr_scores, key=lambda x: x[1])

            if best_correlation > 0.6:  # Strong cyclical pattern
                return PerformancePattern(
                    pattern_type="cyclical_pattern",
                    pattern_strength=best_correlation,
                    conditions={"metric_name": metric_name, "cycle_length": best_lag},
                    impact_score=best_correlation,
                    frequency=len(values) // best_lag,
                    last_observed=timestamps[-1],
                )

        return None

    def _detect_anomaly_patterns(
        self, metric_name: str, values: List[float], timestamps: List[datetime]
    ) -> List[PerformancePattern]:
        """Detect anomalous patterns in performance data"""
        if len(values) < 5:
            return []

        patterns = []
        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val == 0:
            return []

        # Detect outliers (values beyond 2 standard deviations)
        outliers = []
        for i, val in enumerate(values):
            z_score = abs(val - mean_val) / std_val
            if z_score > 2.0:
                outliers.append((i, val, z_score))

        if len(outliers) >= 2:
            avg_z_score = float(np.mean([z for _, _, z in outliers]))
            pattern = PerformancePattern(
                pattern_type="anomaly_cluster",
                pattern_strength=min(1.0, avg_z_score / 3.0),
                conditions={"metric_name": metric_name, "outlier_count": len(outliers)},
                impact_score=len(outliers) / len(values),
                frequency=len(outliers),
                last_observed=timestamps[-1],
            )
            patterns.append(pattern)

        return patterns

    def _store_pattern(self, model_name: str, pattern: PerformancePattern):
        """Store identified pattern in memory"""
        pattern_data = {
            "pattern_type": pattern.pattern_type,
            "pattern_strength": pattern.pattern_strength,
            "conditions": pattern.conditions,
            "impact_score": pattern.impact_score,
            "frequency": pattern.frequency,
        }

        self.memory_store.store_enhanced_journal_entry(
            model_name=model_name,
            session_id="pattern_analysis",
            event_type="performance_pattern",
            event_data=pattern_data,
            confidence_score=pattern.pattern_strength,
        )

    def get_learning_recommendations(
        self, model_name: str, current_assessment: MetacognitiveAssessment
    ) -> Dict[str, Any]:
        """Generate specific learning recommendations based on metacognitive assessment"""

        recommendations = {
            "learning_rate_adjustment": self._recommend_learning_rate(
                current_assessment
            ),
            "training_focus_areas": self._recommend_focus_areas(current_assessment),
            "exploration_strategy": self._recommend_exploration_strategy(
                current_assessment
            ),
            "memory_management": self._recommend_memory_management(current_assessment),
            "evaluation_frequency": self._recommend_evaluation_frequency(
                current_assessment
            ),
        }

        # Store recommendations
        self.memory_store.store_enhanced_journal_entry(
            model_name=model_name,
            session_id="learning_recommendations",
            event_type="metacognitive_recommendations",
            event_data=recommendations,
            confidence_score=current_assessment.confidence_score,
        )

        return recommendations

    def _recommend_learning_rate(
        self, assessment: MetacognitiveAssessment
    ) -> Dict[str, float]:
        """Recommend learning rate adjustments"""
        base_lr = 0.001

        if assessment.recommended_strategy == LearningStrategy.AGGRESSIVE:
            return {"base_rate": base_lr * 2.0, "decay_factor": 0.9}
        elif assessment.recommended_strategy == LearningStrategy.CONSERVATIVE:
            return {"base_rate": base_lr * 0.5, "decay_factor": 0.99}
        else:
            return {"base_rate": base_lr, "decay_factor": 0.95}

    def _recommend_focus_areas(self, assessment: MetacognitiveAssessment) -> List[str]:
        """Recommend areas to focus training on"""
        focus_areas = []

        # High-priority areas from knowledge gaps
        for gap in assessment.knowledge_gaps[:3]:  # Top 3 gaps
            focus_areas.append(gap)

        # Add general recommendations based on confidence
        if assessment.confidence_score < 0.5:
            focus_areas.append("fundamental_concepts")

        if assessment.uncertainty_estimate > 0.7:
            focus_areas.append("uncertainty_reduction")

        return focus_areas

    def _recommend_exploration_strategy(
        self, assessment: MetacognitiveAssessment
    ) -> str:
        """Recommend exploration vs exploitation strategy"""
        if assessment.confidence_score < 0.4:
            return "high_exploration"
        elif assessment.confidence_score > 0.8:
            return "exploitation_focused"
        else:
            return "balanced_exploration"

    def _recommend_memory_management(
        self, assessment: MetacognitiveAssessment
    ) -> Dict[str, Any]:
        """Recommend memory management strategies"""
        return {
            "retention_priority": "high"
            if assessment.confidence_score > 0.7
            else "medium",
            "compression_level": "low"
            if assessment.uncertainty_estimate > 0.5
            else "medium",
            "update_frequency": "high"
            if assessment.recommended_strategy == LearningStrategy.AGGRESSIVE
            else "normal",
        }

    def _recommend_evaluation_frequency(
        self, assessment: MetacognitiveAssessment
    ) -> int:
        """Recommend how often to perform evaluations"""
        if assessment.uncertainty_estimate > 0.7:
            return 50  # More frequent evaluation when uncertain
        elif assessment.confidence_score > 0.8:
            return 200  # Less frequent when confident
        else:
            return 100  # Default frequency
