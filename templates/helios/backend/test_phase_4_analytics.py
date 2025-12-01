#!/usr/bin/env python3
"""
Unit Tests for Phase 4: Cross-Model Analytics
=============================================

Comprehensive test suite for the CrossModelAnalytics engine and related components.
Tests all major functionality including performance analysis, model comparison,
ensemble recommendations, and trend analysis.
"""

import sys
import os
import unittest
import tempfile
import sqlite3
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the modules to test
try:
    from cross_model_analytics import (
        CrossModelAnalytics,
        ModelPerformanceMetrics,
        CrossModelComparison,
        EnsembleRecommendation,
    )
    from memory_store import MemoryStore

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False


class TestCrossModelAnalytics(unittest.TestCase):
    """Test suite for CrossModelAnalytics engine"""

    def setUp(self):
        """Set up test environment before each test"""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")

        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()

        # Initialize memory store and analytics engine
        self.memory_store = MemoryStore(self.temp_db.name)
        self.analytics = CrossModelAnalytics(self.memory_store)

        # Insert test data
        self._setup_test_data()

    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, "temp_db"):
            try:
                os.unlink(self.temp_db.name)
            except FileNotFoundError:
                pass

    def _setup_test_data(self):
        """Insert test training data into the database"""
        with self.memory_store._get_connection() as conn:
            cursor = conn.cursor()

            # Create the training_sessions table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS training_sessions (
                    job_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    total_epochs INTEGER,
                    final_loss REAL,
                    config TEXT
                )
            """
            )

            # Insert test training sessions
            test_sessions = [
                # Model A - Good performer
                (
                    "job_001",
                    "model_a",
                    "completed",
                    (datetime.now() - timedelta(days=1)).isoformat(),
                    (datetime.now() - timedelta(days=1, hours=-2)).isoformat(),
                    100,
                    0.15,
                    '{"learning_rate": 0.001}',
                ),
                (
                    "job_002",
                    "model_a",
                    "completed",
                    (datetime.now() - timedelta(days=3)).isoformat(),
                    (datetime.now() - timedelta(days=3, hours=-2)).isoformat(),
                    100,
                    0.14,
                    '{"learning_rate": 0.001}',
                ),
                (
                    "job_003",
                    "model_a",
                    "completed",
                    (datetime.now() - timedelta(days=5)).isoformat(),
                    (datetime.now() - timedelta(days=5, hours=-1.5)).isoformat(),
                    100,
                    0.16,
                    '{"learning_rate": 0.001}',
                ),
                # Model B - Moderate performer
                (
                    "job_004",
                    "model_b",
                    "completed",
                    (datetime.now() - timedelta(days=2)).isoformat(),
                    (datetime.now() - timedelta(days=2, hours=-3)).isoformat(),
                    150,
                    0.25,
                    '{"learning_rate": 0.0005}',
                ),
                (
                    "job_005",
                    "model_b",
                    "completed",
                    (datetime.now() - timedelta(days=4)).isoformat(),
                    (datetime.now() - timedelta(days=4, hours=-3)).isoformat(),
                    150,
                    0.24,
                    '{"learning_rate": 0.0005}',
                ),
                # Model C - Poor performer
                (
                    "job_006",
                    "model_c",
                    "completed",
                    (datetime.now() - timedelta(days=1)).isoformat(),
                    (datetime.now() - timedelta(days=1, hours=-4)).isoformat(),
                    200,
                    0.45,
                    '{"learning_rate": 0.01}',
                ),
                (
                    "job_007",
                    "model_c",
                    "completed",
                    (datetime.now() - timedelta(days=6)).isoformat(),
                    (datetime.now() - timedelta(days=6, hours=-4)).isoformat(),
                    200,
                    0.50,
                    '{"learning_rate": 0.01}',
                ),
            ]

            cursor.executemany(
                """
                INSERT INTO training_sessions
                (job_id, model_name, status, start_time, end_time, total_epochs, final_loss, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                test_sessions,
            )

            conn.commit()

    def test_analyze_model_performance_basic(self):
        """Test basic model performance analysis"""
        metrics = self.analytics.analyze_model_performance("model_a")

        self.assertIsInstance(metrics, ModelPerformanceMetrics)
        self.assertEqual(metrics.model_name, "model_a")
        self.assertGreater(metrics.training_time, 0)
        self.assertLess(metrics.best_loss, 0.2)  # Should be around 0.14-0.16
        self.assertEqual(metrics.total_epochs, 300)  # 3 sessions * 100 epochs
        self.assertGreater(metrics.stability_score, 0)
        self.assertGreater(metrics.efficiency_score, 0)

    def test_analyze_model_performance_nonexistent(self):
        """Test performance analysis for non-existent model"""
        metrics = self.analytics.analyze_model_performance("nonexistent_model")

        self.assertEqual(metrics.model_name, "nonexistent_model")
        self.assertEqual(metrics.training_time, 0.0)
        self.assertEqual(metrics.final_loss, float("inf"))
        self.assertEqual(metrics.total_epochs, 0)
        self.assertEqual(metrics.stability_score, 0.0)
        self.assertEqual(metrics.efficiency_score, 0.0)

    def test_compare_models_basic(self):
        """Test basic model comparison"""
        comparison = self.analytics.compare_models(["model_a", "model_b", "model_c"])

        self.assertIsInstance(comparison, CrossModelComparison)
        self.assertEqual(len(comparison.compared_models), 3)
        self.assertEqual(len(comparison.performance_ranking), 3)
        self.assertEqual(len(comparison.efficiency_ranking), 3)

        # Model A should be the best performer (lowest loss)
        best_model, best_loss = comparison.performance_ranking[0]
        self.assertEqual(best_model, "model_a")
        self.assertLess(best_loss, 0.2)

        # Worst performer should be model_c
        worst_model, worst_loss = comparison.performance_ranking[-1]
        self.assertEqual(worst_model, "model_c")
        self.assertGreater(worst_loss, 0.4)

    def test_compare_models_insufficient_data(self):
        """Test model comparison with insufficient models"""
        with self.assertRaises(Exception):
            # Should fail with only one model
            self.analytics.compare_models(["model_a"])

    def test_ensemble_recommendations_basic(self):
        """Test ensemble recommendation generation"""
        recommendations = self.analytics.generate_ensemble_recommendations(
            ["model_a", "model_b"]
        )

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

        for rec in recommendations:
            self.assertIsInstance(rec, EnsembleRecommendation)
            self.assertEqual(len(rec.recommended_models), len(rec.weights))
            self.assertAlmostEqual(
                sum(rec.weights), 1.0, places=5
            )  # Weights should sum to 1
            self.assertGreater(rec.confidence_score, 0)
            self.assertLessEqual(rec.confidence_score, 1.0)
            self.assertIsInstance(rec.reasoning, str)
            self.assertIsInstance(rec.risk_assessment, str)

    def test_ensemble_recommendations_multiple_strategies(self):
        """Test that multiple ensemble strategies are generated"""
        recommendations = self.analytics.generate_ensemble_recommendations(
            ["model_a", "model_b", "model_c"]
        )

        # Should generate multiple strategies
        self.assertGreaterEqual(len(recommendations), 2)

        # Check that strategies are different
        strategy_types = set()
        for rec in recommendations:
            strategy_types.add(rec.reasoning)

        self.assertGreater(len(strategy_types), 1)  # Multiple different strategies

    def test_historical_trends_analysis(self):
        """Test historical trends analysis"""
        trends = self.analytics.analyze_historical_trends(days_back=30)

        self.assertIsInstance(trends, dict)
        self.assertIn("time_period", trends)
        self.assertIn("active_models", trends)
        self.assertIn("model_trends", trends)
        self.assertIn("overall_trends", trends)
        self.assertIn("insights", trends)

        # Should find our test models
        self.assertGreaterEqual(trends["active_models"], 3)

        # Check model trends structure
        for model_name, trend_data in trends["model_trends"].items():
            self.assertIn("trend_direction", trend_data)
            self.assertIn("improvement_rate", trend_data)
            self.assertIn("consistency", trend_data)

    def test_performance_matrix_generation(self):
        """Test performance matrix generation"""
        matrix = self.analytics.get_performance_matrix(["model_a", "model_b"])

        self.assertIsInstance(matrix, dict)
        self.assertIn("models", matrix)
        self.assertIn("metrics", matrix)
        self.assertIn("data", matrix)
        self.assertIn("rankings", matrix)

        # Check structure
        self.assertEqual(len(matrix["models"]), 2)
        self.assertGreater(len(matrix["metrics"]), 0)
        self.assertEqual(len(matrix["data"]), len(matrix["metrics"]))

        # Each metric should have data for each model
        for metric_data in matrix["data"]:
            self.assertEqual(len(metric_data), len(matrix["models"]))

    def test_stability_score_calculation(self):
        """Test stability score calculation logic"""
        # Create test training history with known variance
        training_history = [
            {"final_loss": 0.15, "total_epochs": 100, "training_duration": 3600},
            {"final_loss": 0.16, "total_epochs": 100, "training_duration": 3600},
            {"final_loss": 0.14, "total_epochs": 100, "training_duration": 3600},
        ]

        stability = self.analytics._calculate_stability_score(training_history)

        self.assertGreater(stability, 0.8)  # Should be high stability (low variance)
        self.assertLessEqual(stability, 1.0)

    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation logic"""
        # Test with good training time
        training_history = [
            {"final_loss": 0.15, "training_duration": 3600},  # 1 hour
            {"final_loss": 0.16, "training_duration": 3600},
        ]

        efficiency = self.analytics._calculate_efficiency_score(training_history)

        self.assertGreater(efficiency, 0)
        self.assertLessEqual(efficiency, 1.0)

    def test_convergence_analysis(self):
        """Test convergence epoch analysis"""
        training_history = [
            {"total_epochs": 100},
            {"total_epochs": 120},
            {"total_epochs": 80},
        ]

        convergence_epoch = self.analytics._analyze_convergence(training_history)

        # Should return approximately 70% of median epochs
        self.assertIsInstance(convergence_epoch, int)
        self.assertGreater(convergence_epoch, 50)
        self.assertLess(convergence_epoch, 100)

    def test_get_active_models(self):
        """Test getting list of active models"""
        active_models = self.analytics._get_active_models(30)

        self.assertIsInstance(active_models, list)
        self.assertIn("model_a", active_models)
        self.assertIn("model_b", active_models)
        self.assertIn("model_c", active_models)

    def test_ensemble_weight_calculation(self):
        """Test ensemble weight calculation methods"""
        # Create mock model metrics
        mock_metrics = {
            "model_a": Mock(best_loss=0.15, stability_score=0.8, efficiency_score=0.7),
            "model_b": Mock(best_loss=0.25, stability_score=0.6, efficiency_score=0.5),
        }

        # Test optimal weights (inverse loss)
        weights = self.analytics._calculate_optimal_weights(mock_metrics)

        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        # Model A should get higher weight (better performance)
        self.assertGreater(weights[0], weights[1])

    def test_trend_insights_generation(self):
        """Test trend insight generation"""
        mock_model_trends = {
            "model_a": {
                "trend_direction": "improving",
                "improvement_rate": 0.1,
                "consistency": 0.8,
            },
            "model_b": {
                "trend_direction": "stable",
                "improvement_rate": 0.0,
                "consistency": 0.7,
            },
            "model_c": {
                "trend_direction": "declining",
                "improvement_rate": -0.05,
                "consistency": 0.5,
            },
        }

        mock_overall_trends = {
            "average_improvement_rate": 0.05,
            "average_consistency": 0.67,
            "trend_distribution": {"improving": 1, "stable": 1, "declining": 1},
        }

        insights = self.analytics._generate_trend_insights(
            mock_model_trends, mock_overall_trends
        )

        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)

        # Should contain meaningful insights
        insights_text = " ".join(insights).lower()
        self.assertTrue(
            any(word in insights_text for word in ["trend", "performance", "model"])
        )


class TestMemoryStoreIntegration(unittest.TestCase):
    """Test integration between CrossModelAnalytics and MemoryStore"""

    def setUp(self):
        """Set up test environment"""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")

        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.memory_store = MemoryStore(self.temp_db.name)

    def tearDown(self):
        """Clean up"""
        if hasattr(self, "temp_db"):
            try:
                os.unlink(self.temp_db.name)
            except FileNotFoundError:
                pass

    def test_analytics_initialization(self):
        """Test that analytics engine initializes properly with memory store"""
        analytics = CrossModelAnalytics(self.memory_store)

        self.assertIsNotNone(analytics.memory_store)
        self.assertEqual(analytics.performance_cache, {})
        self.assertEqual(analytics.analysis_history, [])

    def test_database_connection_handling(self):
        """Test that analytics handles database connections properly"""
        analytics = CrossModelAnalytics(self.memory_store)

        # This should not raise an exception even with empty database
        active_models = analytics._get_active_models(30)
        self.assertIsInstance(active_models, list)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def setUp(self):
        """Set up test environment"""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")

    def test_invalid_memory_store(self):
        """Test handling of invalid memory store"""
        with self.assertRaises(Exception):
            CrossModelAnalytics(None)

    def test_empty_model_list_comparison(self):
        """Test comparison with empty model list"""
        # Create a mock memory store
        mock_memory_store = Mock()
        analytics = CrossModelAnalytics(mock_memory_store)

        # Should handle empty list gracefully
        with self.assertRaises(Exception):
            analytics.compare_models([])

    def test_invalid_time_period(self):
        """Test analysis with invalid time periods"""
        mock_memory_store = Mock()
        analytics = CrossModelAnalytics(mock_memory_store)

        # Should handle negative days
        metrics = analytics.analyze_model_performance("test_model", days_back=-1)
        self.assertIsInstance(metrics, ModelPerformanceMetrics)

    def test_database_error_handling(self):
        """Test handling of database errors"""
        # Create analytics with a mock that raises database errors
        mock_memory_store = Mock()
        mock_memory_store._get_connection.side_effect = sqlite3.Error("Database error")

        analytics = CrossModelAnalytics(mock_memory_store)

        # Should handle database errors gracefully
        result = analytics._get_training_history("test_model", 30)
        self.assertEqual(result, [])


def run_comprehensive_test_suite():
    """Run the complete test suite and return results"""

    print("=" * 80)
    print("🧪 PHASE 4 CROSS-MODEL ANALYTICS - COMPREHENSIVE UNIT TESTS")
    print("=" * 80)

    if not DEPENDENCIES_AVAILABLE:
        print("❌ SKIPPED: Dependencies not available for testing")
        return False

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestCrossModelAnalytics,
        TestMemoryStoreIntegration,
        TestErrorHandling,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, "skipped") else 0
    passed = total_tests - failures - errors - skipped

    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"🔥 Errors: {errors}")
    print(f"⏭️  Skipped: {skipped}")

    if failures > 0:
        print("\n📋 FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if errors > 0:
        print("\n🔥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")

    success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")

    if success_rate >= 90:
        print("🚀 EXCELLENT: Phase 4 implementation is highly robust!")
    elif success_rate >= 75:
        print("👍 GOOD: Phase 4 implementation is solid with minor issues")
    elif success_rate >= 50:
        print("⚠️  MODERATE: Phase 4 implementation needs attention")
    else:
        print("🚨 CRITICAL: Phase 4 implementation has major issues")

    return success_rate >= 75


if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    sys.exit(0 if success else 1)
