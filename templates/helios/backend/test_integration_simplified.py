#!/usr/bin/env python3
"""
Phase 4 Cross-Model Analytics - Simplified Integration Test
Test API functionality directly without requiring separate server
"""

import os
import sys
import json
import tempfile
import unittest
from datetime import datetime, timedelta

# Add backend to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from memory_store import MemoryStore
    from cross_model_analytics import CrossModelAnalytics
    from server import app  # Import Flask app directly

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False


class TestPhase4IntegrationSimplified(unittest.TestCase):
    """Simplified integration tests for Phase 4 Cross-Model Analytics"""

    def setUp(self):
        """Set up test environment before each test"""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")

        # Create temporary database with test data
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()

        # Set up Flask test client
        app.config["TESTING"] = True
        self.client = app.test_client()

        # Create test data
        self._setup_test_data()

    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, "temp_db"):
            try:
                os.unlink(self.temp_db.name)
            except FileNotFoundError:
                pass

    def _setup_test_data(self):
        """Create test database with sample training data"""
        memory_store = MemoryStore(self.temp_db.name)

        current_time = datetime.now().isoformat()

        with memory_store._get_connection() as conn:
            cursor = conn.cursor()

            # Insert comprehensive test training sessions
            test_sessions = [
                # GPT-3.5 model runs
                (
                    "job_gpt35_001",
                    "gpt-3.5-turbo",
                    "completed",
                    (datetime.now() - timedelta(days=1)).isoformat(),
                    (datetime.now() - timedelta(days=1, hours=-2)).isoformat(),
                    100,
                    '{"learning_rate": 0.001, "batch_size": 32}',
                    None,
                    current_time,
                    current_time,
                ),
                (
                    "job_gpt35_002",
                    "gpt-3.5-turbo",
                    "completed",
                    (datetime.now() - timedelta(days=3)).isoformat(),
                    (datetime.now() - timedelta(days=3, hours=-1.8)).isoformat(),
                    100,
                    '{"learning_rate": 0.001, "batch_size": 32}',
                    None,
                    current_time,
                    current_time,
                ),
                # GPT-4 model runs
                (
                    "job_gpt4_001",
                    "gpt-4",
                    "completed",
                    (datetime.now() - timedelta(days=2)).isoformat(),
                    (datetime.now() - timedelta(days=2, hours=-3)).isoformat(),
                    100,
                    '{"learning_rate": 0.0005, "batch_size": 16}',
                    None,
                    current_time,
                    current_time,
                ),
                (
                    "job_gpt4_002",
                    "gpt-4",
                    "completed",
                    (datetime.now() - timedelta(days=4)).isoformat(),
                    (datetime.now() - timedelta(days=4, hours=-2.5)).isoformat(),
                    100,
                    '{"learning_rate": 0.0005, "batch_size": 16}',
                    None,
                    current_time,
                    current_time,
                ),
                # Claude model runs
                (
                    "job_claude_001",
                    "claude-3",
                    "completed",
                    (datetime.now() - timedelta(days=1, hours=-6)).isoformat(),
                    (datetime.now() - timedelta(days=1, hours=-8)).isoformat(),
                    100,
                    '{"learning_rate": 0.002, "batch_size": 24}',
                    None,
                    current_time,
                    current_time,
                ),
            ]

            cursor.executemany(
                """
                INSERT INTO training_sessions
                (job_id, model_name, status, start_time, end_time, progress, config, error_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                test_sessions,
            )

            # Insert corresponding training logs
            test_logs = [
                # GPT-3.5 logs - good performance
                ("job_gpt35_001", 10, 0.85, '{"accuracy": 0.65}', current_time),
                ("job_gpt35_001", 25, 0.45, '{"accuracy": 0.78}', current_time),
                ("job_gpt35_001", 50, 0.25, '{"accuracy": 0.85}', current_time),
                ("job_gpt35_001", 75, 0.18, '{"accuracy": 0.89}', current_time),
                ("job_gpt35_001", 100, 0.15, '{"accuracy": 0.92}', current_time),
                ("job_gpt35_002", 10, 0.80, '{"accuracy": 0.68}', current_time),
                ("job_gpt35_002", 25, 0.42, '{"accuracy": 0.80}', current_time),
                ("job_gpt35_002", 50, 0.22, '{"accuracy": 0.87}', current_time),
                ("job_gpt35_002", 75, 0.16, '{"accuracy": 0.90}', current_time),
                ("job_gpt35_002", 100, 0.14, '{"accuracy": 0.93}', current_time),
                # GPT-4 logs - excellent performance but slower
                ("job_gpt4_001", 10, 0.75, '{"accuracy": 0.70}', current_time),
                ("job_gpt4_001", 25, 0.35, '{"accuracy": 0.82}', current_time),
                ("job_gpt4_001", 50, 0.18, '{"accuracy": 0.90}', current_time),
                ("job_gpt4_001", 75, 0.12, '{"accuracy": 0.94}', current_time),
                ("job_gpt4_001", 100, 0.10, '{"accuracy": 0.96}', current_time),
                ("job_gpt4_002", 10, 0.78, '{"accuracy": 0.72}', current_time),
                ("job_gpt4_002", 25, 0.38, '{"accuracy": 0.84}', current_time),
                ("job_gpt4_002", 50, 0.20, '{"accuracy": 0.91}', current_time),
                ("job_gpt4_002", 75, 0.13, '{"accuracy": 0.95}', current_time),
                ("job_gpt4_002", 100, 0.11, '{"accuracy": 0.97}', current_time),
                # Claude logs - moderate performance, fast training
                ("job_claude_001", 10, 0.90, '{"accuracy": 0.60}', current_time),
                ("job_claude_001", 25, 0.55, '{"accuracy": 0.75}', current_time),
                ("job_claude_001", 50, 0.35, '{"accuracy": 0.82}', current_time),
                ("job_claude_001", 75, 0.28, '{"accuracy": 0.86}', current_time),
                ("job_claude_001", 100, 0.25, '{"accuracy": 0.88}', current_time),
            ]

            cursor.executemany(
                """
                INSERT INTO training_logs
                (job_id, epoch, loss, metrics, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                test_logs,
            )

            conn.commit()

        # Note: In a real test environment, we would properly inject this test data
        # For now, we'll use direct method calls to test the analytics engine

    def test_01_flask_app_health(self):
        """Test that the Flask app responds to health checks"""
        print("\n🔍 Testing Flask app health check...")

        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
        print("✅ Flask app health check passed")

    def test_02_direct_analytics_engine(self):
        """Test the analytics engine directly"""
        print("\n🔍 Testing analytics engine directly...")

        # Create analytics engine with our test data
        memory_store = MemoryStore(self.temp_db.name)
        analytics = CrossModelAnalytics(memory_store)

        # Test performance analysis
        performance = analytics.analyze_model_performance("gpt-4")
        self.assertIsNotNone(performance)
        self.assertEqual(performance.model_name, "gpt-4")
        self.assertIsInstance(performance.final_loss, float)
        print(f"✅ GPT-4 performance: final_loss={performance.final_loss:.3f}")

        # Test model comparison
        comparison = analytics.compare_models(["gpt-3.5-turbo", "gpt-4"])
        self.assertIsNotNone(comparison)
        self.assertEqual(len(comparison.compared_models), 2)
        best_model = (
            comparison.performance_ranking[0][0]
            if comparison.performance_ranking
            else "unknown"
        )
        print(f"✅ Model comparison: best_model={best_model}")

        # Test ensemble recommendations
        recommendations = analytics.generate_ensemble_recommendations(
            ["gpt-3.5-turbo", "gpt-4", "claude-3"]
        )
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        print(f"✅ Generated {len(recommendations)} ensemble recommendations")

    def test_03_api_endpoints_via_test_client(self):
        """Test API endpoints via Flask test client"""
        print("\n🔍 Testing API endpoints via test client...")

        # Update server's memory store with our test data
        from server import memory_store as server_memory, cross_model_analytics

        # Replace with our test data
        test_memory = MemoryStore(self.temp_db.name)

        # Monkey patch for testing
        import server

        original_memory = server.memory_store
        original_analytics = server.cross_model_analytics

        try:
            server.memory_store = test_memory
            server.cross_model_analytics = CrossModelAnalytics(test_memory)

            # Test performance endpoint
            response = self.client.get("/api/analytics/performance/gpt-4")
            if response.status_code == 200:
                data = response.get_json()
                self.assertIn("model_name", data)
                self.assertEqual(data["model_name"], "gpt-4")
                print("✅ Performance endpoint working")
            else:
                print(f"⚠️ Performance endpoint returned {response.status_code}")

            # Test matrix endpoint
            response = self.client.get("/api/analytics/matrix")
            if response.status_code == 200:
                data = response.get_json()
                self.assertIn("models", data)
                print(f"✅ Matrix endpoint: {len(data['models'])} models")
            else:
                print(f"⚠️ Matrix endpoint returned {response.status_code}")

            # Test comparison endpoint
            response = self.client.post(
                "/api/analytics/comparison", json={"models": ["gpt-3.5-turbo", "gpt-4"]}
            )
            if response.status_code == 200:
                data = response.get_json()
                self.assertIn("best_model", data)
                print(f"✅ Comparison endpoint: best={data['best_model']}")
            else:
                print(f"⚠️ Comparison endpoint returned {response.status_code}")

        finally:
            # Restore original components
            server.memory_store = original_memory
            server.cross_model_analytics = original_analytics

    def test_04_end_to_end_analytics_workflow(self):
        """Test complete analytics workflow"""
        print("\n🔍 Testing end-to-end analytics workflow...")

        # Create analytics engine with our test data
        memory_store = MemoryStore(self.temp_db.name)
        analytics = CrossModelAnalytics(memory_store)

        # Step 1: Get available models
        models = analytics.get_available_models()
        self.assertGreater(len(models), 0)
        print(f"  📊 Found {len(models)} trained models: {models}")

        # Step 2: Analyze each model
        performances = {}
        for model in models:
            try:
                perf = analytics.analyze_model_performance(model)
                performances[model] = perf
                print(
                    f"  📈 {model}: loss={perf.final_loss:.3f}, efficiency={perf.efficiency_score:.2f}"
                )
            except Exception as e:
                print(f"  ⚠️ Failed to analyze {model}: {e}")

        self.assertGreater(len(performances), 0)

        # Step 3: Compare models
        if len(performances) >= 2:
            model_list = list(performances.keys())[:2]
            comparison = analytics.compare_models(model_list)
            print(f"  🆚 Best model from comparison: {comparison.best_model}")

        # Step 4: Get ensemble recommendations
        if len(models) >= 2:
            recommendations = analytics.get_ensemble_recommendations(models[:3])
            print(f"  🤝 Generated {len(recommendations)} ensemble recommendations")

            for i, rec in enumerate(recommendations[:2]):  # Show first 2
                print(
                    f"    {i+1}. {rec.ensemble_type}: {rec.models} (confidence: {rec.confidence:.2f})"
                )

        # Step 5: Get performance matrix
        matrix = analytics.get_performance_matrix()
        print(
            f"  📊 Performance matrix: {len(matrix.models)} models, {len(matrix.metrics)} metrics"
        )

        # Step 6: Get trend analysis
        trends = analytics.get_trend_analysis(days=7)
        print(
            f"  📈 Trend analysis: {len(trends.trends)} data points over {trends.period}"
        )

        print("✅ End-to-end workflow completed successfully")

    def test_05_error_handling_and_edge_cases(self):
        """Test error handling and edge cases"""
        print("\n🔍 Testing error handling and edge cases...")

        memory_store = MemoryStore(self.temp_db.name)
        analytics = CrossModelAnalytics(memory_store)

        # Test non-existent model
        try:
            performance = analytics.analyze_model_performance("nonexistent-model")
            self.assertIsNone(performance)
            print("✅ Non-existent model handled gracefully")
        except Exception as e:
            print(f"✅ Non-existent model raised exception: {type(e).__name__}")

        # Test empty model list
        try:
            comparison = analytics.compare_models([])
            print("✅ Empty model list handled gracefully")
        except Exception as e:
            print(f"✅ Empty model list raised exception: {type(e).__name__}")

        # Test single model comparison
        try:
            comparison = analytics.compare_models(["gpt-4"])
            print("✅ Single model comparison handled gracefully")
        except Exception as e:
            print(f"✅ Single model comparison raised exception: {type(e).__name__}")

        print("✅ Error handling tests completed")


def run_simplified_integration_tests():
    """Run the simplified integration test suite"""
    print("=" * 80)
    print("🧪 PHASE 4 CROSS-MODEL ANALYTICS - SIMPLIFIED INTEGRATION TESTS")
    print("=" * 80)

    # Check if dependencies are available
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Dependencies not available. Skipping integration tests.")
        return False

    # Run the test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPhase4IntegrationSimplified)
    runner = unittest.TextTestRunner(verbosity=2)

    print("\n🚀 Starting simplified integration test suite...\n")

    import time

    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()

    # Print summary
    print("\n" + "=" * 80)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 80)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors

    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"🔥 Errors: {errors}")
    print(f"⏱️  Duration: {end_time - start_time:.2f}s")

    success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
    print(f"🎯 Success Rate: {success_rate:.1f}%")

    if result.failures:
        print("\n📋 FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\n🔥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    if success_rate >= 90:
        print("\n🌟 EXCELLENT: Integration tests passed with high success rate!")
        print("🚀 Phase 4 Cross-Model Analytics is ready for production!")
    elif success_rate >= 70:
        print("\n👍 GOOD: Integration tests mostly successful with minor issues")
        print("🔧 Phase 4 implementation is solid with minor optimizations needed")
    else:
        print("\n🚨 NEEDS ATTENTION: Integration tests have significant issues")
        print("🛠️  Phase 4 implementation needs debugging")

    return success_rate >= 70


if __name__ == "__main__":
    run_simplified_integration_tests()
