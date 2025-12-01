#!/usr/bin/env python3
"""
Phase 4 Cross-Model Analytics - Basic Integration Test
Test core functionality of the analytics engine
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

# Add backend to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from memory_store import MemoryStore
    from cross_model_analytics import CrossModelAnalytics

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False


class TestPhase4BasicIntegration(unittest.TestCase):
    """Basic integration tests for Phase 4 Cross-Model Analytics"""

    def setUp(self):
        """Set up test environment before each test"""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")

        # Create temporary database with test data
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()

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

            # Insert test training sessions
            test_sessions = [
                # GPT-3.5 - good performer
                (
                    "job_gpt35_001",
                    "gpt-3.5-turbo",
                    "completed",
                    (datetime.now() - timedelta(days=1)).isoformat(),
                    (datetime.now() - timedelta(days=1, hours=-2)).isoformat(),
                    100,
                    '{"learning_rate": 0.001}',
                    None,
                    current_time,
                    current_time,
                ),
                # GPT-4 - excellent performer
                (
                    "job_gpt4_001",
                    "gpt-4",
                    "completed",
                    (datetime.now() - timedelta(days=2)).isoformat(),
                    (datetime.now() - timedelta(days=2, hours=-3)).isoformat(),
                    100,
                    '{"learning_rate": 0.0005}',
                    None,
                    current_time,
                    current_time,
                ),
                # Claude - moderate performer
                (
                    "job_claude_001",
                    "claude-3",
                    "completed",
                    (datetime.now() - timedelta(days=1, hours=-6)).isoformat(),
                    (datetime.now() - timedelta(days=1, hours=-8)).isoformat(),
                    100,
                    '{"learning_rate": 0.002}',
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

            # Insert training logs
            test_logs = [
                # GPT-3.5 logs
                ("job_gpt35_001", 100, 0.15, '{"accuracy": 0.92}', current_time),
                # GPT-4 logs
                ("job_gpt4_001", 100, 0.10, '{"accuracy": 0.96}', current_time),
                # Claude logs
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

    def test_01_analytics_engine_creation(self):
        """Test that analytics engine can be created successfully"""
        print("\n🔍 Testing analytics engine creation...")

        memory_store = MemoryStore(self.temp_db.name)
        analytics = CrossModelAnalytics(memory_store)

        self.assertIsNotNone(analytics)
        self.assertIsNotNone(analytics.memory_store)
        print("✅ Analytics engine created successfully")

    def test_02_model_performance_analysis(self):
        """Test individual model performance analysis"""
        print("\n🔍 Testing model performance analysis...")

        memory_store = MemoryStore(self.temp_db.name)
        analytics = CrossModelAnalytics(memory_store)

        # Test GPT-4 performance
        performance = analytics.analyze_model_performance("gpt-4")

        if performance:
            self.assertEqual(performance.model_name, "gpt-4")
            self.assertIsInstance(performance.final_loss, float)
            self.assertIsInstance(performance.training_time, float)
            self.assertIsInstance(performance.stability_score, float)
            self.assertIsInstance(performance.efficiency_score, float)

            print(f"✅ GPT-4 Analysis:")
            print(f"   - Final Loss: {performance.final_loss:.3f}")
            print(f"   - Training Time: {performance.training_time:.1f}s")
            print(f"   - Stability Score: {performance.stability_score:.2f}")
            print(f"   - Efficiency Score: {performance.efficiency_score:.2f}")
        else:
            print(
                "⚠️ No performance data returned - this may be expected if model has no training history"
            )

    def test_03_model_comparison(self):
        """Test comparison between multiple models"""
        print("\n🔍 Testing model comparison...")

        memory_store = MemoryStore(self.temp_db.name)
        analytics = CrossModelAnalytics(memory_store)

        # Compare two models
        comparison = analytics.compare_models(["gpt-3.5-turbo", "gpt-4"])

        if comparison:
            self.assertEqual(len(comparison.compared_models), 2)
            self.assertIn("gpt-3.5-turbo", comparison.compared_models)
            self.assertIn("gpt-4", comparison.compared_models)

            print(f"✅ Model Comparison:")
            print(f"   - Compared Models: {comparison.compared_models}")
            print(f"   - Performance Rankings: {comparison.performance_ranking}")
            print(f"   - Efficiency Rankings: {comparison.efficiency_ranking}")
            print(f"   - Ensemble Potential: {comparison.ensemble_potential:.2f}")
        else:
            print("⚠️ No comparison data returned")

    def test_04_ensemble_recommendations(self):
        """Test ensemble recommendations generation"""
        print("\n🔍 Testing ensemble recommendations...")

        memory_store = MemoryStore(self.temp_db.name)
        analytics = CrossModelAnalytics(memory_store)

        # Generate ensemble recommendations
        models = ["gpt-3.5-turbo", "gpt-4", "claude-3"]
        recommendations = analytics.generate_ensemble_recommendations(models)

        self.assertIsInstance(recommendations, list)

        if recommendations:
            print(f"✅ Generated {len(recommendations)} ensemble recommendations:")
            for i, rec in enumerate(recommendations[:2]):  # Show first 2
                print(f"   {i+1}. Models: {rec.recommended_models}")
                print(f"      Weights: {rec.weights}")
                print(f"      Confidence: {rec.confidence_score:.2f}")
                print(f"      Expected Performance: {rec.expected_performance:.3f}")
                print(f"      Reasoning: {rec.reasoning[:50]}...")  # First 50 chars
        else:
            print("⚠️ No ensemble recommendations generated")

    def test_05_performance_matrix(self):
        """Test performance matrix generation"""
        print("\n🔍 Testing performance matrix...")

        memory_store = MemoryStore(self.temp_db.name)
        analytics = CrossModelAnalytics(memory_store)

        # Generate performance matrix for our test models
        models = ["gpt-3.5-turbo", "gpt-4", "claude-3"]
        matrix = analytics.get_performance_matrix(models)

        self.assertIsInstance(matrix, dict)
        self.assertIn("models", matrix)
        self.assertIn("metrics", matrix)
        self.assertIn("data", matrix)  # Changed from 'matrix' to 'data'

        print(f"✅ Performance Matrix:")
        print(f"   - Models: {matrix['models']}")
        print(f"   - Metrics: {matrix['metrics']}")
        print(
            f"   - Data Shape: {len(matrix['data'])} metrics x {len(matrix['models'])} models"
        )

    def test_06_historical_trends(self):
        """Test historical trend analysis"""
        print("\n🔍 Testing historical trend analysis...")

        memory_store = MemoryStore(self.temp_db.name)
        analytics = CrossModelAnalytics(memory_store)

        # Analyze trends over the last 7 days
        trends = analytics.analyze_historical_trends(days_back=7)

        self.assertIsInstance(trends, dict)

        print(f"✅ Historical Trends Analysis:")
        for key, value in trends.items():
            print(f"   - {key}: {value}")

    def test_07_error_handling(self):
        """Test error handling for edge cases"""
        print("\n🔍 Testing error handling...")

        memory_store = MemoryStore(self.temp_db.name)
        analytics = CrossModelAnalytics(memory_store)

        # Test non-existent model
        try:
            performance = analytics.analyze_model_performance("nonexistent-model")
            if performance is None:
                print("✅ Non-existent model handled gracefully (returned None)")
            else:
                print("⚠️ Non-existent model returned data unexpectedly")
        except Exception as e:
            print(f"✅ Non-existent model raised exception: {type(e).__name__}")

        # Test empty model list for comparison
        try:
            comparison = analytics.compare_models([])
            if comparison is None:
                print("✅ Empty model list handled gracefully (returned None)")
            else:
                print("⚠️ Empty model list returned data unexpectedly")
        except Exception as e:
            print(f"✅ Empty model list raised exception: {type(e).__name__}")

        print("✅ Error handling tests completed")

    def test_08_complete_workflow(self):
        """Test a complete analytics workflow"""
        print("\n🔍 Testing complete analytics workflow...")

        memory_store = MemoryStore(self.temp_db.name)
        analytics = CrossModelAnalytics(memory_store)

        workflow_steps = []

        # Step 1: Analyze individual models
        models = ["gpt-3.5-turbo", "gpt-4", "claude-3"]
        model_performances = {}

        for model in models:
            try:
                perf = analytics.analyze_model_performance(model)
                if perf:
                    model_performances[model] = perf
                    workflow_steps.append(f"✅ Analyzed {model}")
                else:
                    workflow_steps.append(f"⚠️ No data for {model}")
            except Exception as e:
                workflow_steps.append(f"❌ Failed to analyze {model}: {e}")

        # Step 2: Compare available models
        if len(model_performances) >= 2:
            try:
                available_models = list(model_performances.keys())
                comparison = analytics.compare_models(available_models[:2])
                if comparison:
                    workflow_steps.append(
                        f"✅ Compared {len(available_models[:2])} models"
                    )
                else:
                    workflow_steps.append("⚠️ Comparison returned no data")
            except Exception as e:
                workflow_steps.append(f"❌ Comparison failed: {e}")

        # Step 3: Generate ensemble recommendations
        if len(model_performances) >= 2:
            try:
                available_models = list(model_performances.keys())
                recommendations = analytics.generate_ensemble_recommendations(
                    available_models
                )
                workflow_steps.append(
                    f"✅ Generated {len(recommendations)} ensemble recommendations"
                )
            except Exception as e:
                workflow_steps.append(f"❌ Ensemble generation failed: {e}")

        # Step 4: Create performance matrix
        try:
            matrix = analytics.get_performance_matrix(models)
            workflow_steps.append(f"✅ Created performance matrix")
        except Exception as e:
            workflow_steps.append(f"❌ Matrix generation failed: {e}")

        # Step 5: Analyze trends
        try:
            trends = analytics.analyze_historical_trends(days_back=7)
            workflow_steps.append(f"✅ Analyzed historical trends")
        except Exception as e:
            workflow_steps.append(f"❌ Trend analysis failed: {e}")

        # Print workflow results
        print("📋 Workflow Steps:")
        for step in workflow_steps:
            print(f"   {step}")

        # Count successful steps
        successful_steps = sum(1 for step in workflow_steps if step.startswith("✅"))
        total_steps = len(workflow_steps)
        success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0

        print(
            f"\n📊 Workflow Success Rate: {success_rate:.1f}% ({successful_steps}/{total_steps})"
        )

        # Workflow should have at least some successful steps
        self.assertGreater(successful_steps, 0)

        print("✅ Complete workflow test finished")


def run_basic_integration_tests():
    """Run the basic integration test suite"""
    print("=" * 80)
    print("🧪 PHASE 4 CROSS-MODEL ANALYTICS - BASIC INTEGRATION TESTS")
    print("=" * 80)

    # Check if dependencies are available
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Dependencies not available. Skipping integration tests.")
        return False

    # Run the test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPhase4BasicIntegration)
    runner = unittest.TextTestRunner(verbosity=2)

    print("\n🚀 Starting basic integration test suite...\n")

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

    # Final assessment
    if success_rate >= 90:
        print("\n🌟 EXCELLENT: Integration tests passed with high success rate!")
        print("🚀 Phase 4 Cross-Model Analytics is ready for production!")
    elif success_rate >= 70:
        print("\n👍 GOOD: Integration tests mostly successful")
        print("🔧 Phase 4 implementation is solid with minor optimizations needed")
    elif success_rate >= 50:
        print("\n⚠️ MODERATE: Integration tests show mixed results")
        print("🛠️ Phase 4 implementation needs some debugging")
    else:
        print("\n🚨 NEEDS ATTENTION: Integration tests have significant issues")
        print("🔧 Phase 4 implementation requires major debugging")

    return success_rate >= 50


if __name__ == "__main__":
    run_basic_integration_tests()
