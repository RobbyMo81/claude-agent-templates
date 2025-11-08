#!/usr/bin/env python3
"""
Helios Phase 3 Core Functions Test Suite
========================================

Engineering diagnostic tool to test core functionality of Phase 3 components:
- MemoryStore: Database operations and model metadata management
- MetacognitiveEngine: Self-assessment and performance analysis
- DecisionEngine: Goal management and autonomous decision making

This script validates that the internal logic of our "brains" is working as designed.
"""

import sys
import os
import logging
import traceback
from datetime import datetime
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure logging for test output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test results tracking
test_results = {
    "memory_store": {"passed": 0, "failed": 0, "errors": []},
    "metacognitive_engine": {"passed": 0, "failed": 0, "errors": []},
    "decision_engine": {"passed": 0, "failed": 0, "errors": []},
}


def log_test_result(component, test_name, passed, error=None):
    """Log test result and update tracking"""
    if passed:
        test_results[component]["passed"] += 1
        logger.info(f"✅ {component.upper()} - {test_name}: PASSED")
    else:
        test_results[component]["failed"] += 1
        test_results[component]["errors"].append(f"{test_name}: {error}")
        logger.error(f"❌ {component.upper()} - {test_name}: FAILED - {error}")


def test_memory_store():
    """Test MemoryStore core functions"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING MEMORY STORE CORE FUNCTIONS")
    logger.info("=" * 60)

    try:
        from memory_store import MemoryStore

        # Test 1: Initialize MemoryStore
        try:
            test_db_path = "test_helios_memory.db"
            # Clean up any existing test database
            if os.path.exists(test_db_path):
                os.remove(test_db_path)

            memory_store = MemoryStore(test_db_path)
            log_test_result("memory_store", "Initialization", True)
        except Exception as e:
            log_test_result("memory_store", "Initialization", False, str(e))
            return

        # Test 2: Create all tables
        try:
            memory_store.create_all_tables()
            log_test_result("memory_store", "create_all_tables()", True)
        except Exception as e:
            log_test_result("memory_store", "create_all_tables()", False, str(e))
            return

        # Test 3: Save model metadata
        try:
            mock_metadata = {
                "name": "test_model_v1",
                "architecture": "neural_network",
                "version": "1.0.0",
                "file_path": "/models/test_model_v1.pkl",
                "metadata": {
                    "training_completed": True,
                    "total_epochs": 100,
                    "best_loss": 0.25,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                },
            }

            result = memory_store.save_model_metadata(**mock_metadata)
            if result:
                log_test_result("memory_store", "save_model_metadata()", True)
            else:
                log_test_result(
                    "memory_store",
                    "save_model_metadata()",
                    False,
                    "Function returned False",
                )
        except Exception as e:
            log_test_result("memory_store", "save_model_metadata()", False, str(e))

        # Test 4: Retrieve model metadata
        try:
            retrieved = memory_store.get_model_metadata("test_model_v1")
            if retrieved and retrieved["name"] == "test_model_v1":
                log_test_result("memory_store", "get_model_metadata()", True)
            else:
                log_test_result(
                    "memory_store",
                    "get_model_metadata()",
                    False,
                    "Retrieved data mismatch",
                )
        except Exception as e:
            log_test_result("memory_store", "get_model_metadata()", False, str(e))

        # Test 5: List models
        try:
            models = memory_store.list_models(active_only=True)
            if isinstance(models, list) and len(models) > 0:
                log_test_result("memory_store", "list_models()", True)
            else:
                log_test_result(
                    "memory_store",
                    "list_models()",
                    False,
                    f"Expected list with data, got {type(models)} with {len(models) if hasattr(models, '__len__') else 0} items",
                )
        except Exception as e:
            log_test_result("memory_store", "list_models()", False, str(e))

        # Test 6: Save prediction
        try:
            mock_prediction = {
                "white_balls": [5, 12, 23, 34, 45],
                "red_ball": 18,
                "confidence": [0.75, 0.68, 0.82, 0.71, 0.79, 0.85],
            }

            result = memory_store.save_prediction(
                model_name="test_model_v1",
                prediction_data=mock_prediction,
                confidence=0.76,
            )
            if result:
                log_test_result("memory_store", "save_prediction()", True)
            else:
                log_test_result(
                    "memory_store",
                    "save_prediction()",
                    False,
                    "Function returned False",
                )
        except Exception as e:
            log_test_result("memory_store", "save_prediction()", False, str(e))

        # Cleanup test database
        try:
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
                logger.info("🧹 Test database cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up test database: {e}")

    except ImportError as e:
        log_test_result(
            "memory_store", "Import", False, f"Could not import MemoryStore: {e}"
        )
    except Exception as e:
        log_test_result("memory_store", "General", False, f"Unexpected error: {e}")


def test_metacognitive_engine():
    """Test MetacognitiveEngine core functions"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING METACOGNITIVE ENGINE CORE FUNCTIONS")
    logger.info("=" * 60)

    try:
        from memory_store import MemoryStore
        from metacognition import MetacognitiveEngine

        # Test 1: Initialize with MemoryStore
        try:
            test_db_path = "test_metacog_memory.db"
            if os.path.exists(test_db_path):
                os.remove(test_db_path)

            memory_store = MemoryStore(test_db_path)
            metacog_engine = MetacognitiveEngine(memory_store)
            log_test_result("metacognitive_engine", "Initialization", True)
        except Exception as e:
            log_test_result("metacognitive_engine", "Initialization", False, str(e))
            return

        # Test 2: Assess current state
        try:
            mock_metrics = {
                "accuracy": 0.78,
                "loss": 0.25,
                "precision": 0.82,
                "recall": 0.75,
            }

            mock_performance = [
                {"epoch": 90, "loss": 0.28, "accuracy": 0.76},
                {"epoch": 95, "loss": 0.26, "accuracy": 0.77},
                {"epoch": 100, "loss": 0.25, "accuracy": 0.78},
            ]

            mock_context = {
                "training_phase": "final",
                "model_type": "neural_network",
                "dataset_size": 10000,
            }

            assessment = metacog_engine.assess_current_state(
                model_name="test_model",
                current_metrics=mock_metrics,
                recent_performance=mock_performance,
                context=mock_context,
            )

            # Validate assessment structure
            required_fields = [
                "confidence_score",
                "predicted_performance",
                "uncertainty_estimate",
                "knowledge_gaps",
                "recommended_strategy",
                "assessment_timestamp",
            ]

            missing_fields = [
                field for field in required_fields if not hasattr(assessment, field)
            ]

            if not missing_fields:
                log_test_result("metacognitive_engine", "assess_current_state()", True)
            else:
                log_test_result(
                    "metacognitive_engine",
                    "assess_current_state()",
                    False,
                    f"Missing fields: {missing_fields}",
                )

        except Exception as e:
            log_test_result(
                "metacognitive_engine", "assess_current_state()", False, str(e)
            )

        # Test 3: Analyze performance patterns
        try:
            patterns = metacog_engine.analyze_performance_patterns("test_model", days=7)

            if isinstance(patterns, list):
                log_test_result(
                    "metacognitive_engine", "analyze_performance_patterns()", True
                )
            else:
                log_test_result(
                    "metacognitive_engine",
                    "analyze_performance_patterns()",
                    False,
                    f"Expected list, got {type(patterns)}",
                )
        except Exception as e:
            log_test_result(
                "metacognitive_engine", "analyze_performance_patterns()", False, str(e)
            )

        # Test 4: Get learning recommendations
        try:
            # First create a mock assessment
            from metacognition import MetacognitiveAssessment, LearningStrategy

            mock_assessment = MetacognitiveAssessment(
                confidence_score=0.85,
                predicted_performance=0.80,
                uncertainty_estimate=0.15,
                knowledge_gaps=["pattern_recognition", "edge_cases"],
                recommended_strategy=LearningStrategy.ACTIVE_LEARNING,
                assessment_timestamp=datetime.now(),
                context=mock_context,
            )

            recommendations = metacog_engine.get_learning_recommendations(
                "test_model", mock_assessment
            )

            if isinstance(recommendations, dict):
                log_test_result(
                    "metacognitive_engine", "get_learning_recommendations()", True
                )
            else:
                log_test_result(
                    "metacognitive_engine",
                    "get_learning_recommendations()",
                    False,
                    f"Expected dict, got {type(recommendations)}",
                )
        except Exception as e:
            log_test_result(
                "metacognitive_engine", "get_learning_recommendations()", False, str(e)
            )

        # Cleanup
        try:
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
        except Exception:
            pass

    except ImportError as e:
        log_test_result(
            "metacognitive_engine", "Import", False, f"Could not import components: {e}"
        )
    except Exception as e:
        log_test_result(
            "metacognitive_engine", "General", False, f"Unexpected error: {e}"
        )


def test_decision_engine():
    """Test DecisionEngine core functions"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING DECISION ENGINE CORE FUNCTIONS")
    logger.info("=" * 60)

    try:
        from memory_store import MemoryStore
        from metacognition import MetacognitiveEngine
        from decision_engine import DecisionEngine, Goal

        # Test 1: Initialize with MemoryStore and MetacognitiveEngine
        try:
            test_db_path = "test_decision_memory.db"
            if os.path.exists(test_db_path):
                os.remove(test_db_path)

            memory_store = MemoryStore(test_db_path)
            metacog_engine = MetacognitiveEngine(memory_store)
            decision_engine = DecisionEngine(memory_store, metacog_engine)
            log_test_result("decision_engine", "Initialization", True)
        except Exception as e:
            log_test_result("decision_engine", "Initialization", False, str(e))
            return

        # Test 2: Add goal
        try:
            mock_goal = Goal(
                goal_id="test_goal_001",
                name="Improve Model Accuracy",
                target_metric="accuracy",
                target_value=0.90,
                current_value=0.78,
                priority=1,
                deadline=datetime(2025, 8, 1),
                dependencies=[],
            )

            result = decision_engine.add_goal(mock_goal)
            if result:
                log_test_result("decision_engine", "add_goal()", True)
            else:
                log_test_result(
                    "decision_engine", "add_goal()", False, "Function returned False"
                )
        except Exception as e:
            log_test_result("decision_engine", "add_goal()", False, str(e))

        # Test 3: Make autonomous decision
        try:
            mock_metrics = {
                "accuracy": 0.78,
                "loss": 0.25,
                "training_time": 3600,
                "convergence_rate": 0.85,
            }

            mock_performance = [
                {"epoch": 95, "loss": 0.26, "accuracy": 0.77},
                {"epoch": 100, "loss": 0.25, "accuracy": 0.78},
            ]

            mock_context = {
                "training_phase": "optimization",
                "resource_usage": "moderate",
                "time_constraints": "flexible",
            }

            decisions = decision_engine.make_autonomous_decision(
                model_name="test_model",
                current_metrics=mock_metrics,
                recent_performance=mock_performance,
                context=mock_context,
            )

            if isinstance(decisions, list):
                log_test_result("decision_engine", "make_autonomous_decision()", True)
            else:
                log_test_result(
                    "decision_engine",
                    "make_autonomous_decision()",
                    False,
                    f"Expected list, got {type(decisions)}",
                )
        except Exception as e:
            log_test_result(
                "decision_engine", "make_autonomous_decision()", False, str(e)
            )

        # Test 4: Get system status
        try:
            status = decision_engine.get_system_status()

            if isinstance(status, dict):
                log_test_result("decision_engine", "get_system_status()", True)
            else:
                log_test_result(
                    "decision_engine",
                    "get_system_status()",
                    False,
                    f"Expected dict, got {type(status)}",
                )
        except Exception as e:
            log_test_result("decision_engine", "get_system_status()", False, str(e))

        # Test 5: Get goal status
        try:
            goal_status = decision_engine.get_goal_status()

            if isinstance(goal_status, dict):
                log_test_result("decision_engine", "get_goal_status()", True)
            else:
                log_test_result(
                    "decision_engine",
                    "get_goal_status()",
                    False,
                    f"Expected dict, got {type(goal_status)}",
                )
        except Exception as e:
            log_test_result("decision_engine", "get_goal_status()", False, str(e))

        # Test 6: Start/Stop autonomous mode
        try:
            decision_engine.start_autonomous_mode()
            decision_engine.stop_autonomous_mode()
            log_test_result("decision_engine", "autonomous_mode_control()", True)
        except Exception as e:
            log_test_result(
                "decision_engine", "autonomous_mode_control()", False, str(e)
            )

        # Cleanup
        try:
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
        except Exception:
            pass

    except ImportError as e:
        log_test_result(
            "decision_engine", "Import", False, f"Could not import components: {e}"
        )
    except Exception as e:
        log_test_result("decision_engine", "General", False, f"Unexpected error: {e}")


def print_test_summary():
    """Print comprehensive test summary"""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 CORE FUNCTIONS TEST SUMMARY")
    logger.info("=" * 60)

    total_passed = 0
    total_failed = 0

    for component, results in test_results.items():
        passed = results["passed"]
        failed = results["failed"]
        total_passed += passed
        total_failed += failed

        status_icon = "✅" if failed == 0 else "⚠️" if passed > failed else "❌"
        logger.info(
            f"{status_icon} {component.upper()}: {passed} passed, {failed} failed"
        )

        if results["errors"]:
            for error in results["errors"]:
                logger.info(f"   • {error}")

    logger.info("-" * 60)
    logger.info(f"TOTAL RESULTS: {total_passed} passed, {total_failed} failed")

    if total_failed == 0:
        logger.info("🎉 ALL TESTS PASSED! Phase 3 components are fully functional.")
    elif total_passed > total_failed:
        logger.info(
            "⚠️  MOSTLY FUNCTIONAL: Some issues detected but core functionality works."
        )
    else:
        logger.info("❌ SIGNIFICANT ISSUES: Phase 3 components need attention.")

    return total_failed == 0


def main():
    """Main test execution"""
    logger.info("🔬 HELIOS PHASE 3 ENGINEERING DIAGNOSTIC TOOL")
    logger.info("=" * 60)
    logger.info("Testing core functionality of Phase 3 'brains':")
    logger.info("• MemoryStore: Database operations")
    logger.info("• MetacognitiveEngine: Self-assessment")
    logger.info("• DecisionEngine: Autonomous decision making")
    logger.info("=" * 60)

    try:
        # Run all component tests
        test_memory_store()
        test_metacognitive_engine()
        test_decision_engine()

        # Print comprehensive summary
        all_passed = print_test_summary()

        # Exit with appropriate code
        sys.exit(0 if all_passed else 1)

    except Exception as e:
        logger.error(f"Critical test framework error: {e}")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
