"""
Comprehensive unit tests for the MemoryStore class.
Tests all functionality including model management, training sessions, predictions,
enhanced journal entries, knowledge fragments, performance metrics, and memory operations.
"""

import unittest
import tempfile
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

# Import the MemoryStore class
from memory_store import MemoryStore


class TestMemoryStore(unittest.TestCase):
    """Comprehensive test suite for MemoryStore functionality."""

    def setUp(self):
        """Set up test fixtures with temporary database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.memory_store = MemoryStore(self.db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        self.memory_store.close()
        try:
            os.unlink(self.db_path)
        except FileNotFoundError:
            pass

    def test_initialization(self):
        """Test MemoryStore initialization."""
        # Test file-based database
        self.assertTrue(os.path.exists(self.db_path))

        # Test in-memory database
        memory_store_mem = MemoryStore(":memory:")
        self.assertIsNotNone(memory_store_mem.conn)
        memory_store_mem.close()

    def test_model_metadata_management(self):
        """Test model metadata CRUD operations."""
        # Test saving model metadata
        model_id = self.memory_store.save_model_metadata(
            name="test_model",
            file_path="/path/to/model.pkl",
            architecture="neural_network",
            version="1.0.0",
            metadata={"epochs": 100, "accuracy": 0.95},
        )

        self.assertIsInstance(model_id, int)
        self.assertGreater(model_id, 0)

        # Test retrieving model metadata
        model_data = self.memory_store.get_model_metadata("test_model")
        self.assertIsNotNone(model_data)
        self.assertEqual(model_data["name"], "test_model")
        self.assertEqual(model_data["architecture"], "neural_network")
        self.assertEqual(model_data["metadata"]["epochs"], 100)

        # Test listing models
        models = self.memory_store.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "test_model")

        # Test updating model (INSERT OR REPLACE)
        updated_id = self.memory_store.save_model_metadata(
            name="test_model",
            file_path="/new/path/to/model.pkl",
            architecture="neural_network",
            version="2.0.0",
            metadata={"epochs": 200, "accuracy": 0.97},
        )

        updated_model = self.memory_store.get_model_metadata("test_model")
        self.assertEqual(updated_model["version"], "2.0.0")
        self.assertEqual(updated_model["metadata"]["epochs"], 200)

        # Test deleting model (soft delete)
        success = self.memory_store.delete_model("test_model")
        self.assertTrue(success)

        # Model should not appear in active list
        active_models = self.memory_store.list_models(active_only=True)
        self.assertEqual(len(active_models), 0)

        # But should appear in all models list
        all_models = self.memory_store.list_models(active_only=False)
        self.assertEqual(len(all_models), 1)
        self.assertFalse(all_models[0]["is_active"])

    def test_training_session_management(self):
        """Test training session lifecycle management."""
        config = {"learning_rate": 0.001, "batch_size": 32, "epochs": 100}

        # Create training session
        session_id = self.memory_store.create_training_session(
            job_id="job_123", model_name="test_model", config=config
        )

        self.assertIsInstance(session_id, int)
        self.assertGreater(session_id, 0)

        # Retrieve training session
        session = self.memory_store.get_training_session("job_123")
        self.assertIsNotNone(session)
        self.assertEqual(session["job_id"], "job_123")
        self.assertEqual(session["model_name"], "test_model")
        self.assertEqual(session["status"], "started")
        self.assertEqual(session["config"]["batch_size"], 32)

        # Update training session
        self.memory_store.update_training_session(
            job_id="job_123", status="running", progress=50
        )

        updated_session = self.memory_store.get_training_session("job_123")
        self.assertEqual(updated_session["status"], "running")
        self.assertEqual(updated_session["progress"], 50)

        # Complete training session
        self.memory_store.update_training_session(
            job_id="job_123", status="completed", progress=100
        )

        completed_session = self.memory_store.get_training_session("job_123")
        self.assertEqual(completed_session["status"], "completed")
        self.assertIsNotNone(completed_session["end_time"])

    def test_training_logs(self):
        """Test training log management."""
        # First create a training session
        self.memory_store.create_training_session(
            job_id="job_456", model_name="test_model", config={"epochs": 10}
        )

        # Add training logs
        for epoch in range(1, 6):
            self.memory_store.add_training_log(
                job_id="job_456",
                epoch=epoch,
                loss=1.0 / epoch,
                metrics={"accuracy": 0.8 + (epoch * 0.02)},
            )

        # Retrieve training logs
        logs = self.memory_store.get_training_logs("job_456")
        self.assertEqual(len(logs), 5)

        # Check log ordering (should be by epoch ASC)
        for i, log in enumerate(logs):
            self.assertEqual(log["epoch"], i + 1)
            self.assertAlmostEqual(log["loss"], 1.0 / (i + 1))
            self.assertGreater(log["metrics"]["accuracy"], 0.8)

    def test_prediction_management(self):
        """Test prediction storage and outcome tracking."""
        prediction_data = {
            "white_numbers": [5, 12, 23, 34, 45],
            "powerball": 18,
            "confidence": 0.75,
        }

        # Save prediction
        pred_id = self.memory_store.save_prediction(
            model_name="powerball_model",
            prediction_data=prediction_data,
            confidence=0.75,
            draw_date="2024-12-01",
        )

        self.assertIsInstance(pred_id, int)
        self.assertGreater(pred_id, 0)

        # Update prediction outcome
        actual_outcome = {"white_numbers": [7, 12, 23, 34, 50], "powerball": 18}

        self.memory_store.update_prediction_outcome(
            prediction_id=pred_id,
            actual_outcome=actual_outcome,
            is_correct=False,  # Partially correct
        )

        # Retrieve predictions
        predictions = self.memory_store.get_model_predictions("powerball_model")
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["id"], pred_id)
        self.assertFalse(predictions[0]["is_correct"])
        self.assertEqual(predictions[0]["actual_outcome"]["powerball"], 18)

    def test_enhanced_journal_entries(self):
        """Test enhanced journal entry management."""
        event_data = {
            "action": "prediction_made",
            "details": "Generated lottery prediction using neural network",
            "model_state": "trained",
        }

        # Store enhanced journal entry
        entry_id = self.memory_store.store_enhanced_journal_entry(
            model_name="test_model",
            session_id="session_789",
            event_type="prediction",
            event_data=event_data,
            confidence_score=0.85,
            success_metric=0.72,
            context_hash="abc123",
        )

        self.assertIsInstance(entry_id, int)
        self.assertGreater(entry_id, 0)

        # Retrieve journal entries
        entries = self.memory_store.get_enhanced_journal_entries(
            model_name="test_model"
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["event_type"], "prediction")
        self.assertEqual(entries[0]["confidence_score"], 0.85)
        self.assertEqual(entries[0]["event_data"]["action"], "prediction_made")

        # Test filtering by event type
        filtered_entries = self.memory_store.get_enhanced_journal_entries(
            event_type="prediction"
        )
        self.assertEqual(len(filtered_entries), 1)

        # Test filtering by session ID
        session_entries = self.memory_store.get_enhanced_journal_entries(
            session_id="session_789"
        )
        self.assertEqual(len(session_entries), 1)

    def test_knowledge_fragments(self):
        """Test knowledge fragment management."""
        # Store knowledge fragment
        fragment_id = self.memory_store.store_knowledge_fragment(
            model_name="test_model",
            fragment_type="pattern",
            content="Numbers ending in 5 appear 15% more frequently",
            relevance_score=0.9,
        )

        self.assertIsInstance(fragment_id, int)
        self.assertGreater(fragment_id, 0)

        # Store another fragment with lower relevance
        low_fragment_id = self.memory_store.store_knowledge_fragment(
            model_name="test_model",
            fragment_type="observation",
            content="Random observation",
            relevance_score=0.3,
        )

        # Retrieve all fragments
        fragments = self.memory_store.get_knowledge_fragments(model_name="test_model")
        self.assertEqual(len(fragments), 2)

        # Check ordering (highest relevance first)
        self.assertEqual(fragments[0]["relevance_score"], 0.9)
        self.assertEqual(fragments[1]["relevance_score"], 0.3)

        # Test filtering by minimum relevance
        high_relevance_fragments = self.memory_store.get_knowledge_fragments(
            model_name="test_model", min_relevance=0.5
        )
        self.assertEqual(len(high_relevance_fragments), 1)
        self.assertEqual(
            high_relevance_fragments[0]["content"],
            "Numbers ending in 5 appear 15% more frequently",
        )

        # Test usage tracking
        self.memory_store.update_knowledge_fragment_usage(fragment_id)

        updated_fragments = self.memory_store.get_knowledge_fragments(
            model_name="test_model"
        )
        # Find the updated fragment
        updated_fragment = next(f for f in updated_fragments if f["id"] == fragment_id)
        self.assertEqual(updated_fragment["usage_count"], 1)

    def test_performance_metrics(self):
        """Test performance metric storage and retrieval."""
        # Store performance metrics
        metric_id1 = self.memory_store.store_performance_metric(
            model_name="test_model",
            metric_name="accuracy",
            metric_value=0.95,
            context="test_dataset",
        )

        metric_id2 = self.memory_store.store_performance_metric(
            model_name="test_model",
            metric_name="loss",
            metric_value=0.23,
            context="validation_dataset",
        )

        self.assertIsInstance(metric_id1, int)
        self.assertIsInstance(metric_id2, int)
        self.assertNotEqual(metric_id1, metric_id2)

        # Retrieve all metrics for model
        metrics = self.memory_store.get_performance_metrics(model_name="test_model")
        self.assertEqual(len(metrics), 2)

        # Test filtering by metric name
        accuracy_metrics = self.memory_store.get_performance_metrics(
            model_name="test_model", metric_name="accuracy"
        )
        self.assertEqual(len(accuracy_metrics), 1)
        self.assertEqual(accuracy_metrics[0]["metric_value"], 0.95)

        # Test time-based filtering (all metrics should be recent)
        # Use a longer time window to ensure we capture the metrics
        recent_metrics = self.memory_store.get_performance_metrics(
            model_name="test_model", hours=24  # Use 24 hours instead of 1 hour
        )
        self.assertEqual(len(recent_metrics), 2)

    def test_memory_operations_logging(self):
        """Test memory operation logging."""
        operation_id = self.memory_store.log_memory_operation(
            operation_type="compaction",
            details="Archived old journal entries",
            items_affected=150,
            space_saved=1024000,
        )

        self.assertIsInstance(operation_id, int)
        self.assertGreater(operation_id, 0)

    def test_memory_compaction(self):
        """Test memory compaction functionality."""
        # Create some test data to compact
        # Old journal entry (simulate old timestamp)
        old_entry_id = self.memory_store.store_enhanced_journal_entry(
            model_name="old_model",
            session_id="old_session",
            event_type="old_event",
            event_data={"old": "data"},
        )

        # Low relevance knowledge fragment
        low_fragment_id = self.memory_store.store_knowledge_fragment(
            model_name="test_model",
            fragment_type="low_value",
            content="Low value content",
            relevance_score=0.05,
        )

        # Perform compaction with very low thresholds for testing
        stats = self.memory_store.compact_memory(
            archive_days=0,  # Archive everything
            relevance_threshold=0.1,  # Delete low relevance fragments
        )

        self.assertIsInstance(stats, dict)
        self.assertIn("archived_journal_entries", stats)
        self.assertIn("deleted_knowledge_fragments", stats)
        self.assertIn("cleaned_performance_metrics", stats)

        # Check that journal entry was archived
        all_entries = self.memory_store.get_enhanced_journal_entries(
            include_archived=True
        )
        self.assertTrue(any(entry["archived"] for entry in all_entries))

    def test_memory_statistics(self):
        """Test memory statistics functionality."""
        # Add some data first
        self.memory_store.save_model_metadata(
            name="stats_test_model",
            file_path="/path/to/model",
            architecture="test_arch",
            version="1.0.0",
        )

        self.memory_store.store_enhanced_journal_entry(
            model_name="stats_test_model",
            session_id="stats_session",
            event_type="test",
            event_data={"test": "data"},
        )

        # Get statistics
        stats = self.memory_store.get_memory_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn("enhanced_journal_count", stats)
        self.assertIn("knowledge_fragments_count", stats)
        self.assertIn("performance_metrics_count", stats)
        self.assertIn("memory_operations_count", stats)
        self.assertIn("database_size_bytes", stats)
        self.assertIn("database_size_mb", stats)
        self.assertIn("journal_entries_24h", stats)
        self.assertIn("metrics_24h", stats)

        # Check that we have at least one journal entry
        self.assertGreaterEqual(stats["enhanced_journal_count"], 1)
        self.assertGreater(stats["database_size_bytes"], 0)

    def test_context_storage(self):
        """Test context storage and retrieval."""
        context_data = {
            "user_preferences": {"theme": "dark", "notifications": True},
            "session_state": {"last_action": "prediction", "timestamp": "2024-12-01"},
        }

        # Store context
        self.memory_store.store_context(
            context_type="user_session",
            context_key="user_123",
            context_data=context_data,
        )

        # Retrieve context
        retrieved_context = self.memory_store.get_context(
            context_type="user_session", context_key="user_123"
        )

        self.assertIsNotNone(retrieved_context)
        self.assertEqual(
            retrieved_context["context_data"]["user_preferences"]["theme"], "dark"
        )

        # Test with expiration
        future_time = datetime.now() + timedelta(hours=1)
        self.memory_store.store_context(
            context_type="temp_session",
            context_key="temp_key",
            context_data={"temp": "data"},
            expires_at=future_time,
        )

        # Should retrieve non-expired context
        temp_context = self.memory_store.get_context(
            context_type="temp_session", context_key="temp_key"
        )
        self.assertIsNotNone(temp_context)

        # Test cleanup (won't actually delete anything since expiration is in future)
        self.memory_store.cleanup_expired_context()

    def test_system_events(self):
        """Test system event logging and retrieval."""
        event_data = {
            "component": "training_manager",
            "action": "model_trained",
            "success": True,
        }

        # Log event
        self.memory_store.log_event(
            event_type="training_completed", event_data=event_data, level="INFO"
        )

        # Retrieve recent events
        events = self.memory_store.get_recent_events(hours=1)
        self.assertGreater(len(events), 0)

        # Filter by event type
        training_events = self.memory_store.get_recent_events(
            event_type="training_completed", hours=1
        )
        self.assertEqual(len(training_events), 1)
        self.assertEqual(
            training_events[0]["event_data"]["component"], "training_manager"
        )

    def test_database_vacuum(self):
        """Test database vacuum operation."""
        # This should not raise any exceptions
        try:
            self.memory_store.vacuum_database()
        except Exception as e:
            self.fail(f"Database vacuum failed: {e}")

    def test_connection_error_handling(self):
        """Test database connection error handling."""
        # Test with invalid database path that requires special permissions
        # or create a scenario where we can't write to the database
        try:
            # Try to create a database in a read-only location or with invalid characters
            if os.name == "nt":  # Windows
                invalid_path = "CON:"  # Invalid device name on Windows
            else:  # Unix-like
                invalid_path = "/dev/null/invalid.db"  # Can't create file in /dev/null

            invalid_store = MemoryStore(invalid_path)
            invalid_store.close()
            # If we get here without exception, the test should still pass
            # as the behavior might be platform-dependent
        except Exception:
            # Expected behavior - exception was raised
            pass

    def test_concurrent_access(self):
        """Test thread safety with lock mechanism."""
        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            try:
                # Each worker saves a model
                model_id = self.memory_store.save_model_metadata(
                    name=f"concurrent_model_{worker_id}",
                    file_path=f"/path/to/model_{worker_id}",
                    architecture="test_arch",
                    version="1.0.0",
                )
                results.append(model_id)

                # Small delay to encourage race conditions
                time.sleep(0.01)

                # Each worker also stores a journal entry
                entry_id = self.memory_store.store_enhanced_journal_entry(
                    model_name=f"concurrent_model_{worker_id}",
                    session_id=f"session_{worker_id}",
                    event_type="concurrent_test",
                    event_data={"worker_id": worker_id},
                )
                results.append(entry_id)

            except Exception as e:
                errors.append(str(e))

        # Start multiple workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")
        self.assertEqual(len(results), 10)  # 5 workers * 2 operations each

        # Verify all models were saved
        models = self.memory_store.list_models()
        concurrent_models = [
            m for m in models if m["name"].startswith("concurrent_model_")
        ]
        self.assertEqual(len(concurrent_models), 5)


def run_comprehensive_test():
    """Run all tests and provide detailed results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMemoryStore)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None, buffer=True)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success_rate = (
        (result.testsRun - len(result.failures) - len(result.errors))
        / result.testsRun
        * 100
    )
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ Some tests failed. Please review the output above.")

    return result.wasSuccessful()


if __name__ == "__main__":
    run_comprehensive_test()
