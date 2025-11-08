# backend/tests/test_memory_store.py

import pytest
from memory_store import MemoryStore
import os
from datetime import datetime

# A pytest "fixture" that creates a fresh, in-memory database for each test
@pytest.fixture
def memory_store():
    """
    This fixture sets up a fresh, in-memory SQLite database for each test function.
    Using ":memory:" is fast and ensures that tests are isolated from each other
    and from the production database file.
    """
    store = MemoryStore(db_path=":memory:")
    yield store
    # Clean up the connection after the test is done
    store.close()

def test_initialization(memory_store):
    """
    Tests that the database is initialized correctly with all expected tables.
    """
    with memory_store._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'enhanced_journal', 'knowledge_fragments', 'performance_metrics',
            'memory_operations', 'models', 'training_sessions', 'training_logs',
            'predictions', 'context_storage', 'system_events'
        ]
        
        for table in expected_tables:
            assert table in tables

def test_save_and_get_model_metadata(memory_store):
    """
    Tests that saving and retrieving model metadata works correctly.
    """
    # Arrange: Define the data we want to save
    model_name = "test_model_1"
    metadata = {"param_a": 10, "param_b": "hello"}
    timestamp_before = datetime.now().isoformat()

    # Act: Save the model metadata
    memory_store.save_model_metadata(
        name=model_name,
        file_path=f"models/{model_name}.pth",
        architecture="TestNet",
        version="1.0.1",
        metadata=metadata
    )

    # Assert: Retrieve the data and check if it's correct
    retrieved = memory_store.get_model_metadata(model_name)

    assert retrieved is not None
    assert retrieved["name"] == model_name
    assert retrieved["architecture"] == "TestNet"
    assert retrieved["version"] == "1.0.1"
    assert retrieved["metadata"]["param_a"] == 10
    assert retrieved["metadata"]["param_b"] == "hello"
    assert retrieved["created_at"] >= timestamp_before
    assert retrieved["updated_at"] >= timestamp_before
    assert retrieved["is_active"] == 1

def test_get_model_metadata_not_found(memory_store):
    """
    Tests that getting metadata for a non-existent model returns None.
    """
    retrieved = memory_store.get_model_metadata("non_existent_model")
    assert retrieved is None

def test_list_models(memory_store):
    """
    Tests that listing models works as expected.
    """
    # Arrange: Add multiple models
    memory_store.save_model_metadata(name="model_a", file_path="a.pth", architecture="NetA")
    memory_store.save_model_metadata(name="model_b", file_path="b.pth", architecture="NetB")
    
    # Act: List active models
    models = memory_store.list_models(active_only=True)
    
    # Assert
    assert len(models) == 2
    assert {m["name"] for m in models} == {"model_a", "model_b"}

def test_delete_model(memory_store):
    """
    Tests that deleting a model marks it as inactive and it's no longer listed by default.
    """
    # Arrange: Add a model to the store
    model_name = "model_to_delete"
    memory_store.save_model_metadata(
        name=model_name,
        file_path=f"models/{model_name}.pth",
        architecture="TestNet"
    )

    # Act: Delete the model
    success = memory_store.delete_model(model_name)
    assert success is True

    # Assert: The model should not be found when querying for active models
    retrieved_active = memory_store.get_model_metadata(model_name)
    assert retrieved_active is None
    
    active_list = memory_store.list_models(active_only=True)
    assert len(active_list) == 0

    # Assert: The model should be found if we query for all models (including inactive)
    all_models = memory_store.list_models(active_only=False)
    assert len(all_models) == 1
    deleted_model_data = all_models[0]
    assert deleted_model_data["name"] == model_name
    assert deleted_model_data["is_active"] == 0

def test_create_and_get_training_session(memory_store):
    """
    Tests the creation and retrieval of a training session.
    """
    # Arrange
    job_id = "job_123"
    model_name = "training_model"
    config = {"epochs": 10, "lr": 0.01}
    
    # Act
    memory_store.create_training_session(job_id, model_name, config)
    session = memory_store.get_training_session(job_id)
    
    # Assert
    assert session is not None
    assert session["job_id"] == job_id
    assert session["model_name"] == model_name
    assert session["status"] == "started"
    assert session["config"]["epochs"] == 10

def test_update_training_session(memory_store):
    """
    Tests updating a training session's status and progress.
    """
    # Arrange
    job_id = "job_456"
    memory_store.create_training_session(job_id, "model", {})
    
    # Act
    memory_store.update_training_session(job_id, status="completed", progress=100, error_message="None")
    session = memory_store.get_training_session(job_id)
    
    # Assert
    assert session["status"] == "completed"
    assert session["progress"] == 100
    assert session["error_message"] == "None"
    assert session["end_time"] is not None

def test_add_and_get_training_logs(memory_store):
    """
    Tests adding and retrieving training logs for a session.
    """
    # Arrange
    job_id = "job_789"
    memory_store.create_training_session(job_id, "log_model", {})
    
    # Act
    memory_store.add_training_log(job_id, epoch=1, loss=0.5, metrics={"accuracy": 0.9})
    memory_store.add_training_log(job_id, epoch=2, loss=0.4)
    logs = memory_store.get_training_logs(job_id)
    
    # Assert
    assert len(logs) == 2
    assert logs[0]["epoch"] == 1
    assert logs[0]["loss"] == 0.5
    assert logs[0]["metrics"]["accuracy"] == 0.9
    assert logs[1]["epoch"] == 2
    assert logs[1]["loss"] == 0.4
