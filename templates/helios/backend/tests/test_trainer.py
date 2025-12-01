# backend/tests/test_trainer.py

import pytest
import torch
import pandas as pd
from unittest.mock import MagicMock, patch

from trainer import Trainer, TrainingConfig, DataPreprocessor
from agent import MLPowerballAgent


@pytest.fixture
def mock_trainer(mocker):
    """
    Fixture to create a Trainer with a mocked MLPowerballAgent.
    """
    # Mock the agent that the trainer would use
    mock_agent_instance = MagicMock(spec=MLPowerballAgent)

    # The trainer accesses agent.model.parameters(), so we need to mock this nested structure.
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    mock_agent_instance.model = mock_model

    mocker.patch("trainer.MLPowerballAgent", return_value=mock_agent_instance)

    trainer = Trainer(model_dir="models")
    trainer.agent = mock_agent_instance
    return trainer


@pytest.fixture
def sample_raw_data():
    """
    Provides a small, raw DataFrame to test preprocessing.
    """
    data = {
        "drawing_date": pd.to_datetime(["2023-01-01", "2023-01-04"]),
        "w1": [1, 10],
        "w2": [2, 11],
        "w3": [3, 12],
        "w4": [4, 13],
        "w5": [5, 14],
        "pb": [6, 15],
    }
    return pd.DataFrame(data)


def test_data_preprocessor_standardize_columns(sample_raw_data):
    """
    Tests that the preprocessor correctly standardizes column names.
    """
    # Arrange
    preprocessor = DataPreprocessor()

    # Act
    processed_df = preprocessor._standardize_columns(sample_raw_data)

    # Assert
    expected_cols = [
        "draw_date",
        "white_ball_1",
        "white_ball_2",
        "white_ball_3",
        "white_ball_4",
        "white_ball_5",
        "powerball",
    ]
    assert all(col in processed_df.columns for col in expected_cols)


def test_data_preprocessor_add_features():
    """
    Tests that the feature engineering step adds new columns.
    """
    # Arrange
    preprocessor = DataPreprocessor()
    data = {
        "draw_date": pd.to_datetime(["2023-01-01", "2023-01-04"]),
        "white_ball_1": [1, 10],
        "white_ball_2": [2, 11],
        "white_ball_3": [3, 12],
        "white_ball_4": [4, 13],
        "white_ball_5": [5, 14],
        "powerball": [6, 15],
    }
    df = pd.DataFrame(data)

    # Act
    featured_df = preprocessor.add_features(df)

    # Assert
    assert "white_balls_sum" in featured_df.columns
    assert "odd_count" in featured_df.columns
    assert "consecutive_count" in featured_df.columns


@patch("trainer.DataPreprocessor.load_historical_data")
def test_start_training_job_success(mock_load_data, mock_trainer):
    """
    Tests the successful orchestration of a training job.
    """
    # Arrange
    model_name = "successful_model"
    data_source = "mock_data.csv"
    # Mock the data loader to return a simple DataFrame
    mock_load_data.return_value = pd.DataFrame(
        {
            "draw_date": pd.to_datetime(["2023-01-01"]),
            "white_ball_1": [1],
            "white_ball_2": [2],
            "white_ball_3": [3],
            "white_ball_4": [4],
            "white_ball_5": [5],
            "powerball": [6],
        }
    )
    # Mock the agent's methods
    # The target tensors (y_white, y_powerball) must be of type torch.long
    mock_trainer.agent.prepare_data.return_value = (
        (torch.randn(1, 3, 5), torch.randn(1, 3, 1)),  # X data
        (
            torch.randint(0, 69, (1, 5), dtype=torch.long),
            torch.randint(0, 26, (1,), dtype=torch.long),
        ),  # y data (targets)
    )
    # Mock the return value of the model call itself to be a dictionary of tensors
    # The output tensors must require gradients to be used in the loss function's backward pass.
    mock_trainer.agent.model.return_value = {
        "white_balls": torch.randn(1, 70, requires_grad=True),
        "powerball": torch.randn(1, 27, requires_grad=True),
    }
    mock_trainer.agent.save_model.return_value = f"models/{model_name}.pth"

    # Act
    job_info = mock_trainer.start_training_job(model_name, data_source)

    # Assert
    assert job_info["status"] == "completed"
    assert job_info["model_name"] == model_name
    assert job_info["progress"] == 100
    assert "final_metrics" in job_info
    # Check that the agent's methods were called
    mock_trainer.agent.prepare_data.assert_called()
    mock_trainer.agent.save_model.assert_called_with(model_name)


@patch("trainer.DataPreprocessor.load_historical_data")
def test_start_training_job_failure(mock_load_data, mock_trainer):
    """
    Tests that the trainer correctly handles a failure during training.
    """
    # Arrange
    model_name = "failed_model"
    data_source = "bad_data.csv"
    # Mock the data loader to raise an exception
    mock_load_data.side_effect = ValueError("Failed to load data")

    # Act & Assert
    with pytest.raises(ValueError, match="Failed to load data"):
        mock_trainer.start_training_job(model_name, data_source)

    # Check that the job status was updated to 'failed'
    job_info = mock_trainer.get_job_status(mock_trainer.training_journal[0]["job_id"])
    assert job_info["status"] == "failed"
    assert job_info["error"] == "Failed to load data"
