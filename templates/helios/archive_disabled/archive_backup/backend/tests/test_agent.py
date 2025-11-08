# backend/tests/test_agent.py

import pytest
import torch
import pandas as pd
from unittest.mock import MagicMock

from agent import MLPowerballAgent, PowerballNet

@pytest.fixture
def mock_agent(mocker):
    """
    Fixture to create an MLPowerballAgent with a mocked PowerballNet model.
    This allows us to test the agent's logic without running the actual neural network.
    """
    # Mock the PowerballNet model itself
    mock_model_instance = MagicMock(spec=PowerballNet)
    # When the agent calls the model, it will get our mock
    mocker.patch('agent.PowerballNet', return_value=mock_model_instance)
    
    # Create the agent, which will now use the mocked model
    agent = MLPowerballAgent(model_dir='models')
    agent.model = mock_model_instance # Explicitly assign the mock
    return agent

@pytest.fixture
def sample_historical_data():
    """
    Provides a small, consistent sample of historical data for testing.
    """
    data = {
        'draw_date': pd.to_datetime(['2023-01-01', '2023-01-04', '2023-01-07', '2023-01-10']),
        'white_ball_1': [1, 10, 20, 30],
        'white_ball_2': [2, 11, 21, 31],
        'white_ball_3': [3, 12, 22, 32],
        'white_ball_4': [4, 13, 23, 33],
        'white_ball_5': [5, 14, 24, 34],
        'powerball': [6, 15, 25, 26]
    }
    # Create a larger dataset to allow for sequence creation
    df = pd.DataFrame(data)
    df_large = pd.concat([df] * 20, ignore_index=True)
    return df_large

def test_agent_initialization(mock_agent):
    """
    Tests that the agent initializes correctly.
    """
    assert mock_agent is not None
    assert mock_agent.model is not None
    assert mock_agent.metadata['training_completed'] is False

def test_prepare_data(mock_agent, sample_historical_data):
    """
    Tests the data preparation logic to ensure it creates tensors of the correct shape.
    """
    # Arrange
    mock_agent.sequence_length = 3 # Use a small sequence for this test
    
    # Act
    (X_white, X_powerball), (y_white, y_powerball) = mock_agent.prepare_data(sample_historical_data)
    
    # Assert
    # Expected number of sequences = total_draws - sequence_length
    expected_num_sequences = len(sample_historical_data) - mock_agent.sequence_length
    
    assert X_white.shape == (expected_num_sequences, mock_agent.sequence_length, 5)
    assert X_powerball.shape == (expected_num_sequences, mock_agent.sequence_length, 1)
    assert y_white.shape == (expected_num_sequences, 5)
    assert y_powerball.shape == (expected_num_sequences,)
    
    # Check tensor data types
    assert X_white.dtype == torch.long
    assert X_powerball.dtype == torch.long
    assert y_white.dtype == torch.long
    assert y_powerball.dtype == torch.long

def test_predict_logic(mock_agent, sample_historical_data):
    """
    Tests the agent's predict method to ensure it calls the model and formats the output.
    """
    # Arrange
    # Configure the mock model to return a predictable response
    mock_prediction = {
        'white_balls': {'numbers': [[1, 2, 3, 4, 5]], 'probabilities': [[0.9, 0.8, 0.7, 0.6, 0.5]]},
        'powerball': {'numbers': [[10]], 'probabilities': [[0.99]]},
        'confidence': [[0.95]],
        'model_features': [[0.1, 0.2, 0.3]]
    }
    mock_agent.model.predict_next_draw.return_value = mock_prediction
    
    # Act
    result = mock_agent.predict(sample_historical_data)
    
    # Assert
    # Verify that the model's predict_next_draw method was called once
    mock_agent.model.predict_next_draw.assert_called_once()
    
    # Verify that the agent correctly added its own metadata to the result
    assert 'model_info' in result
    assert result['model_info']['architecture'] == 'PowerballNet'
    assert result['white_balls']['numbers'][0] == [1, 2, 3, 4, 5]
    assert result['powerball']['numbers'][0] == [10]
    assert result['confidence'][0] == [0.95]

def test_save_and_load_model(mock_agent, tmp_path, mocker):
    """
    Tests saving and loading a model, mocking the actual torch.save and torch.load.
    """
    # Arrange
    model_name = "test_save_load"
    # Use a temporary directory provided by pytest's tmp_path fixture
    mock_agent.model_dir = tmp_path
    
    # Mock the filesystem and torch functions
    mock_torch_save = mocker.patch('torch.save')
    mock_torch_load = mocker.patch('torch.load', return_value={
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'training_history': [{'epoch': 1, 'loss': 0.5}],
        'metadata': {'total_epochs': 1}
    })
    mocker.patch('os.path.exists', return_value=True)
    
    # Act: Save the model
    save_path = mock_agent.save_model(model_name)
    
    # Assert: Check that torch.save was called correctly
    mock_torch_save.assert_called_once()
    assert save_path == str(tmp_path / f"{model_name}.pth")
    
    # Act: Load the model
    success = mock_agent.load_model(model_name)
    
    # Assert: Check that torch.load was called and state was restored
    assert success is True
    mock_torch_load.assert_called_once()
    assert mock_agent.metadata['total_epochs'] == 1
    assert len(mock_agent.training_history) == 1

def test_load_model_not_found(mock_agent, mocker):
    """
    Tests that loading a non-existent model returns False.
    """
    # Arrange
    mocker.patch('os.path.exists', return_value=False)
    
    # Act
    success = mock_agent.load_model("non_existent_model")
    
    # Assert
    assert success is False
