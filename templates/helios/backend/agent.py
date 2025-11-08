"""
Helios Agent - Neural Network Architecture for Powerball Analysis
Implements PowerballNet and MLPowerballAgent classes for intelligent lottery analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class PowerballNet(nn.Module):
    """
    Neural network architecture specifically designed for Powerball lottery analysis.
    Features:
    - Separate pathways for white balls (1-69) and Powerball (1-26)
    - Attention mechanism for pattern recognition
    - Temporal encoding for historical sequences
    - Multi-head output for different prediction strategies
    """

    def __init__(
        self,
        sequence_length: int = 50,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        attention_heads: int = 8,
    ):
        super(PowerballNet, self).__init__()

        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention_heads = attention_heads

        # Input dimensions
        self.white_ball_vocab = 69  # 1-69
        self.powerball_vocab = 26  # 1-26

        # Embedding layers
        self.white_ball_embedding = nn.Embedding(
            self.white_ball_vocab + 1, hidden_dim // 2
        )
        self.powerball_embedding = nn.Embedding(
            self.powerball_vocab + 1, hidden_dim // 4
        )

        # Positional encoding for temporal information
        self.positional_encoding = nn.Parameter(
            torch.randn(sequence_length, hidden_dim)
        )

        # Multi-head attention for pattern recognition
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # LSTM layers for sequence processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )

        # Feature processing layers
        self.feature_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads for different prediction strategies (need +1 for 0-indexed)
        self.white_ball_head = nn.Linear(hidden_dim // 2, self.white_ball_vocab + 1)
        self.powerball_head = nn.Linear(hidden_dim // 2, self.powerball_vocab + 1)

        # Frequency analysis head
        self.frequency_head = nn.Linear(hidden_dim // 2, 64)

        # Pattern confidence head
        self.confidence_head = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self, white_balls: torch.Tensor, powerball: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            white_balls: Tensor of shape (batch_size, sequence_length, 5) - white ball numbers
            powerball: Tensor of shape (batch_size, sequence_length, 1) - powerball numbers

        Returns:
            Dictionary containing predictions for different heads
        """
        batch_size, seq_len = white_balls.shape[:2]

        # Embed white balls (sum across the 5 balls per draw)
        white_embedded = self.white_ball_embedding(white_balls).sum(
            dim=2
        )  # (batch, seq, hidden//2)

        # Embed powerball
        powerball_embedded = self.powerball_embedding(
            powerball.squeeze(-1)
        )  # (batch, seq, hidden//4)

        # Combine embeddings
        # Pad powerball embedding to match white ball embedding size
        powerball_padded = F.pad(powerball_embedded, (0, self.hidden_dim // 4))
        combined = torch.cat(
            [white_embedded, powerball_padded], dim=-1
        )  # (batch, seq, hidden)

        # Add positional encoding
        combined = combined + self.positional_encoding[:seq_len].unsqueeze(0)

        # Apply attention
        attended, attention_weights = self.attention(combined, combined, combined)

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(attended)

        # Use the last output for prediction
        last_output = lstm_out[:, -1, :]  # (batch, hidden*2)

        # Feature processing
        features = self.feature_layers(last_output)  # (batch, hidden//2)

        # Generate predictions from different heads
        outputs = {
            "white_balls": self.white_ball_head(features),
            "powerball": self.powerball_head(features),
            "frequency_features": self.frequency_head(features),
            "confidence": torch.sigmoid(self.confidence_head(features)),
            "attention_weights": attention_weights,
            "features": features,
        }

        return outputs

    def predict_next_draw(
        self, white_balls: torch.Tensor, powerball: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Generate predictions for the next draw with interpretable results.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(white_balls, powerball)

            # Get top predictions for white balls
            white_probs = F.softmax(outputs["white_balls"], dim=-1)
            white_top5 = torch.topk(white_probs, k=5, dim=-1)

            # Get top prediction for powerball
            powerball_probs = F.softmax(outputs["powerball"], dim=-1)
            powerball_top = torch.topk(powerball_probs, k=3, dim=-1)

            predictions = {
                "white_balls": {
                    "numbers": (white_top5.indices + 1)
                    .cpu()
                    .numpy()
                    .tolist(),  # +1 for 1-based indexing
                    "probabilities": white_top5.values.cpu().numpy().tolist(),
                },
                "powerball": {
                    "numbers": (powerball_top.indices + 1).cpu().numpy().tolist(),
                    "probabilities": powerball_top.values.cpu().numpy().tolist(),
                },
                "confidence": outputs["confidence"].cpu().numpy().tolist(),
                "model_features": outputs["features"].cpu().numpy().tolist(),
            }

        return predictions


class MLPowerballAgent:
    """
    Intelligent agent for Powerball lottery analysis with persistent memory.
    Manages model lifecycle, training, and prediction generation.
    """

    def __init__(
        self,
        model_dir: str = "models",
        sequence_length: int = 50,
        device: Optional[str] = None,
    ):
        """
        Initialize the ML Powerball Agent.

        Args:
            model_dir: Directory to store model artifacts
            sequence_length: Number of historical draws to consider
            device: PyTorch device ('cpu', 'cuda', etc.)
        """
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Initialize model
        self.model = PowerballNet(sequence_length=sequence_length)
        self.model.to(self.device)

        # Training state
        self.optimizer = None
        self.criterion = None
        self.training_history = []

        # Model metadata
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "architecture": "PowerballNet",
            "sequence_length": sequence_length,
            "device": self.device,
            "training_completed": False,
            "total_epochs": 0,
            "best_loss": float("inf"),
        }

        logger.info(f"MLPowerballAgent initialized on device: {self.device}")

    def prepare_data(
        self, historical_data: pd.DataFrame
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare historical lottery data for training.

        Args:
            historical_data: DataFrame with columns ['draw_date', 'white_ball_1', ..., 'white_ball_5', 'powerball']

        Returns:
            Tuple of ((X_white, X_powerball), (y_white, y_powerball))
        """
        logger.info(f"Preparing data from {len(historical_data)} historical draws")

        # Extract white balls and powerball
        white_balls = historical_data[
            [
                "white_ball_1",
                "white_ball_2",
                "white_ball_3",
                "white_ball_4",
                "white_ball_5",
            ]
        ].values
        powerball = historical_data["powerball"].values

        # Create sequences
        sequences_white = []
        sequences_powerball = []
        targets_white = []
        targets_powerball = []

        for i in range(len(historical_data) - self.sequence_length):
            # Input sequence
            seq_white = white_balls[i : i + self.sequence_length]
            seq_powerball = powerball[i : i + self.sequence_length]

            # Target (next draw)
            target_white = white_balls[i + self.sequence_length]
            target_powerball = powerball[i + self.sequence_length]

            sequences_white.append(seq_white)
            sequences_powerball.append(seq_powerball)
            targets_white.append(target_white)
            targets_powerball.append(target_powerball)

        # Convert to tensors
        X_white = torch.tensor(np.array(sequences_white), dtype=torch.long).to(
            self.device
        )
        X_powerball = (
            torch.tensor(np.array(sequences_powerball), dtype=torch.long)
            .unsqueeze(-1)
            .to(self.device)
        )
        y_white = torch.tensor(np.array(targets_white), dtype=torch.long).to(
            self.device
        )
        y_powerball = torch.tensor(np.array(targets_powerball), dtype=torch.long).to(
            self.device
        )

        logger.info(f"Created {len(sequences_white)} training sequences")
        return (X_white, X_powerball), (y_white, y_powerball)

    def train(
        self,
        historical_data: pd.DataFrame,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ):
        """
        Train the model on historical lottery data.
        """
        logger.info(f"Starting training for {epochs} epochs")

        # Prepare data
        (X_white, X_powerball), (y_white, y_powerball) = self.prepare_data(
            historical_data
        )

        # Initialize training components
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_white, X_powerball)

            # Calculate losses for each ball position
            white_losses = []
            for i in range(5):  # 5 white balls
                white_loss = self.criterion(outputs["white_balls"], y_white[:, i])
                white_losses.append(white_loss)

            powerball_loss = self.criterion(outputs["powerball"], y_powerball)

            # Combined loss
            total_loss = sum(white_losses) + powerball_loss

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Track progress
            epoch_loss = total_loss.item()
            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            if epoch_loss < self.metadata["best_loss"]:
                self.metadata["best_loss"] = epoch_loss

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Update metadata
        self.metadata["training_completed"] = True
        self.metadata["total_epochs"] = epochs
        self.metadata["final_loss"] = self.training_history[-1]["loss"]

        logger.info("Training completed successfully")

    def predict(self, historical_data: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate predictions for the next lottery draw.
        """
        logger.info("Generating predictions for next draw")

        # Get the most recent sequence
        recent_data = historical_data.tail(self.sequence_length)
        (X_white, X_powerball), _ = self.prepare_data(
            pd.concat([recent_data, recent_data.iloc[-1:]], ignore_index=True)
        )

        # Generate predictions
        predictions = self.model.predict_next_draw(X_white[-1:], X_powerball[-1:])

        # Add metadata
        predictions["model_info"] = {
            "architecture": self.metadata["architecture"],
            "training_epochs": self.metadata["total_epochs"],
            "best_loss": self.metadata["best_loss"],
            "prediction_timestamp": datetime.now().isoformat(),
        }

        return predictions

    def save_model(self, model_name: str) -> str:
        """
        Save the trained model and metadata to disk.

        Returns:
            Path to the saved model file
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.pth")
        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")

        # Save model state
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
                if self.optimizer
                else None,
                "training_history": self.training_history,
                "metadata": self.metadata,
            },
            model_path,
        )

        # Save metadata separately for easy access
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        return model_path

    def load_model(self, model_name: str) -> bool:
        """
        Load a previously trained model from disk.

        Returns:
            True if successful, False otherwise
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.pth")

        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False

        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Load training history and metadata
            self.training_history = checkpoint.get("training_history", [])
            self.metadata = checkpoint.get("metadata", self.metadata)

            # Restore optimizer if available
            if checkpoint.get("optimizer_state_dict") and self.optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            logger.info(f"Model loaded successfully from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current model.
        """
        return {
            "metadata": self.metadata,
            "training_history": self.training_history,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": self.device,
            "memory_usage": torch.cuda.memory_allocated()
            if torch.cuda.is_available()
            else 0,
        }

    @classmethod
    def list_saved_models(cls, model_dir: str = "models") -> List[Dict[str, Any]]:
        """
        List all saved models in the model directory.
        """
        if not os.path.exists(model_dir):
            return []

        models = []
        for filename in os.listdir(model_dir):
            if filename.endswith(".pth"):
                model_name = filename[:-4]  # Remove .pth extension
                metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")

                model_info = {"name": model_name, "file": filename}

                # Load metadata if available
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        model_info.update(metadata)
                    except Exception as e:
                        logger.warning(
                            f"Could not load metadata for {model_name}: {str(e)}"
                        )

                models.append(model_info)

        return sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)
