"""DeepAR+ Time Series Forecasting Pipeline."""

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeepARConfig:
    """Configuration for DeepAR+ model."""

    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    use_attention: bool = True
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    forecast_horizon: int = 30
    context_length: int = 60
    num_samples: int = 100
    loss_function: str = "neg_binomial"
    quantiles: List[float] = None

    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]


class DeepARPipeline:
    """
    DeepAR+ Pipeline for time series forecasting.

    Uses recurrent neural networks to model temporal dependencies
    and generate probabilistic forecasts.
    """

    def __init__(self, config: Optional[DeepARConfig] = None):
        """
        Initialize DeepAR+ pipeline.

        Args:
            config: DeepARConfig object
        """
        self.config = config or DeepARConfig()
        self.model = None
        self.scaler = None
        self.is_trained = False

        logger.info(f"Initialized DeepAR+ pipeline with config: {self.config}")

    def build_model(self, input_dim: int):
        """
        Build DeepAR+ neural network model.

        Args:
            input_dim: Dimension of input features
        """
        self.model = DeepARNet(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            use_attention=self.config.use_attention,
            forecast_horizon=self.config.forecast_horizon,
            num_quantiles=len(self.config.quantiles),
        )

        logger.info(f"Built DeepAR+ model with input_dim={input_dim}")

    def prepare_data(
        self, demand_df: pd.DataFrame, scaler=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.

        Args:
            demand_df: DataFrame with demand data
            scaler: Optional sklearn scaler

        Returns:
            Tuple of (X, y, metadata)
        """
        if scaler is None:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()

        logger.info("Preparing data for DeepAR+ training...")

        # Sort by timestamp
        demand_df = demand_df.sort_values("timestamp")

        # Extract features
        feature_cols = [
            col
            for col in demand_df.columns
            if col not in ["timestamp", "store_id", "product_id", "demand"]
        ]
        X = demand_df[feature_cols].values
        y = demand_df["demand"].values

        # Normalize
        if not hasattr(self, "_scaler_fitted"):
            X = scaler.fit_transform(X)
            self._scaler_fitted = True
        else:
            X = scaler.transform(X)

        self.scaler = scaler

        # Create sequences
        X_seq, y_seq = self._create_sequences(
            X, y, context_length=self.config.context_length
        )

        logger.info(
            f"Prepared sequences: X_seq.shape={X_seq.shape}, y_seq.shape={y_seq.shape}"
        )

        return X_seq, y_seq, demand_df[["timestamp", "store_id", "product_id"]]

    def train(
        self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.1
    ) -> Dict[str, List[float]]:
        """
        Train DeepAR+ model.

        Args:
            X: Input sequences
            y: Target values
            validation_split: Proportion of data for validation

        Returns:
            Dictionary with training history
        """
        if self.model is None:
            self.build_model(X.shape[-1])

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Setup training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting training on device: {device}")

        for epoch in range(self.config.epochs):
            # Training
            train_loss = self._train_epoch(
                X_train, y_train, optimizer, device, self.config.batch_size
            )
            history["train_loss"].append(train_loss)

            # Validation
            val_loss = self._validate_epoch(X_val, y_val, device, self.config.batch_size)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (Best)"
                )
            else:
                patience_counter += 1
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                if patience_counter >= 10:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        self.is_trained = True
        return history

    def predict(
        self,
        X_context: np.ndarray,
        return_intervals: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecasts.

        Args:
            X_context: Context sequences for prediction
            return_intervals: Whether to return prediction intervals

        Returns:
            Dictionary with predictions and optionally quantiles
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        predictions = {
            "point_forecast": [],
            "lower_bound": [],
            "upper_bound": [],
        }

        with torch.no_grad():
            for i in range(0, len(X_context), self.config.batch_size):
                batch = X_context[
                    i : i + self.config.batch_size
                ]  # (batch_size, context_length, input_dim)
                batch_tensor = torch.FloatTensor(batch).to(device)

                # Model outputs quantile predictions
                quantile_preds = self.model(batch_tensor)  # (batch_size, forecast_horizon, num_quantiles)

                # Extract quantiles
                q10_idx = self.config.quantiles.index(0.1)
                q50_idx = self.config.quantiles.index(0.5)
                q90_idx = self.config.quantiles.index(0.9)

                predictions["point_forecast"].append(
                    quantile_preds[:, :, q50_idx].cpu().numpy()
                )
                if return_intervals:
                    predictions["lower_bound"].append(
                        quantile_preds[:, :, q10_idx].cpu().numpy()
                    )
                    predictions["upper_bound"].append(
                        quantile_preds[:, :, q90_idx].cpu().numpy()
                    )

        # Concatenate batches
        for key in predictions:
            predictions[key] = np.concatenate(predictions[key], axis=0)

        logger.info(f"Generated predictions with shape: {predictions['point_forecast'].shape}")

        return predictions

    @staticmethod
    def _create_sequences(
        X: np.ndarray, y: np.ndarray, context_length: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        X_seq, y_seq = [], []

        for i in range(len(X) - context_length):
            X_seq.append(X[i : i + context_length])
            y_seq.append(y[i + context_length])

        return np.array(X_seq), np.array(y_seq)

    def _train_epoch(self, X, y, optimizer, device, batch_size):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for i in range(0, len(X), batch_size):
            batch_X = X[i : i + batch_size]
            batch_y = y[i : i + batch_size]

            batch_X = torch.FloatTensor(batch_X).to(device)
            batch_y = torch.FloatTensor(batch_y).to(device)

            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self._compute_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / (len(X) // batch_size)

    def _validate_epoch(self, X, y, device, batch_size):
        """Validate model."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i : i + batch_size]
                batch_y = y[i : i + batch_size]

                batch_X = torch.FloatTensor(batch_X).to(device)
                batch_y = torch.FloatTensor(batch_y).to(device)

                outputs = self.model(batch_X)
                loss = self._compute_loss(outputs, batch_y)
                total_loss += loss.item()

        return total_loss / (len(X) // batch_size)

    def _compute_loss(self, predictions, targets):
        """Compute loss based on configured loss function."""
        if self.config.loss_function == "mse":
            return torch.nn.MSELoss()(predictions[:, :, 1], targets)
        elif self.config.loss_function == "neg_binomial":
            # Simplified neg_binomial loss
            return torch.nn.MSELoss()(predictions[:, :, 1], targets.unsqueeze(-1))
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")


class DeepARNet(torch.nn.Module):
    """DeepAR+ Neural Network Architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
        forecast_horizon: int = 30,
        num_quantiles: int = 3,
    ):
        """Initialize DeepAR network."""
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.num_quantiles = num_quantiles

        # LSTM layers
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Attention (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = torch.nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=4, batch_first=True
            )

        # Output layers for quantile regression
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, forecast_horizon * num_quantiles),
        )

    def forward(self, x):
        """Forward pass."""
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_len, hidden_dim)

        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out + attn_out  # Residual connection

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Generate forecast
        quantile_logits = self.fc(last_hidden)  # (batch_size, forecast_horizon * num_quantiles)

        # Reshape to (batch_size, forecast_horizon, num_quantiles)
        quantile_preds = quantile_logits.reshape(
            -1, self.forecast_horizon, self.num_quantiles
        )

        return quantile_preds
