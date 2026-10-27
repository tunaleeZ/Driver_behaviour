"""Model construction helpers."""

from __future__ import annotations

from dbm.models.lstm_classifier import LSTMClassifier


def build_lstm(config: dict, input_dim: int):
    """Build the configured shared Geo/landmark LSTM classifier."""
    return LSTMClassifier(input_dim=input_dim, hidden_size=int(config.get("hidden_size", 64)),
                          num_layers=int(config.get("num_layers", 1)),
                          dropout=float(config.get("dropout", 0.3)),
                          pooling=str(config.get("pooling", "mean")))

