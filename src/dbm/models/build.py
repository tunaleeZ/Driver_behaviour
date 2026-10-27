"""Model construction helpers."""

from __future__ import annotations

from dbm.models.lstm_classifier import LSTMClassifier
from dbm.models.cnn_lstm import CNNLSTM


def build_lstm(config: dict, input_dim: int):
    """Build the configured shared Geo/landmark LSTM classifier."""
    return LSTMClassifier(input_dim=input_dim, hidden_size=int(config.get("hidden_size", 64)),
                          num_layers=int(config.get("num_layers", 1)),
                          dropout=float(config.get("dropout", 0.3)),
                          pooling=str(config.get("pooling", "mean")))


def build_cnn_lstm(config: dict) -> CNNLSTM:
    """Build the configured CNN-LSTM frame classifier."""
    return CNNLSTM(channels=config.get("cnn_channels", (32, 64, 128, 256)),
                   input_channels=int(config.get("input_channels", 3)),
                   hidden_size=int(config.get("lstm_hidden", 128)),
                   num_layers=int(config.get("lstm_layers", 1)),
                   lstm_dropout=float(config.get("lstm_dropout", 0.0)),
                   head_dropout=float(config.get("head_dropout", 0.4)),
                   pooling=str(config.get("pooling", "mean")))

