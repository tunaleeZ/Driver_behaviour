"""CPU tests for phase 4 sequence modeling primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbm.data.stats import fit_feature_stats, normalize_features
from dbm.data.windows import make_windows
from dbm.losses import CostSensitiveLoss, FocalLoss, class_weights_from_labels
from dbm.metrics import aggregate_segment_probabilities, classification_metrics
from dbm.models.lstm_classifier import LSTMClassifier
from dbm.models.pooling import TemporalPool


def test_window_edges_and_short_policy():
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    windows = make_windows(values, "SEG000001", 0, 4, 3)
    assert [window.start for window in windows] == [0, 3, 6]
    assert all(window.values.shape == (4, 2) for window in windows)
    padded = make_windows(values[:2], "SEG000001", 0, 4, 2)[0]
    np.testing.assert_array_equal(padded.values[-1], values[1])
    assert padded.padded
    assert make_windows(values[:2], "SEG000001", 0, 4, 2, "drop") == []


def test_stats_fit_from_given_training_arrays_only():
    stats = fit_feature_stats([np.array([[1., 2.], [3., 4.]])])
    np.testing.assert_allclose(stats["mean"], [2, 3])
    np.testing.assert_allclose(normalize_features(np.array([[3., 4.]]), stats), [[1, 1]])


@pytest.mark.parametrize("mode", ["last", "mean", "attention"])
def test_pooling_shapes_and_attention(mode):
    pool = TemporalPool(mode, 128)
    pooled, alpha = pool(torch.randn(3, 7, 128))
    assert pooled.shape == (3, 128)
    if mode == "attention":
        assert alpha.shape == (3, 7)
        torch.testing.assert_close(alpha.sum(1), torch.ones(3))


@pytest.mark.parametrize("dim", [4, 70])
def test_lstm_geo_and_landmark_shapes(dim):
    model = LSTMClassifier(dim, pooling="mean")
    assert model(torch.randn(2, 30, dim)).shape == (2, 5)


def test_losses_and_metrics():
    logits = torch.tensor([[2., 0.], [0., 2.]])
    labels = torch.tensor([0, 1])
    ce = torch.nn.functional.cross_entropy(logits, labels)
    torch.testing.assert_close(FocalLoss(gamma=0)(logits, labels), ce)
    matrix = torch.tensor([[0., 1.], [2., 0.]])
    loss = CostSensitiveLoss(matrix, weight=1.0)(logits, labels)
    expected_cost = ((logits.softmax(1) * matrix[labels]).sum(1)).mean()
    torch.testing.assert_close(loss, ce + expected_cost)
    assert class_weights_from_labels([0, 0, 1], 2).shape == (2,)
    report = classification_metrics([0, 1], [0, 0], 2)
    assert report["confusion_matrix"] == [[1, 0], [1, 0]]
    ids, probs, y = aggregate_segment_probabilities(["a", "a", "b"], [[.8,.2],[.4,.6],[.1,.9]], [0,0,1])
    assert ids == ["a", "b"]
    np.testing.assert_allclose(probs[0], [.6, .4])
    np.testing.assert_array_equal(y, [0, 1])

