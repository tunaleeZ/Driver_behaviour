from pathlib import Path

import numpy as np

from dbm.benchmark import benchmark_model
from dbm.infer_video import smooth_probabilities
from dbm.run_sweep import expand_sweep


def test_quick_sweeps_expand_expected_models():
    repo = Path(__file__).resolve().parents[1]
    configs = sorted((repo / "configs/sweeps").glob("e*.yaml"))
    runs = [item for path in configs for item in expand_sweep(path, quick=True)]
    assert len(runs) == 3
    assert {run["model_type"] for run in runs} == {"lstm_classifier", "cnn_lstm"}
    assert all(run["max_epochs"] == 2 for run in runs)


def test_tensor_benchmark_returns_synthetic_diagnostics():
    result = benchmark_model({"window_size": 2, "input_dim": 4, "hidden_size": 4}, steps=2, warmup=1, device="cpu")
    assert result["synthetic"] is True
    assert result["steps"] == 2
    assert result["latency_ms_p95"] >= 0


def test_probability_smoothing_methods():
    probs = np.asarray([[.9, .1], [.1, .9], [.1, .9]], dtype=np.float32)
    assert smooth_probabilities(probs, "vote", 2).shape == probs.shape
    smoothed = smooth_probabilities(probs, "ema", 2)
    assert smoothed.shape == probs.shape
    assert np.allclose(smoothed.sum(axis=1), 1.0)
