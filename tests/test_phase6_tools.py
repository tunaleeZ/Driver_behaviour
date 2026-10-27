from pathlib import Path
import csv
import json

import numpy as np
import pytest
import torch
import yaml

from dbm.benchmark import benchmark_model, benchmark_extraction, benchmark_extraction_if_available
from dbm.data.datasets import compose_features
from dbm.data.synthetic import generate_synthetic_dataset
from dbm.features.landmark_select import process_frame_landmarks
from dbm.infer_video import infer_video, smooth_probabilities
from dbm.labels import label_names
from dbm.models.build import build_cnn_lstm, build_lstm
from dbm.run_sweep import expand_sweep


def test_quick_sweeps_expand_expected_models():
    repo = Path(__file__).resolve().parents[1]
    configs = sorted((repo / "configs/sweeps").glob("e*.yaml"))
    runs = [item for path in configs for item in expand_sweep(path, quick=True)]
    assert len(runs) == 9
    assert {run["model_type"] for run in runs} == {"lstm_classifier", "cnn_lstm"}
    assert {run["pooling"] for run in runs} == {"last", "mean", "attention"}
    assert all(run["max_epochs"] == 1 for run in runs)


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


class MockFeatureExtractor:
    def __init__(self, payload):
        self.payload = payload

    def extract_from_video(self, video, target_fps=None):
        return self.payload


def test_mock_landmark_and_feature_benchmark_and_missing_resources(tmp_path, caplog):
    root = generate_synthetic_dataset(tmp_path / "benchmark", frames_per_segment=3, size=32, fps=3)
    import pandas as pd
    metadata = pd.read_csv(root / "annotations" / "metadata.csv")
    sid = metadata.iloc[0].segment_id
    with np.load(root / "features" / "landmarks" / f"{sid}.npz") as archive:
        payload = {key: archive[key] for key in archive.files}
    result = benchmark_extraction(root / "videos" / f"{sid}.mp4", MockFeatureExtractor(payload),
                                 {"groups": ["ear", "mar", "d_yaw", "d_pitch"]})
    assert result["frames"] == 3
    assert result["feature_dim"] == 4
    assert result["synthetic"] is True
    assert benchmark_extraction_if_available(root / "missing.mp4", root / "missing.task",
                                             {"groups": ["ear"]}) is None
    assert "thiếu face_landmarker.task" in caplog.text


class MockFrameExtractor:
    def extract_frames_from_video(self, video, input_size, target_fps=None):
        return np.full((5, input_size, input_size, 3), 128, dtype=np.uint8), 5.0


@pytest.mark.parametrize("feature_variant", ["geo", "landmark_lstm"])
def test_geo_and_landmark_video_inference_end_to_end(tmp_path, feature_variant):
    repo = Path(__file__).resolve().parents[1]
    root = generate_synthetic_dataset(tmp_path / "data", frames_per_segment=5, size=32, fps=5)
    metadata = __import__("pandas").read_csv(root / "annotations" / "metadata.csv")
    sid = metadata.iloc[0].segment_id
    with np.load(root / "features" / "landmarks" / f"{sid}.npz") as archive:
        payload = {key: archive[key] for key in archive.files}
    from dbm.train import _load_feature_config
    fc = _load_feature_config(repo, feature_variant)
    dims = compose_features(payload, fc).shape[1]
    config = {"model_type": "lstm_classifier", "feature_config": feature_variant,
              "window_size": 2, "stride": 1, "hidden_size": 4,
              "num_layers": 1, "dropout": 0.0, "pooling": "mean", "synthetic": True}
    model = build_lstm(config, dims)
    run = tmp_path / feature_variant
    run.mkdir()
    torch.save({"model": model.state_dict(), "config": config, "input_dim": dims}, run / "best.pt")
    (run / "stats.json").write_text(json.dumps({"mean": [0.0] * dims, "std": [1.0] * dims}))
    output = infer_video(root / "videos" / f"{sid}.mp4", run / "best.pt", run / "timeline.csv",
                         extractor=MockFeatureExtractor(payload), smoothing="ema", span=2)
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert rows[0]["prediction"] in label_names()
    assert rows[0]["synthetic"] == "True"
    assert len(rows[0]["probabilities"].split()) == len(label_names())


def test_cnn_video_inference_end_to_end_with_mock_extractor(tmp_path):
    root = generate_synthetic_dataset(tmp_path / "cnn", frames_per_segment=5, size=32, fps=5)
    metadata = __import__("pandas").read_csv(root / "annotations" / "metadata.csv")
    sid = metadata.iloc[0].segment_id
    config = {"model_type": "cnn_lstm", "cnn_channels": [4, 8], "input_channels": 3,
              "input_size": 32, "lstm_hidden": 4, "lstm_layers": 1,
              "head_dropout": 0.0, "pooling": "attention", "window_size": 2, "stride": 1,
              "synthetic": True}
    model = build_cnn_lstm(config)
    run = tmp_path / "cnn_run"
    run.mkdir()
    torch.save({"model": model.state_dict(), "config": config}, run / "best.pt")
    (run / "stats.json").write_text(json.dumps({"mean": [0.0] * 3, "std": [1.0] * 3}))
    output = infer_video(root / "videos" / f"{sid}.mp4", run / "best.pt", run / "timeline.csv",
                         extractor=MockFrameExtractor(), smoothing="vote", span=3)
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert rows[0]["prediction"] in label_names()
    assert rows[0]["synthetic"] == "True"
    assert float(rows[-1]["end_seconds"]) >= float(rows[0]["start_seconds"])
