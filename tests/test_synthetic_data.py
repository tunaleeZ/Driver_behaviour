import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from dbm.data.synthetic import generate_synthetic_dataset
from dbm.evaluate import evaluate
from dbm.labels import label_names
from dbm.models.lstm_classifier import LSTMClassifier
from dbm.train import train, append_result_summary


def test_synthetic_generator_writes_marked_features_frames_and_video(tmp_path):
    root = generate_synthetic_dataset(tmp_path / "fixture", samples_per_class=3,
                                     frames_per_segment=3, size=32, fps=3)
    metadata = pd.read_csv(root / "annotations" / "metadata.csv")
    assert len(metadata) == 3 * len(label_names())
    assert metadata.synthetic.all()
    assert (root / "SYNTHETIC_DATASET.txt").exists()
    first = metadata.iloc[0].segment_id
    with np.load(root / "features" / "landmarks" / f"{first}.npz") as data:
        assert data["synthetic"].item()
        assert data["geo"].shape == (3, 7)
    assert len(list((root / "processed" / "frames" / first).glob("*.jpg"))) == 3
    import cv2
    cap = cv2.VideoCapture(str(root / "videos" / f"{first}.mp4"))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    cap.release()


def test_lstm_overfits_tiny_synthetic_feature_set():
    torch.manual_seed(4)
    classes, steps, width = 5, 4, 7
    x = torch.zeros(classes, steps, width)
    for label in range(classes):
        x[label, :, label] = 2.0
    y = torch.arange(classes)
    model = LSTMClassifier(input_dim=width, hidden_size=12, num_layers=1, dropout=0.0, pooling="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    with torch.no_grad():
        initial_loss = torch.nn.functional.cross_entropy(model(x), y).item()
    for _ in range(90):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_logits = model(x)
        assert torch.nn.functional.cross_entropy(final_logits, y).item() < initial_loss
        assert (final_logits.argmax(1) == y).all()


def test_cnn_lstm_cpu_smoke_train_and_evaluate(tmp_path):
    root = generate_synthetic_dataset(tmp_path / "cnn_data", samples_per_class=3,
                                     frames_per_segment=2, size=32, fps=2)
    config_path = tmp_path / "cnn.yaml"
    config_path.write_text(yaml.safe_dump({
        "model_type": "cnn_lstm", "cnn_channels": [4, 8], "input_size": 32,
        "input_channels": 3, "lstm_hidden": 4, "lstm_layers": 1,
        "head_dropout": 0.0, "pooling": "mean", "window_size": 2,
        "stride": 2, "short_segment_policy": "pad", "batch_size": 5,
        "num_workers": 0, "lr": 0.001, "weight_decay": 0.0,
        "max_epochs": 1, "early_stopping_patience": 1,
        "scheduler": "cosine", "loss": "cross_entropy", "seed": 4,
        "use_amp": False,
    }), encoding="utf-8")
    out = train(config_path, root=root, work_root=root, run_name=f"_smoke/cnn_test_{tmp_path.name}")
    assert (out / "best.pt").is_file()
    import json
    assert json.loads((out / "results.json").read_text()) ["synthetic"] is True
    result = evaluate(out / "best.pt", root=root, work_root=root, split_name="test", batch_size=5)
    assert result["window"]["n_samples"] == 5


@pytest.mark.parametrize("feature_variant", ["geo", "landmark_lstm"])
def test_sequence_model_smoke_train_and_evaluate(tmp_path, feature_variant):
    root = generate_synthetic_dataset(tmp_path / f"{feature_variant}_data", samples_per_class=3,
                                     frames_per_segment=3, size=32, fps=3)
    config_path = tmp_path / f"{feature_variant}.yaml"
    config_path.write_text(yaml.safe_dump({
        "model_type": "lstm_classifier", "feature_config": feature_variant,
        "hidden_size": 4, "num_layers": 1, "dropout": 0.0, "pooling": "mean",
        "window_size": 2, "stride": 1, "short_segment_policy": "pad",
        "batch_size": 5, "num_workers": 0, "lr": 0.01, "max_epochs": 1,
        "early_stopping_patience": 1, "scheduler": "cosine", "loss": "cross_entropy",
        "seed": 7, "use_amp": False,
    }), encoding="utf-8")
    out = train(config_path, root=root, work_root=root,
                run_name=f"_smoke/{feature_variant}_{tmp_path.name}")
    result = evaluate(out / "best.pt", root=root, work_root=root, split_name="test", batch_size=5)
    assert result["window"]["n_samples"] == 10
    assert result["synthetic"] is True
    assert (out / "confusion_matrix.png").is_file()


def test_synthetic_results_cannot_enter_summary_csv(tmp_path):
    target = tmp_path / "results_summary.csv"
    import pytest
    with pytest.raises(ValueError, match="Synthetic runs are forbidden"):
        append_result_summary({"synthetic": True, "accuracy": 1.0}, target)
    assert not target.exists()
